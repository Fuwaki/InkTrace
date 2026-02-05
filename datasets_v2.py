"""
InkTrace Dense Dataset v2

直接使用 Rust 生成 Dense GT Maps，无需 Python 端后处理。
支持渐进式训练（Curriculum Learning）：
  - Stage 0: 单笔画
  - Stage 1-3: 多独立笔画（递增）
  - Stage 4-6: 多段连续笔画（递增）
  - Stage 7-9: 混合模式（多条多段路径）

输出格式与 DenseHeads 模型对齐:
  - skeleton:   [B, 1, H, W] 骨架
  - keypoints:  [B, 2, H, W] 关键点
      - Ch0: 拓扑节点 - 样条曲线的起点/终点 (MUST break)
      - Ch1: 几何锚点 - 多段贝塞尔曲线之间的连接点 (SHOULD break)
  - tangent:    [B, 2, H, W] 切向场 (cos2θ, sin2θ)
  - width:      [B, 1, H, W] 宽度
  - offset:     [B, 2, H, W] 亚像素偏移
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import ink_trace_rs
import math


class DenseInkTraceDataset(IterableDataset):
    """
    Dense Prediction 训练用数据集 (V2 - 使用 Rust 直接生成 Dense Maps)

    输出:
        - image: [1, H, W] 输入图像
        - targets: dict 包含 Dense GT Maps
            - skeleton:  [1, H, W]
            - keypoints: [2, H, W] - 拓扑节点 + 几何锚点
            - tangent:   [2, H, W]
            - width:     [1, H, W]
            - offset:    [2, H, W]

    训练阶段 (Curriculum):
        Stage 0: 单笔画
        Stage 1-3: 多独立笔画 (递增: 1-3, 2-5, 3-8)
        Stage 4-6: 多段连续笔画 (递增: 2-3, 3-5, 4-8)
        Stage 7-9: 混合模式 (多条多段路径)
    """

    def __init__(
        self,
        img_size: int = 64,
        batch_size: int = 64,
        epoch_length: int = 10000,
        curriculum_stage: int = 0,
        rust_threads: int = None,
    ):
        """
        Args:
            img_size: 图像尺寸 (64, 128 等)
            batch_size: Rust 批量生成大小
            epoch_length: 每个 epoch 的样本数
            curriculum_stage: 训练阶段 (0-9)
            rust_threads: Rayon 线程数 (None 表示自动)
        """
        super().__init__()
        self.img_size = img_size
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.curriculum_stage = curriculum_stage
        self.rust_threads = rust_threads

        # 获取阶段信息
        self.stage_info = ink_trace_rs.get_stage_info(curriculum_stage)

    def set_curriculum(self, stage: int):
        """切换训练阶段"""
        self.curriculum_stage = stage
        self.stage_info = ink_trace_rs.get_stage_info(stage)
        print(
            f"[Curriculum] Stage {stage}: {self.stage_info['name']} "
            f"({self.stage_info['mode']}) - max_strokes: {self.stage_info['max_strokes']}"
        )

    def _generate_batch(self):
        """使用 Rust 生成一批数据"""
        return ink_trace_rs.generate_dense_batch(
            self.batch_size, self.img_size, self.curriculum_stage
        )

    def __iter__(self):
        # 配置线程
        worker_info = torch.utils.data.get_worker_info()
        target_threads = self.rust_threads
        if target_threads is None and worker_info is not None:
            target_threads = 1

        if target_threads is not None:
            try:
                ink_trace_rs.set_rayon_threads(target_threads)
            except RuntimeError:
                pass  # 已经初始化过了

        # 计算此 worker 需要生成的样本数
        if worker_info is None:
            total_items = self.epoch_length
        else:
            per_worker = int(
                math.ceil(self.epoch_length / float(worker_info.num_workers))
            )
            total_items = per_worker

        generated_count = 0

        while generated_count < total_items:
            # 从 Rust 获取一批数据
            batch = self._generate_batch()
            curr_batch_len = batch["image"].shape[0]

            for i in range(curr_batch_len):
                if generated_count >= total_items:
                    break

                # 转换为 PyTorch tensors
                img_tensor = torch.from_numpy(batch["image"][i]).unsqueeze(0)

                targets = {
                    "skeleton": torch.from_numpy(batch["skeleton"][i]).unsqueeze(0),
                    "keypoints": torch.from_numpy(
                        batch["keypoints"][i].copy()
                    ),  # [2, H, W]
                    "tangent": torch.from_numpy(
                        batch["tangent"][i].copy()
                    ),  # [2, H, W]
                    "width": torch.from_numpy(batch["width"][i]).unsqueeze(0),
                    "offset": torch.from_numpy(batch["offset"][i].copy()),  # [2, H, W]
                }

                yield img_tensor, targets
                generated_count += 1

    def __len__(self):
        return self.epoch_length


def collate_dense_batch(batch):
    """自定义 collate 函数"""
    imgs, targets_list = zip(*batch)

    imgs = torch.stack(imgs)

    targets = {
        "skeleton": torch.stack([t["skeleton"] for t in targets_list]),
        "keypoints": torch.stack([t["keypoints"] for t in targets_list]),
        "tangent": torch.stack([t["tangent"] for t in targets_list]),
        "width": torch.stack([t["width"] for t in targets_list]),
        "offset": torch.stack([t["offset"] for t in targets_list]),
    }

    return imgs, targets


def create_dense_dataloader(
    img_size: int = 64,
    batch_size: int = 32,
    epoch_length: int = 10000,
    curriculum_stage: int = 0,
    num_workers: int = 4,
):
    """创建 Dense 训练用 DataLoader

    Args:
        img_size: 图像尺寸
        batch_size: DataLoader 批量大小
        epoch_length: 每个 epoch 的样本数
        curriculum_stage: 训练阶段 (0-9)
        num_workers: DataLoader worker 数量

    Returns:
        (dataloader, dataset) 元组
    """
    dataset = DenseInkTraceDataset(
        img_size=img_size,
        batch_size=batch_size,  # Rust 内部批量大小
        epoch_length=epoch_length,
        curriculum_stage=curriculum_stage,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_dense_batch,
        pin_memory=True,
    )

    return dataloader, dataset


def list_curriculum_stages():
    """列出所有可用的训练阶段"""
    stages = ink_trace_rs.list_stages()
    print("可用训练阶段:")
    print("-" * 60)
    for info in stages:
        print(
            f"  Stage {info['stage']:2d}: {info['name']:20s} "
            f"({info['mode']:30s}) max={info['max_strokes']}"
        )
    print("-" * 60)
    return stages


# =============================================================================
# 快速测试
# =============================================================================

if __name__ == "__main__":
    print("测试 DenseInkTraceDataset...")

    # 测试单个阶段
    dataset = DenseInkTraceDataset(
        img_size=64,
        batch_size=8,
        epoch_length=16,
        curriculum_stage=0,
    )

    # 列出所有阶段
    list_curriculum_stages()

    # 测试迭代
    count = 0
    for img, targets in dataset:
        count += 1
        if count == 1:
            print(f"\n样本形状:")
            print(f"  image:     {img.shape}")
            print(f"  skeleton:  {targets['skeleton'].shape}")
            print(f"  keypoints: {targets['keypoints'].shape}")
            print(f"  tangent:   {targets['tangent'].shape}")
            print(f"  width:     {targets['width'].shape}")
            print(f"  offset:    {targets['offset'].shape}")

            # 检查 keypoints 的两个通道
            topo_sum = targets["keypoints"][0].sum().item()
            geom_sum = targets["keypoints"][1].sum().item()
            print(f"\n  keypoints Ch0 (topo): sum={topo_sum:.1f}")
            print(f"  keypoints Ch1 (geom): sum={geom_sum:.1f}")

        if count >= 4:
            break

    print(f"\n迭代了 {count} 个样本")

    # 测试 DataLoader
    print("\n测试 DataLoader...")
    dataloader, _ = create_dense_dataloader(
        img_size=64,
        batch_size=4,
        epoch_length=16,
        curriculum_stage=0,
        num_workers=0,
    )

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        print(
            f"  Batch {batch_idx}: imgs={imgs.shape}, keypoints={targets['keypoints'].shape}"
        )
        if batch_idx >= 1:
            break

    print("\n✓ 测试完成!")
