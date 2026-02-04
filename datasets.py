"""
InkTrace Dense Dataset

封装 Rust 数据生成 + Python Dense GT 生成，为模型训练提供统一接口。
支持渐进式训练（Curriculum Learning）：分辨率/笔画数递增。
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import ink_trace_rs
import math

from dense_gen import batch_generate_dense_maps


class DenseInkTraceDataset(IterableDataset):
    """
    Dense Prediction 训练用数据集。

    输出:
        - image: [1, H, W] 输入图像 (带噪声/纹理的墨迹)
        - targets: dict 包含 Dense GT Maps
            - skeleton: [1, H, W]
            - junction: [1, H, W]
            - tangent:  [2, H, W]
            - width:    [1, H, W]
            - offset:   [2, H, W]
        - labels: [N, 11] 原始矢量参数 (用于验证/可视化)

    渐进训练支持:
        通过 set_curriculum() 动态调整:
        - img_size: 分辨率 (32 -> 64 -> 128)
        - max_strokes: 最大笔画数 (1 -> 3 -> 5 -> 8)
        - stroke_range: 笔画数范围 (min, max)
    """

    def __init__(
        self,
        mode="independent",
        img_size=64,
        batch_size=64,
        epoch_length=10000,
        max_strokes=5,
        stroke_range=None,
        curriculum_stage=0,
        seed=None,
        rust_threads=None,
        return_vector_labels=False,
    ):
        super().__init__()
        self.mode = mode
        self.img_size = img_size
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.max_strokes = max_strokes
        self.stroke_range = stroke_range
        self.curriculum_stage = curriculum_stage
        self.rust_threads = rust_threads
        self.return_vector_labels = return_vector_labels

        self.curriculum_configs = {
            0: (64, 1, (1, 1)),
            1: (64, 3, (1, 3)),
            2: (64, 5, (2, 5)),
            3: (64, 8, (3, 8)),
            4: (128, 8, (3, 8)),
            5: (128, 12, (5, 12)),
        }

        self._apply_curriculum()

    def _apply_curriculum(self):
        if self.curriculum_stage in self.curriculum_configs:
            cfg = self.curriculum_configs[self.curriculum_stage]
            self.img_size = cfg[0]
            self.max_strokes = cfg[1]
            self.stroke_range = cfg[2]

    def set_curriculum(self, stage: int):
        self.curriculum_stage = stage
        self._apply_curriculum()
        print(
            f"[Curriculum] Stage {stage}: img_size={self.img_size}, "
            f"max_strokes={self.max_strokes}, range={self.stroke_range}"
        )

    def set_custom_curriculum(
        self, img_size: int, max_strokes: int, stroke_range: tuple
    ):
        self.img_size = img_size
        self.max_strokes = max_strokes
        self.stroke_range = stroke_range
        print(
            f"[Curriculum] Custom: img_size={img_size}, "
            f"max_strokes={max_strokes}, range={stroke_range}"
        )

    def _generate_batch(self):
        fixed_count = None
        if self.stroke_range is not None:
            fixed_count = np.random.randint(
                self.stroke_range[0], self.stroke_range[1] + 1
            )

        if self.mode == "single":
            return ink_trace_rs.generate_single_stroke_batch(
                self.batch_size, self.img_size
            )
        elif self.mode == "independent":
            return ink_trace_rs.generate_independent_strokes_batch(
                self.batch_size, self.img_size, self.max_strokes, fixed_count
            )
        elif self.mode == "continuous":
            return ink_trace_rs.generate_continuous_strokes_batch(
                self.batch_size, self.img_size, self.max_strokes
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        target_threads = None
        if self.rust_threads is not None:
            target_threads = self.rust_threads
        elif worker_info is not None:
            target_threads = 1

        if target_threads is not None:
            try:
                if hasattr(ink_trace_rs, "set_rayon_threads"):
                    ink_trace_rs.set_rayon_threads(target_threads)
            except RuntimeError:
                pass

        if worker_info is None:
            total_items = self.epoch_length
        else:
            per_worker = int(
                math.ceil(self.epoch_length / float(worker_info.num_workers))
            )
            total_items = per_worker

        generated_count = 0

        while generated_count < total_items:
            try:
                imgs, labels = self._generate_batch()
            except Exception as e:
                print(f"Rust generation failed: {e}")
                raise e

            imgs = np.array(imgs)
            labels = np.array(labels)
            curr_batch_len = len(imgs)

            dense_maps = batch_generate_dense_maps(labels, img_size=self.img_size)

            for i in range(curr_batch_len):
                if generated_count >= total_items:
                    break

                img_tensor = torch.from_numpy(imgs[i]).unsqueeze(0)

                targets = {
                    "skeleton": torch.from_numpy(dense_maps["skeleton"][i]),
                    "junction": torch.from_numpy(dense_maps["junction"][i]),
                    "tangent": torch.from_numpy(dense_maps["tangent"][i]),
                    "width": torch.from_numpy(dense_maps["width"][i]),
                    "offset": torch.from_numpy(dense_maps["offset"][i]),
                }

                if self.return_vector_labels:
                    yield img_tensor, targets, torch.from_numpy(labels[i].copy())
                else:
                    yield img_tensor, targets

                generated_count += 1

    def __len__(self):
        return self.epoch_length


def collate_dense_batch(batch):
    """自定义 collate 函数"""
    has_labels = len(batch[0]) == 3

    if has_labels:
        imgs, targets_list, labels = zip(*batch)
    else:
        imgs, targets_list = zip(*batch)
        labels = None

    imgs = torch.stack(imgs)

    targets = {
        "skeleton": torch.stack([t["skeleton"] for t in targets_list]),
        "junction": torch.stack([t["junction"] for t in targets_list]),
        "tangent": torch.stack([t["tangent"] for t in targets_list]),
        "width": torch.stack([t["width"] for t in targets_list]),
        "offset": torch.stack([t["offset"] for t in targets_list]),
    }

    if labels is not None:
        labels = torch.stack(labels)
        return imgs, targets, labels
    else:
        return imgs, targets


def create_dense_dataloader(
    mode="independent",
    img_size=64,
    batch_size=32,
    epoch_length=10000,
    max_strokes=5,
    stroke_range=None,
    curriculum_stage=0,
    num_workers=4,
    return_vector_labels=False,
):
    """创建 Dense 训练用 DataLoader"""
    dataset = DenseInkTraceDataset(
        mode=mode,
        img_size=img_size,
        batch_size=batch_size,
        epoch_length=epoch_length,
        max_strokes=max_strokes,
        stroke_range=stroke_range,
        curriculum_stage=curriculum_stage,
        return_vector_labels=return_vector_labels,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_dense_batch,
        pin_memory=True,
    )

    return dataloader, dataset


# =============================================================================
# Legacy (旧版兼容)
# =============================================================================


class InkTraceDataset(IterableDataset):
    """[Legacy] 旧版数据集，保留用于旧训练脚本兼容"""

    def __init__(
        self,
        mode="single",
        img_size=64,
        batch_size=64,
        epoch_length=10000,
        max_strokes=None,
        fixed_count=None,
        max_paths=None,
        max_segments=None,
        seed=None,
        rust_threads=None,
        for_detr=False,
    ):
        super().__init__()
        self.mode = mode
        self.img_size = img_size
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.max_strokes = max_strokes
        self.fixed_count = fixed_count
        self.max_paths = max_paths
        self.max_segments = max_segments
        self.rust_threads = rust_threads
        self.for_detr = for_detr
        self._validate_params()

    def _validate_params(self):
        if self.mode == "independent":
            if self.max_strokes is None:
                raise ValueError("Mode 'independent' requires 'max_strokes'")
        elif self.mode == "continuous":
            if self.max_segments is None:
                raise ValueError("Mode 'continuous' requires 'max_segments'")
        elif self.mode == "multi_connected":
            if self.max_paths is None or self.max_segments is None:
                raise ValueError(
                    "Mode 'multi_connected' requires 'max_paths' and 'max_segments'"
                )

    def _generate_batch(self):
        if self.mode == "single":
            return ink_trace_rs.generate_single_stroke_batch(
                self.batch_size, self.img_size
            )
        elif self.mode == "independent":
            return ink_trace_rs.generate_independent_strokes_batch(
                self.batch_size, self.img_size, self.max_strokes, self.fixed_count
            )
        elif self.mode == "continuous":
            return ink_trace_rs.generate_continuous_strokes_batch(
                self.batch_size, self.img_size, self.max_segments
            )
        elif self.mode == "multi_connected":
            return ink_trace_rs.generate_multi_connected_strokes_batch(
                self.batch_size, self.img_size, self.max_paths, self.max_segments
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        target_threads = None
        if self.rust_threads is not None:
            target_threads = self.rust_threads
        elif worker_info is not None:
            target_threads = 1

        if target_threads is not None:
            try:
                if hasattr(ink_trace_rs, "set_rayon_threads"):
                    ink_trace_rs.set_rayon_threads(target_threads)
            except RuntimeError:
                pass

        if worker_info is None:
            total_items = self.epoch_length
        else:
            per_worker = int(
                math.ceil(self.epoch_length / float(worker_info.num_workers))
            )
            total_items = per_worker

        generated_count = 0
        while generated_count < total_items:
            try:
                imgs, labels = self._generate_batch()
            except Exception as e:
                print(f"Rust generation failed: {e}")
                raise e

            curr_batch_len = len(imgs)
            for i in range(curr_batch_len):
                if generated_count >= total_items:
                    break
                img_tensor = torch.from_numpy(imgs[i]).unsqueeze(0)
                raw_label = labels[i]
                if self.for_detr:
                    raw_label = self._process_label_for_detr(raw_label)
                label_tensor = torch.from_numpy(raw_label.copy())
                yield img_tensor, label_tensor
                generated_count += 1

    def _process_label_for_detr(self, raw_label):
        if self.mode == "single":
            return raw_label
        elif self.mode == "independent":
            return raw_label
        elif self.mode == "continuous":
            label = raw_label.copy()
            first_found = False
            for k in range(len(label)):
                if label[k, 10] > 0.5:
                    if not first_found:
                        label[k, 10] = 1.0
                        first_found = True
                    else:
                        label[k, 10] = 2.0
            return label
        elif self.mode == "multi_connected":
            label = raw_label.copy()
            num_paths, num_segs, _ = label.shape
            for p in range(num_paths):
                first_found = False
                for s in range(num_segs):
                    if label[p, s, 10] > 0.5:
                        if not first_found:
                            label[p, s, 10] = 1.0
                            first_found = True
                        else:
                            label[p, s, 10] = 2.0
            return label
        else:
            return raw_label

    def __len__(self):
        return self.epoch_length
