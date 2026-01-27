import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import ink_trace_rs
import math


class InkTraceDataset(IterableDataset):
    """
    基于 Rust 高性能生成后端的无限数据集。

    支持四种模式：
    1. 'single': 单一贝塞尔曲线
    2. 'independent': 多条独立贝塞尔曲线 (Set Prediction 任务)
    3. 'continuous': 连续多段贝塞尔曲线 (序列生成任务)
    4. 'multi_connected': 多条路径，每条路径包含多段曲线 (复杂文档布局模拟)
    """

    def __init__(
        self,
        mode="single",
        img_size=64,
        batch_size=64,
        epoch_length=10000,
        # 特定模式参数
        max_strokes=None,  # mode='independent'
        fixed_count=None,  # mode='independent' (可选，固定数量)
        max_paths=None,  # mode='multi_connected'
        max_segments=None,  # mode='continuous' or 'multi_connected'
        seed=None,
        rust_threads=None,  # 手动指定 Rust 内部线程数 (None=自动策略)
    ):
        """
        参数:
            mode (str): 生成模式 ['single', 'independent', 'continuous', 'multi_connected']
            img_size (int): 图像尺寸 (WxH)
            batch_size (int): Rust 内部生成时的批次大小。
            epoch_length (int): 虚拟 epoch 长度。

            rust_threads (int):
                控制 Rust (Rayon) 并行线程数。
                - None (默认): 自动策略。
                    - 主进程模式 (num_workers=0): 不限制，使用所有 CPU 核心。
                    -多进程模式 (num_workers>0): 限制为 1 线程，利用进程级并行。
                - Int > 0: 强制指定每个进程/Worker 内的 Rust 线程数。
                  (例如机器核心数很多，worker 数较少时，可以设为 >1)
        """
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

        # 参数校验
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
        """调用 Rust 扩展生成一批数据"""
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
        """
        生成器入口。
        处理多进程 Worker 分割情况，确保数据量分配正确。
        """
        worker_info = torch.utils.data.get_worker_info()

        # 配置 Rayon 线程数
        # 策略：
        # 1. 只有当 worker_info 存在（即处于 DataLoader Worker 子进程中）且 rust_threads 未手动指定时，
        #    才强制设为 1，防止 CPU 过载。
        # 2. 如果 rust_threads 有值，则无条件执行设置。
        # 3. 如果在主进程 (num_workers=0) 且 rust_threads=None，则不限制（Rayon 默认全核）。

        target_threads = None

        if self.rust_threads is not None:
            target_threads = self.rust_threads
        elif worker_info is not None:
            # 多进程默认策略：防止争抢
            target_threads = 1

        if target_threads is not None:
            try:
                if hasattr(ink_trace_rs, "set_rayon_threads"):
                    ink_trace_rs.set_rayon_threads(target_threads)
            except RuntimeError:
                # Rayon 全局线程池只能初始化一次，后续调用会报错，忽略即可
                pass

        if worker_info is None:
            # 单进程
            total_items = self.epoch_length
        else:
            # 多进程：将总长度均分给每个 worker
            per_worker = int(
                math.ceil(self.epoch_length / float(worker_info.num_workers))
            )
            total_items = per_worker

            # 可以在这里根据 worker_id 设置不同的 Rust 种子（如果 Rust 接口支持）
            # 目前 Rust 使用 thread_rng，本身就是线程安全的且各不相同

        generated_count = 0

        while generated_count < total_items:
            # 1. 调用 Rust 高效生成一批
            try:
                imgs, labels = self._generate_batch()
            except Exception as e:
                print(f"Rust generation failed: {e}")
                raise e

            curr_batch_len = len(imgs)

            # 2. 逐个 Yield 给 DataLoader
            for i in range(curr_batch_len):
                if generated_count >= total_items:
                    break

                # numpy -> tensor
                # 增加 Channel 维度: [H, W] -> [1, H, W]
                # 注意：Rust 返回的是 float32 且已归一化到 0-1
                img_tensor = torch.from_numpy(imgs[i]).unsqueeze(0)
                label_tensor = torch.from_numpy(labels[i])

                yield img_tensor, label_tensor
                generated_count += 1

    def __len__(self):
        return self.epoch_length


class MixedInkTraceDataset(IterableDataset):
    """
    混合多种生成模式的数据集。
    如果你想在一个 Epoch 里同时训练单笔画、连续笔画等，可以使用这个类。

    实现方式：
    内部根据配置实例化多个 InkTraceDataset 的生成逻辑，
    在迭代时随机选择一种模式生成一个 Batch。
    """

    def __init__(self, configs, epoch_length=10000, batch_size=64, rust_threads=None):
        """
        Args:
            configs: 列表，每个元素为 (kwargs_dict, probability)
            epoch_length: epoch 长度
            batch_size: batch 大小
            rust_threads: Rayon 线程数设置 (透传给子 Dataset)
        """
        super().__init__()
        self.configs = configs
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.rust_threads = rust_threads

        # 预先根据配置创建多个 dataset 实例（仅用于借用它们的生成逻辑）
        self.datasets = []
        self.probs = []

        for kwargs, prob in configs:
            # 强制与其共享 batch_size，便于管理
            kwargs["batch_size"] = self.batch_size
            kwargs["epoch_length"] = self.epoch_length
            kwargs["rust_threads"] = self.rust_threads  # 统一透传

            ds = InkTraceDataset(**kwargs)
            self.datasets.append(ds)
            self.probs.append(prob)

        # 归一化概率
        total_prob = sum(self.probs)
        self.probs = [p / total_prob for p in self.probs]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # 同样的 Rayon 线程保护逻辑
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
            # 随机选择一个 Dataset 模式
            # np.random.choice 不支持对象数组，所以用索引
            dataset_idx = np.random.choice(len(self.datasets), p=self.probs)
            chosen_dataset = self.datasets[dataset_idx]

            # 生成 Batch
            imgs, labels = chosen_dataset._generate_batch()
            curr_batch_len = len(imgs)

            for i in range(curr_batch_len):
                if generated_count >= total_items:
                    break

                img_tensor = torch.from_numpy(imgs[i]).unsqueeze(0)
                label_tensor = torch.from_numpy(labels[i])

                # 警告：不同模式的 label 形状可能不一致。
                # 如果混合使用，请确保你的 collate_fn 能处理不同形状的 label，
                # 或者只混合 label 形状兼容的模式。
                yield img_tensor, label_tensor
                generated_count += 1

    def __len__(self):
        return self.epoch_length
