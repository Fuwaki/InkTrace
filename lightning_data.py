"""
PyTorch Lightning DataModule for InkTrace

封装 DenseInkTraceDataset，支持：
- 在线数据生成 (Online Generation)
- Curriculum Learning (渐进式训练)
- 多 Worker 并行加载

关键点：
- 数据集是无限的，epoch 长度由 Trainer.limit_train_batches 控制
- 支持动态切换 curriculum stage
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional

from datasets_v2 import DenseInkTraceDataset, collate_dense_batch


class InkTraceDataModule(pl.LightningDataModule):
    """
    InkTrace 数据模块

    封装在线生成的 DenseInkTraceDataset，支持 Curriculum Learning。

    Args:
        img_size: 图像尺寸 (默认 64)
        batch_size: DataLoader 批量大小
        epoch_length: 每个 epoch 的样本数 (用于 IterableDataset 的 __len__)
        curriculum_stage: 初始 curriculum 阶段 (0-9)
        num_workers: DataLoader worker 数量
        rust_threads: Rust 生成器线程数 (None 表示自动)
        pin_memory: 是否使用 pin_memory
        persistent_workers: 是否保持 worker 进程
    """

    def __init__(
        self,
        img_size: int = 64,
        batch_size: int = 128,
        epoch_length: int = 10000,
        curriculum_stage: int = 0,
        num_workers: int = 4,
        rust_threads: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = img_size
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.curriculum_stage = curriculum_stage
        self.num_workers = num_workers
        self.rust_threads = rust_threads
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        # 数据集实例 (会在 setup 中创建)
        self._train_dataset: Optional[DenseInkTraceDataset] = None
        self._val_dataset: Optional[DenseInkTraceDataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        创建数据集实例

        Args:
            stage: "fit", "validate", "test", 或 "predict"
        """
        if stage == "fit" or stage is None:
            self._train_dataset = DenseInkTraceDataset(
                img_size=self.img_size,
                batch_size=self.batch_size,
                epoch_length=self.epoch_length,
                curriculum_stage=self.curriculum_stage,
                rust_threads=self.rust_threads,
            )

            # 验证集使用相同配置但可能不同的 stage
            self._val_dataset = DenseInkTraceDataset(
                img_size=self.img_size,
                batch_size=self.batch_size,
                epoch_length=self.epoch_length // 10,  # 验证集更小
                curriculum_stage=self.curriculum_stage,
                rust_threads=self.rust_threads,
            )

        if stage == "validate":
            self._val_dataset = DenseInkTraceDataset(
                img_size=self.img_size,
                batch_size=self.batch_size,
                epoch_length=self.epoch_length // 10,
                curriculum_stage=self.curriculum_stage,
                rust_threads=self.rust_threads,
            )

    def train_dataloader(self) -> DataLoader:
        """返回训练 DataLoader"""
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_dense_batch,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """返回验证 DataLoader"""
        if self._val_dataset is None:
            return None

        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            num_workers=max(1, self.num_workers // 2),
            collate_fn=collate_dense_batch,
            pin_memory=self.pin_memory,
            persistent_workers=False,  # 验证不需要持久化
        )

    def set_curriculum(self, stage: int):
        """
        动态更新 curriculum stage

        由 CurriculumCallback 调用

        Args:
            stage: 新的 curriculum 阶段 (0-9)
        """
        self.curriculum_stage = stage

        if self._train_dataset is not None:
            self._train_dataset.set_curriculum(stage)

        if self._val_dataset is not None:
            self._val_dataset.set_curriculum(stage)

    @property
    def train_dataset(self) -> Optional[DenseInkTraceDataset]:
        """获取训练数据集实例"""
        return self._train_dataset

    def state_dict(self):
        """保存 DataModule 状态 (用于 checkpoint)"""
        return {
            "curriculum_stage": self.curriculum_stage,
        }

    def load_state_dict(self, state_dict):
        """加载 DataModule 状态"""
        if "curriculum_stage" in state_dict:
            self.set_curriculum(state_dict["curriculum_stage"])


def create_datamodule_from_config(config: dict) -> InkTraceDataModule:
    """
    从配置字典创建 DataModule

    Args:
        config: 包含 data 和 training 配置的字典

    Returns:
        InkTraceDataModule 实例
    """
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    return InkTraceDataModule(
        img_size=data_config.get("img_size", 64),
        batch_size=training_config.get("batch_size", 128),
        epoch_length=training_config.get("epoch_length", 10000),
        curriculum_stage=data_config.get("curriculum_stage", 0),
        num_workers=data_config.get("num_workers", 4),
        rust_threads=data_config.get("rust_threads", None),
    )
