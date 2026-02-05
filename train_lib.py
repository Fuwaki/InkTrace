"""
统一的训练框架 - 解决所有训练脚本的共性问题

使用方法:
    # 单阶段训练
    python train.py --config configs/default.yaml --stage structural

    # 多阶段训练（自动执行所有阶段）
    python train.py --config configs/default.yaml --run-all-stages

    # 恢复训练
    python train.py --resume checkpoints/structural_epoch10.pth
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ============================================================================
# 配置管理
# ============================================================================


@dataclass
class Config:
    """统一配置类 - 从 YAML 或 checkpoint 加载"""

    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    device: Dict[str, Any] = field(default_factory=dict)
    stages: list = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置"""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str) -> "Config":
        """从 checkpoint 加载配置"""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        config_data = ckpt.get("config", {})
        return cls(**config_data)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """从命令行参数创建配置（用于向后兼容）"""
        return cls(
            model={"embed_dim": args.embed_dim, "num_layers": args.num_layers},
            training={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "epoch_length": args.epoch_length,
            },
            data={
                "img_size": args.img_size if hasattr(args, "img_size") else 64,
                "num_workers": args.num_workers,
                "rust_threads": args.rust_threads,
                "curriculum_stage": getattr(args, "stage", 1),
            },
            logging={"vis_interval": args.vis_interval},
        )

    def merge(self, other: "Config") -> "Config":
        """合并两个配置（other 覆盖 self）"""
        merged = Config(**self.__dict__)
        for key, value in other.__dict__.items():
            if isinstance(value, dict):
                merged.__dict__[key].update(value)
            elif value is not None:
                merged.__dict__[key] = value
        return merged

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model": self.model,
            "training": self.training,
            "data": self.data,
            "logging": self.logging,
            "device": self.device,
            "stages": self.stages,
        }


# ============================================================================
# Checkpoint 管理器
# ============================================================================


class CheckpointManager:
    """
    统一的 checkpoint 管理

    功能:
      - 自动保存定期 checkpoint (每 N epoch)
      - 保存最佳 checkpoint
      - 自动清理旧 checkpoint (保留最近 N 个)
      - 完整保存 config + model + optimizer + scheduler
      - 提供恢复训练接口
    """

    def __init__(
        self,
        save_dir: str,
        save_interval: int = 5,
        keep_last_n: int = 3,
        metric_mode: str = "min",  # min 或 max
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.metric_mode = metric_mode

        self.best_metric = float("inf") if metric_mode == "min" else float("-inf")
        self.saved_checkpoints = []

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        metric: float,
        config: Config,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ):
        """保存 checkpoint"""

        checkpoint = {
            "version": 1,  # checkpoint 格式版本
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "metric": metric,
            "config": config.to_dict(),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        # 定期保存
        if (epoch + 1) % self.save_interval == 0:
            path = self.save_dir / f"checkpoint_epoch{epoch + 1}.pth"
            torch.save(checkpoint, path)
            self.saved_checkpoints.append(path)
            self._cleanup_old_checkpoints()
            print(f"  Saved checkpoint: {path}")

        # 保存最新
        latest_path = self.save_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)

        # 保存最佳
        is_best = (self.metric_mode == "min" and metric < self.best_metric) or (
            self.metric_mode == "max" and metric > self.best_metric
        )
        if is_best:
            self.best_metric = metric
            best_path = self.save_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best! ({self.best_metric:.4f})")

        return is_best

    def _cleanup_old_checkpoints(self):
        """清理旧 checkpoint，保留最近 N 个"""
        if len(self.saved_checkpoints) > self.keep_last_n:
            # 排序并删除最旧的
            self.saved_checkpoints.sort(key=lambda p: int(p.stem.split("epoch")[1]))
            for old_ckpt in self.saved_checkpoints[: -self.keep_last_n]:
                if old_ckpt.exists():
                    old_ckpt.unlink()
                    print(f"  Removed old checkpoint: {old_ckpt}")
            self.saved_checkpoints = self.saved_checkpoints[-self.keep_last_n :]

    def load(
        self,
        ckpt_path: str,
        model: torch.nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        load_optimizer: bool = True,
        strict: bool = True,
    ):
        """加载 checkpoint，自动恢复状态"""

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 验证 checkpoint 版本
        version = ckpt.get("version", 0)
        if version != 1:
            print(f"  Warning: checkpoint version {version} != 1")

        # 加载模型
        if strict:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if incompatible.missing_keys:
                print(
                    f"  Warning: Missing keys: {len(incompatible.missing_keys)} keys (expected for transfer learning)"
                )
            if incompatible.unexpected_keys:
                print(
                    f"  Warning: Unexpected keys: {len(incompatible.unexpected_keys)} keys"
                )

        print(f"  Loaded model from: {ckpt_path}")

        # 加载 optimizer/scheduler
        if load_optimizer:
            if optimizer is not None and "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    print("  Restored optimizer state")
                except Exception as e:
                    print(f"  Warning: could not restore optimizer: {e}")

            if scheduler is not None and ckpt.get("scheduler_state_dict"):
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    print("  Restored scheduler state")
                except Exception as e:
                    print(f"  Warning: could not restore scheduler: {e}")

        # 返回训练信息
        return {
            "epoch": ckpt.get("epoch", 0),
            "metric": ckpt.get("metric", None),
            "config": Config(**ckpt.get("config", {})),
            "metadata": ckpt.get("metadata", {}),
        }


# ============================================================================
# 统一训练器
# ============================================================================


class BaseTrainer:
    """
    统一训练器基类 - 处理所有训练的通用逻辑

    子类只需要实现:
      - get_model(): 创建模型
      - get_criterion(): 创建损失函数
      - train_step(): 单步训练逻辑
      - evaluate(): 评估逻辑 (可选)
    """

    def __init__(self, config: Config, stage_name: str = "default"):
        self.config = config
        self.stage_name = stage_name

        # 设备
        self.device = self._setup_device()

        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = (
            Path(config.logging.get("tensorboard_dir", "runs"))
            / f"{stage_name}_{timestamp}"
        )
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard: {log_dir}")

        # Checkpoint manager
        save_dir = Path(config.training.get("save_dir", "checkpoints")) / stage_name
        self.ckpt_manager = CheckpointManager(
            save_dir=save_dir,
            save_interval=config.training.get("save_interval", 5),
            keep_last_n=config.training.get("keep_last_n", 3),
        )

        # 状态
        self.epoch = 0
        self.global_step = 0

    def _setup_device(self) -> torch.device:
        """自动选择设备"""
        device_type = self.config.device.get("type")
        if device_type:
            return torch.device(device_type)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        else:
            return torch.device("cpu")

    def load_checkpoint(
        self,
        ckpt_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
        reset_epoch: bool = False,
    ):
        """加载 checkpoint
        
        Args:
            ckpt_path: checkpoint 路径
            load_optimizer: 是否加载优化器状态
            strict: 是否严格匹配 state_dict
            reset_epoch: 是否重置 epoch 为 0 (用于迁移学习)
        """
        state = self.ckpt_manager.load(
            ckpt_path,
            self.model,
            self.optimizer if load_optimizer else None,
            self.scheduler if load_optimizer else None,
            load_optimizer=load_optimizer,
            strict=strict,
        )
        if reset_epoch:
            self.epoch = 0
            print(f"  Loaded weights (epoch reset to 0 for transfer learning)")
        else:
            self.epoch = state["epoch"] + 1
            print(f"Resumed from epoch {self.epoch}")
        return state

    def train(self, dataloader):
        """主训练循环"""

        for epoch in range(self.epoch, self.config.training["epochs"]):
            self.epoch = epoch
            self.model.train()
            epoch_losses = {}
            step_cnt = 0

            pbar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.config.training['epochs']}"
            )

            for batch in pbar:
                # 单步训练
                losses = self.train_step(batch)

                # 累积损失
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value
                step_cnt += 1
                self.global_step += 1

                # 日志
                if self.global_step % self.config.logging.get("log_interval", 10) == 0:
                    for key, value in losses.items():
                        self.writer.add_scalar(
                            f"Loss/step_{key}", value, self.global_step
                        )
                    self.writer.add_scalar(
                        "LR", self.optimizer.param_groups[0]["lr"], self.global_step
                    )

                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

            # Epoch 结束
            avg_losses = {k: v / step_cnt for k, v in epoch_losses.items()}
            self._epoch_end(epoch, avg_losses)

        self.writer.close()
        print(f"Training finished! Best metric: {self.ckpt_manager.best_metric:.4f}")

    def _epoch_end(self, epoch: int, avg_losses: Dict[str, float]):
        """Epoch 结束处理 - 保存 checkpoint、评估等"""

        # 记录到 TensorBoard
        for key, value in avg_losses.items():
            self.writer.add_scalar(f"Loss/epoch_{key}", value, epoch)

        print(
            f"Epoch {epoch + 1} finished. Avg Loss: {avg_losses.get('total', list(avg_losses.values())[0]):.4f}"
        )

        # 保存 checkpoint
        metric = avg_losses.get("total", list(avg_losses.values())[0])
        self.ckpt_manager.save(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metric=metric,
            config=self.config,
            metadata={"stage": self.stage_name},
        )

        # 评估
        if (epoch + 1) % self.config.logging.get("vis_interval", 2) == 0:
            self.evaluate()

    # 子类需要实现的方法
    def get_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def get_criterion(self) -> torch.nn.Module:
        raise NotImplementedError

    def train_step(self, batch) -> Dict[str, float]:
        raise NotImplementedError

    def evaluate(self):
        pass  # 可选
