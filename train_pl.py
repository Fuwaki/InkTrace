#!/usr/bin/env python3
"""
PyTorch Lightning 训练入口脚本 (Hydra + OmegaConf 版本)

特性：
- Hydra 配置管理：结构化配置 + 命令行覆盖
- OmegaConf 变量插值和类型安全
- 自动处理无限数据集与 OneCycleLR 的兼容问题
- 自动 Checkpoint 管理 (Top-K & Last)
- 支持多阶段训练流水线 (structural -> dense -> finetune)
- Curriculum Learning 支持
- 混合精度训练 (AMP)
- TensorBoard 日志

使用方法：
    # 单阶段训练 (structural pretrain)
    python train_pl.py stage=structural

    # 单阶段训练 (dense)
    python train_pl.py stage=dense

    # Dense 训练并从 structural checkpoint 初始化
    python train_pl.py stage=dense training.init_from=checkpoints/structural/last.ckpt

    # 覆盖配置参数
    python train_pl.py stage=dense training.lr=5e-4 training.epochs=100

    # 断点续训
    python train_pl.py stage=dense training.resume=checkpoints/dense/last.ckpt

    # 多阶段自动训练 (使用 multirun)
    python train_pl.py --multirun stage=structural,dense
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig, OmegaConf
from lightning_model import UnifiedTask
from lightning_data import InkTraceDataModule
from lightning_vis import VisualizationCallback


# =============================================================================
# 配置工具函数
# =============================================================================


def validate_config(cfg: DictConfig) -> None:
    """验证配置有效性并确保类型正确"""
    training = cfg.training
    data = cfg.data

    # =========================================================================
    # 验证 training 配置
    # =========================================================================
    assert training.lr > 0, f"Learning rate must be positive, got {training.lr}"
    assert training.batch_size > 0, f"Batch size must be positive, got {training.batch_size}"
    assert training.epochs > 0, f"Epochs must be positive, got {training.epochs}"
    assert training.epoch_length > 0, f"Epoch length must be positive, got {training.epoch_length}"
    assert training.grad_clip >= 0, f"Grad clip must be non-negative, got {training.grad_clip}"
    assert training.weight_decay >= 0, f"Weight decay must be non-negative, got {training.weight_decay}"

    # =========================================================================
    # 验证 data 配置
    # =========================================================================
    if hasattr(data, "img_size"):
        assert data.img_size > 0, f"img_size must be positive, got {data.img_size}"
    if hasattr(data, "num_workers"):
        assert data.num_workers >= 0, f"num_workers must be non-negative, got {data.num_workers}"
    if hasattr(data, "keypoint_sigma"):
        assert data.keypoint_sigma > 0, f"keypoint_sigma must be positive, got {data.keypoint_sigma}"

    print(f"✅ Config validation passed for stage: {cfg.stage.name}")


def print_config(cfg: DictConfig) -> None:
    """打印配置摘要"""
    print(f"\n{'─' * 60}")
    print(f"📋 Stage: {cfg.stage.name}")
    if cfg.stage.get("description"):
        print(f"   {cfg.stage.description}")
    print(f"{'─' * 60}")

    print("  Training:")
    print(f"    lr: {cfg.training.lr}")
    print(f"    epochs: {cfg.training.epochs}")
    print(f"    batch_size: {cfg.training.batch_size}")
    print(f"    epoch_length: {cfg.training.epoch_length}")

    print("  Data:")
    print(f"    curriculum_stage: {cfg.data.curriculum_stage}")
    print(f"    num_workers: {cfg.data.num_workers}")

    if cfg.stage.get("init_from"):
        print(f"  Init from: {cfg.stage.init_from}")
    if cfg.stage.get("freeze_encoder"):
        print("  Freeze encoder: True")

    print(f"{'─' * 60}\n")


# =============================================================================
# Trainer 工厂
# =============================================================================


def create_trainer(cfg: DictConfig, resume_from: Optional[str] = None) -> pl.Trainer:
    """
    创建配置好的 Trainer

    Args:
        cfg: Hydra DictConfig 配置
        resume_from: 断点续训 checkpoint 路径

    Returns:
        配置好的 pl.Trainer
    """
    training = cfg.training
    logging = cfg.logging
    device = cfg.device
    stage_name = cfg.stage.name

    # =========================================================================
    # 核心修复: limit_train_batches
    # 对于无限数据集，必须设置此参数来定义每个 epoch 的 batch 数量
    # 这样 OneCycleLR 才能正确计算 total_steps
    # =========================================================================
    epoch_length = training.epoch_length
    batch_size = training.batch_size
    limit_train_batches = epoch_length // batch_size

    # TensorBoard Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = logging.get("tensorboard_dir", "runs")
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=stage_name,
        version=timestamp,
    )

    # =========================================================================
    # Callbacks
    # =========================================================================
    callbacks = []

    # 1. ModelCheckpoint - 保存 Top-K 和 Last
    checkpoint_config = training.checkpoint
    save_dir = Path(checkpoint_config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    monitor_metric = checkpoint_config.get("monitor", "val/loss")
    monitor_mode = checkpoint_config.get("mode", "min")

    # 对于 structural 阶段，监控 train/loss（无验证集）
    if stage_name == "structural":
        monitor_metric = checkpoint_config.get("monitor", "train/loss")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="epoch{epoch:02d}-{" + monitor_metric.replace("/", "_") + ":.4f}",
        save_top_k=checkpoint_config.get("keep_top_k", 3),
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=checkpoint_config.get("save_last", True),
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    # 2. LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # 3. RichProgressBar (可选)
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    # 4. Visualization Callback
    vis_config = training.visualization
    if vis_config.get("enabled", True):
        # Dense 阶段自动启用完整可视化 (16列，包含所有预测头)
        use_dense_vis = (stage_name == "dense")

        vis_callback = VisualizationCallback(
            num_samples=int(vis_config.get("num_samples", 4)),
            log_metrics=vis_config.get("log_metrics", True),
            log_interval=int(vis_config.get("log_interval", 1)),
            prefix="Validation" if stage_name != "structural" else "Train",
            use_dense_vis=use_dense_vis,  # Dense 阶段使用完整可视化
        )
        callbacks.append(vis_callback)

        vis_type = "Dense (16 cols)" if use_dense_vis else "Standard (6 cols)"
        print(f"🎨 Visualization: {vis_config.get('num_samples', 4)} samples ({vis_type})")

    # =========================================================================
    # Trainer 配置
    # =========================================================================

    # 设备配置
    accelerator = device.get("accelerator", "auto")
    precision = device.get("precision", "16-mixed")

    # CPU 模式下使用 32 精度
    if accelerator == "cpu":
        precision = "32"

    trainer = pl.Trainer(
        # 基础配置
        max_epochs=int(training.epochs),
        accelerator=accelerator,
        devices="auto",
        precision=precision,
        # 核心: 限制每个 epoch 的 batch 数量
        limit_train_batches=limit_train_batches,
        limit_val_batches=max(1, limit_train_batches // 10),
        # 梯度裁剪
        gradient_clip_val=float(training.grad_clip),
        # Callbacks & Logger
        callbacks=callbacks,
        logger=logger,
        # 日志频率
        log_every_n_steps=int(logging.get("log_interval", 10)),
        # 验证频率
        check_val_every_n_epoch=1,
        # 关闭 sanity check
        num_sanity_val_steps=0,
        # 性能优化
        enable_model_summary=True,
        enable_progress_bar=True,
    )

    return trainer


# =============================================================================
# 训练函数
# =============================================================================


def run_training(cfg: DictConfig, resume_from: Optional[str] = None) -> str:
    """
    运行训练

    Args:
        cfg: Hydra DictConfig 配置
        resume_from: 断点续训 checkpoint 路径

    Returns:
        最佳 checkpoint 路径
    """
    stage_name = cfg.stage.name

    print(f"\n{'=' * 60}")
    print(f"🚀 Starting stage: {stage_name}")
    print(f"{'=' * 60}")

    # 验证和打印配置
    validate_config(cfg)
    print_config(cfg)

    training = cfg.training
    data_cfg = cfg.data

    # =========================================================================
    # 创建 DataModule
    # =========================================================================
    datamodule = InkTraceDataModule(
        img_size=int(data_cfg.img_size),
        batch_size=int(training.batch_size),
        epoch_length=int(training.epoch_length),
        curriculum_stage=int(data_cfg.curriculum_stage),
        num_workers=int(data_cfg.num_workers),
        rust_threads=data_cfg.get("rust_threads"),
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", True),
        keypoint_sigma=float(data_cfg.keypoint_sigma),
    )

    # =========================================================================
    # 创建模型 (使用硬编码的超参数)
    # =========================================================================
    model = UnifiedTask(
        stage=stage_name if stage_name != "debug" else "dense",
        lr=float(training.lr),
        weight_decay=float(training.weight_decay),
        loss_weights=training.get("loss_weights"),
        grad_clip=float(training.grad_clip),
    )

    # =========================================================================
    # 创建 Trainer
    # =========================================================================
    trainer = create_trainer(cfg, resume_from)

    # =========================================================================
    # 开始训练
    # =========================================================================
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=resume_from,
    )

    # 返回最佳 checkpoint 路径
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\n✅ Stage {stage_name} completed!")
    print(f"   Best checkpoint: {best_ckpt}")

    return best_ckpt


# =============================================================================
# 主入口
# =============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """主入口函数"""
    # 设置 float32 矩阵乘法精度以利用 Tensor Cores (RTX 5090 优化)
    torch.set_float32_matmul_precision("high")

    # 打印完整配置（调试用，可选）
    if cfg.get("print_config", False):
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)

    # 设置随机种子
    seed = cfg.get("seed", 114514)
    pl.seed_everything(seed, workers=True)

    # 获取 resume 路径
    resume_from = cfg.training.get("resume")

    # 运行训练
    best_ckpt = run_training(cfg, resume_from=resume_from)

    print("\n🎉 Training completed!")
    print(f"   Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
