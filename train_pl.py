#!/usr/bin/env python3
"""
PyTorch Lightning 训练入口脚本

特性：
- 自动处理无限数据集与 OneCycleLR 的兼容问题
- 自动 Checkpoint 管理 (Top-K & Last)
- 支持多阶段训练流水线 (structural -> dense)
- Curriculum Learning 支持
- 混合精度训练 (AMP)
- TensorBoard 日志

使用方法：
    # 单阶段训练 (structural pretrain)
    python train_pl.py --config configs/default.yaml --stage structural

    # 单阶段训练 (dense)
    python train_pl.py --config configs/default.yaml --stage dense

    # Dense 训练并从 structural checkpoint 初始化
    python train_pl.py --config configs/default.yaml --stage dense \\
        --init_from checkpoints/structural/last.ckpt

    # 多阶段自动训练
    python train_pl.py --config configs/default.yaml --run-all-stages

    # 断点续训
    python train_pl.py --config configs/default.yaml --stage dense \\
        --resume checkpoints/dense/last.ckpt
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_model import UnifiedTask, CurriculumCallback
from lightning_data import InkTraceDataModule
from lightning_vis import VisualizationCallback


# =============================================================================
# 配置加载
# =============================================================================


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_stage_config(config: dict, stage_name: str) -> dict:
    """
    获取特定阶段的配置，合并全局配置

    Args:
        config: 全局配置
        stage_name: 阶段名称 ("structural" 或 "dense")

    Returns:
        合并后的阶段配置
    """
    # 复制基础配置
    stage_config = {
        "model": dict(config.get("model", {})),
        "training": dict(config.get("training", {})),
        "data": dict(config.get("data", {})),
        "logging": dict(config.get("logging", {})),
        "device": dict(config.get("device", {})),
    }

    # 查找并合并阶段特定配置
    stages = config.get("stages", [])
    for stage in stages:
        if stage.get("name") == stage_name:
            # 合并模型配置
            if "model" in stage:
                stage_config["model"].update(stage["model"])
            # 合并训练配置
            if "training" in stage:
                stage_config["training"].update(stage["training"])
            # 合并数据配置
            if "data" in stage:
                stage_config["data"].update(stage["data"])
            # 阶段特定的 epochs
            if "epochs" in stage:
                stage_config["training"]["epochs"] = stage["epochs"]
            # 阶段特定的 init_from
            if "init_from" in stage:
                stage_config["init_from"] = stage["init_from"]
            break

    return stage_config


# =============================================================================
# Trainer 工厂
# =============================================================================


def create_trainer(
    config: dict,
    stage_name: str,
    resume_from: Optional[str] = None,
) -> pl.Trainer:
    """
    创建配置好的 Trainer

    Args:
        config: 阶段配置
        stage_name: 阶段名称
        resume_from: 断点续训 checkpoint 路径

    Returns:
        配置好的 pl.Trainer
    """
    training_config = config.get("training", {})
    logging_config = config.get("logging", {})
    device_config = config.get("device", {})

    # =========================================================================
    # 核心修复: limit_train_batches
    # 对于无限数据集，必须设置此参数来定义每个 epoch 的 batch 数量
    # 这样 OneCycleLR 才能正确计算 total_steps
    # =========================================================================
    epoch_length = training_config.get("epoch_length", 10000)
    batch_size = training_config.get("batch_size", 128)
    limit_train_batches = epoch_length // batch_size

    # TensorBoard Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = logging_config.get("tensorboard_dir", "runs")
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
    save_dir = Path(training_config.get("save_dir", "checkpoints")) / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="epoch{epoch:02d}-loss{train/loss:.4f}",
        save_top_k=training_config.get("keep_last_n", 3),
        monitor="train/loss",
        mode="min",
        save_last=True,  # 始终保存 last.ckpt 用于续训
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    # 2. LearningRateMonitor - 按 step 记录学习率
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # 3. RichProgressBar - 美化进度条 (可选)
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass  # rich 未安装

    # 4. Curriculum Learning Callback (仅 dense 阶段且启用时)
    curriculum_config = training_config.get("curriculum", {})
    if curriculum_config.get("enabled", False) and stage_name == "dense":
        curriculum_callback = CurriculumCallback(
            start_stage=curriculum_config.get("start_stage", 0),
            end_stage=curriculum_config.get("end_stage", 9),
            epochs_per_stage=curriculum_config.get("epochs_per_stage", 10),
        )
        callbacks.append(curriculum_callback)
        print(
            f"📈 Curriculum Learning enabled: "
            f"stage {curriculum_config.get('start_stage', 0)} -> "
            f"{curriculum_config.get('end_stage', 9)}, "
            f"{curriculum_config.get('epochs_per_stage', 10)} epochs/stage"
        )

    # 5. Visualization Callback - 自动生成对比图并记录到 TensorBoard
    # 可视化配置
    vis_config = training_config.get("visualization", {})
    if vis_config.get("enabled", True):
        vis_callback = VisualizationCallback(
            num_samples=vis_config.get("num_samples", 4),
            log_metrics=vis_config.get("log_metrics", True),
            prefix="Validation" if stage_name == "dense" else "Train",
        )
        callbacks.append(vis_callback)
        print(
            f"🎨 Visualization enabled: "
            f"{vis_config.get('num_samples', 4)} samples per validation"
        )

    # =========================================================================
    # Trainer 配置
    # =========================================================================
    accelerator = "auto"
    devices = "auto"

    # 设备配置
    device_type = device_config.get("type")
    if device_type:
        if device_type == "cuda":
            accelerator = "gpu"
        elif device_type == "cpu":
            accelerator = "cpu"
        elif device_type == "xpu":
            accelerator = "xpu"

    # 精度配置
    precision = "16-mixed" if accelerator in ["gpu", "cuda"] else "32"

    trainer = pl.Trainer(
        # 基础配置
        max_epochs=training_config.get("epochs", 50),
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        # 核心修复: 限制每个 epoch 的 batch 数量
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_train_batches // 10,  # 验证更少
        # 梯度裁剪
        gradient_clip_val=training_config.get("grad_clip", 1.0),
        # Callbacks & Logger
        callbacks=callbacks,
        logger=logger,
        # 日志频率
        log_every_n_steps=logging_config.get("log_interval", 10),
        # 验证频率
        val_check_interval=logging_config.get("vis_interval", 2),
        check_val_every_n_epoch=logging_config.get("vis_interval", 2),
        # 性能优化
        enable_model_summary=True,
        enable_progress_bar=True,
        # 断点续训
        # 注意: ckpt_path 在 trainer.fit() 中传入，而非这里
    )

    return trainer


# =============================================================================
# 训练函数
# =============================================================================


def run_stage(
    config: dict,
    stage_name: str,
    init_from: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> str:
    """
    运行单个训练阶段

    Args:
        config: 全局配置
        stage_name: 阶段名称 ("structural" 或 "dense")
        init_from: 初始化权重路径 (用于迁移学习)
        resume_from: 断点续训 checkpoint 路径

    Returns:
        最佳 checkpoint 路径
    """
    print(f"\n{'=' * 60}")
    print(f"🚀 Starting stage: {stage_name}")
    print(f"{'=' * 60}\n")

    # 获取阶段配置
    stage_config = get_stage_config(config, stage_name)
    training_config = stage_config.get("training", {})
    model_config = stage_config.get("model", {})
    data_config = stage_config.get("data", {})

    # =========================================================================
    # 创建 DataModule
    # =========================================================================
    datamodule = InkTraceDataModule(
        img_size=data_config.get("img_size", 64),
        batch_size=training_config.get("batch_size", 128),
        epoch_length=training_config.get("epoch_length", 10000),
        curriculum_stage=data_config.get("curriculum_stage", 0),
        num_workers=data_config.get("num_workers", 4),
        rust_threads=data_config.get("rust_threads", None),
    )

    # =========================================================================
    # 创建模型
    # =========================================================================
    loss_weights = training_config.get("loss_weights", None)

    model = UnifiedTask(
        stage=stage_name,
        embed_dim=model_config.get("embed_dim", 128),
        num_layers=model_config.get("num_layers", 4),
        lr=float(training_config.get("lr", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
        loss_weights=loss_weights,
        mask_ratio=float(training_config.get("mask_ratio", 0.6)),
        mask_strategy=training_config.get("mask_strategy", "block"),
        grad_clip=float(training_config.get("grad_clip", 1.0)),
    )

    # 从 checkpoint 初始化权重 (迁移学习)
    if init_from and not resume_from:
        model.load_pretrained_weights(init_from, strict=False)

    # =========================================================================
    # 创建 Trainer
    # =========================================================================
    trainer = create_trainer(stage_config, stage_name, resume_from)

    # =========================================================================
    # 开始训练
    # =========================================================================
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=resume_from,  # 断点续训
    )

    # 返回最佳 checkpoint 路径
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\n✅ Stage {stage_name} completed!")
    print(f"   Best checkpoint: {best_ckpt}")

    return best_ckpt


def run_all_stages(config: dict):
    """
    运行所有训练阶段

    按配置文件中的 stages 顺序执行
    """
    stages = config.get("stages", [])
    if not stages:
        raise ValueError("No stages defined in config")

    last_ckpt = None

    for stage_info in stages:
        stage_name = stage_info["name"]

        # 确定 init_from
        init_from = stage_info.get("init_from")
        if init_from and last_ckpt and "*" in init_from:
            # 替换通配符
            init_from = init_from.replace("*", str(last_ckpt))
        elif not init_from and last_ckpt:
            # 使用上一阶段的 checkpoint
            init_from = str(last_ckpt)

        # 运行阶段
        best_ckpt = run_stage(config, stage_name, init_from=init_from)
        last_ckpt = best_ckpt

    print(f"\n{'#' * 60}")
    print("🎉 All stages completed!")
    print(f"   Final checkpoint: {last_ckpt}")
    print(f"{'#' * 60}\n")


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="InkTrace PyTorch Lightning Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 配置
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["structural", "dense"],
        help="训练阶段",
    )
    parser.add_argument(
        "--run-all-stages",
        action="store_true",
        help="自动运行所有阶段",
    )

    # Checkpoint
    parser.add_argument(
        "--init_from",
        type=str,
        help="从 checkpoint 初始化模型 (迁移学习)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="从 checkpoint 断点续训",
    )

    # 覆盖配置 (可选)
    parser.add_argument("--lr", type=float, help="覆盖学习率")
    parser.add_argument("--epochs", type=int, help="覆盖训练轮数")
    parser.add_argument("--batch_size", type=int, help="覆盖批次大小")

    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 覆盖配置
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # 设置随机种子
    pl.seed_everything(114514, workers=True)

    # 运行训练
    if args.run_all_stages:
        run_all_stages(config)
    elif args.stage:
        run_stage(
            config,
            args.stage,
            init_from=args.init_from,
            resume_from=args.resume,
        )
    else:
        raise ValueError("请指定 --stage 或 --run-all-stages")


if __name__ == "__main__":
    main()
