#!/usr/bin/env python3
"""
PyTorch Lightning 训练入口脚本

特性：
- 深度配置合并：默认配置 + 阶段覆盖配置
- 自动处理无限数据集与 OneCycleLR 的兼容问题
- 自动 Checkpoint 管理 (Top-K & Last)
- 支持多阶段训练流水线 (structural -> dense -> finetune)
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
import copy
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

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
# 配置工具函数
# =============================================================================


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典，override 中的值会覆盖 base 中的值

    Args:
        base: 基础字典
        override: 覆盖字典

    Returns:
        合并后的新字典
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = deep_merge(result[key], value)
        else:
            # 直接覆盖
            result[key] = copy.deepcopy(value)

    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_stage_config(config: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    """
    获取特定阶段的完整配置

    原理：
    1. 从全局配置中提取默认值
    2. 从 stages[stage_name] 中提取阶段覆盖配置
    3. 深度合并两者

    Args:
        config: 全局配置
        stage_name: 阶段名称 ("structural", "dense", "finetune", "debug")

    Returns:
        合并后的阶段配置
    """
    # 1. 提取全局默认配置
    defaults = {
        "model": config.get("model", {}),
        "training": config.get("training", {}),
        "data": config.get("data", {}),
        "logging": config.get("logging", {}),
        "device": config.get("device", {}),
        "curriculum": config.get("curriculum", {}),
    }

    # 2. 获取阶段特定配置
    stages = config.get("stages", {})
    if stage_name not in stages:
        print(f"⚠️  Stage '{stage_name}' not found in config, using defaults")
        return defaults

    stage_override = stages[stage_name]

    # 3. 深度合并
    merged = deep_merge(defaults, stage_override)

    # 4. 处理特殊字段
    # - training.curriculum 覆盖全局 curriculum
    if "curriculum" in stage_override.get("training", {}):
        merged["curriculum"] = deep_merge(
            merged.get("curriculum", {}), stage_override["training"]["curriculum"]
        )

    # - 阶段级别的 init_from 和 freeze_encoder
    merged["init_from"] = stage_override.get("init_from")
    merged["freeze_encoder"] = stage_override.get("freeze_encoder", False)
    merged["description"] = stage_override.get("description", "")

    return merged


def validate_config(config: Dict[str, Any], stage_name: str) -> None:
    """验证配置有效性并确保类型正确"""
    training = config.get("training", {})
    model = config.get("model", {})
    data = config.get("data", {})

    # =========================================================================
    # 验证 training 配置
    # =========================================================================
    required_training = ["lr", "epochs", "batch_size", "epoch_length"]
    for field in required_training:
        if field not in training:
            raise ValueError(f"Missing required training config: {field}")

    # 数值范围检查（确保类型转换）
    lr = float(training["lr"])
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    epoch_length = int(training["epoch_length"])
    grad_clip = float(training.get("grad_clip", 1.0))
    weight_decay = float(training.get("weight_decay", 1e-4))
    mask_ratio = float(training.get("mask_ratio", 0.6))

    if lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {lr}")
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")
    if epochs <= 0:
        raise ValueError(f"Epochs must be positive, got {epochs}")
    if epoch_length <= 0:
        raise ValueError(f"Epoch length must be positive, got {epoch_length}")
    if grad_clip < 0:
        raise ValueError(f"Grad clip must be non-negative, got {grad_clip}")
    if weight_decay < 0:
        raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
    if not 0 <= mask_ratio <= 1:
        raise ValueError(f"Mask ratio must be in [0, 1], got {mask_ratio}")

    # 更新 training 配置为正确的类型
    training["lr"] = lr
    training["batch_size"] = batch_size
    training["epochs"] = epochs
    training["epoch_length"] = epoch_length
    training["grad_clip"] = grad_clip
    training["weight_decay"] = weight_decay
    training["mask_ratio"] = mask_ratio

    # 验证 scheduler 配置
    scheduler = training.get("scheduler", {})
    if "warmup_epochs" in scheduler:
        training["scheduler"]["warmup_epochs"] = int(scheduler["warmup_epochs"])
    if "pct_start" in scheduler:
        training["scheduler"]["pct_start"] = float(scheduler["pct_start"])

    # =========================================================================
    # 验证 model 配置
    # =========================================================================
    if "embed_dim" in model:
        model["embed_dim"] = int(model["embed_dim"])
        if model["embed_dim"] <= 0:
            raise ValueError(f"embed_dim must be positive, got {model['embed_dim']}")
    if "num_layers" in model:
        model["num_layers"] = int(model["num_layers"])
        if model["num_layers"] <= 0:
            raise ValueError(f"num_layers must be positive, got {model['num_layers']}")

    # =========================================================================
    # 验证 data 配置
    # =========================================================================
    if "img_size" in data:
        data["img_size"] = int(data["img_size"])
        if data["img_size"] <= 0:
            raise ValueError(f"img_size must be positive, got {data['img_size']}")
    if "curriculum_stage" in data:
        data["curriculum_stage"] = int(data["curriculum_stage"])
    if "num_workers" in data:
        data["num_workers"] = int(data["num_workers"])
        if data["num_workers"] < 0:
            raise ValueError(f"num_workers must be non-negative, got {data['num_workers']}")
    if "keypoint_sigma" in data:
        data["keypoint_sigma"] = float(data["keypoint_sigma"])
        if data["keypoint_sigma"] <= 0:
            raise ValueError(f"keypoint_sigma must be positive, got {data['keypoint_sigma']}")

    print(f"✅ Config validation passed for stage: {stage_name}")


def print_stage_config(config: Dict[str, Any], stage_name: str) -> None:
    """打印阶段配置摘要"""
    training = config.get("training", {})
    model = config.get("model", {})
    data = config.get("data", {})
    curriculum = config.get("curriculum", {})

    print(f"\n{'─' * 60}")
    print(f"📋 Stage: {stage_name}")
    if config.get("description"):
        print(f"   {config['description']}")
    print(f"{'─' * 60}")

    print("  Model:")
    print(f"    embed_dim: {model.get('embed_dim', 128)}")
    print(f"    num_layers: {model.get('num_layers', 4)}")
    print(f"    full_heads: {model.get('full_heads', True)}")

    print("  Training:")
    print(f"    lr: {training.get('lr')}")
    print(f"    epochs: {training.get('epochs')}")
    print(f"    batch_size: {training.get('batch_size')}")
    print(f"    epoch_length: {training.get('epoch_length')}")

    print("  Data:")
    print(f"    curriculum_stage: {data.get('curriculum_stage', 0)}")
    print(f"    num_workers: {data.get('num_workers', 8)}")

    if curriculum.get("enabled"):
        print("  Curriculum Learning:")
        print(
            f"    stages: {curriculum.get('start_stage', 0)} -> {curriculum.get('end_stage', 6)}"
        )
        print(f"    epochs_per_stage: {curriculum.get('epochs_per_stage', 10)}")

    if config.get("init_from"):
        print(f"  Init from: {config['init_from']}")
    if config.get("freeze_encoder"):
        print("  Freeze encoder: True")

    print(f"{'─' * 60}\n")


# =============================================================================
# Trainer 工厂
# =============================================================================


def create_trainer(
    config: Dict[str, Any],
    stage_name: str,
    resume_from: Optional[str] = None,
) -> pl.Trainer:
    """
    创建配置好的 Trainer

    Args:
        config: 阶段配置（已合并的完整配置）
        stage_name: 阶段名称
        resume_from: 断点续训 checkpoint 路径

    Returns:
        配置好的 pl.Trainer
    """
    training_config = config.get("training", {})
    logging_config = config.get("logging", {})
    device_config = config.get("device", {})
    curriculum_config = config.get("curriculum", {})

    # =========================================================================
    # 核心修复: limit_train_batches
    # 对于无限数据集，必须设置此参数来定义每个 epoch 的 batch 数量
    # 这样 OneCycleLR 才能正确计算 total_steps
    # =========================================================================
    epoch_length = int(training_config.get("epoch_length", 10000))
    batch_size = int(training_config.get("batch_size", 128))
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
    checkpoint_config = training_config.get("checkpoint", {})
    save_dir = Path(checkpoint_config.get("save_dir", f"checkpoints/{stage_name}"))
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

    # 4. Curriculum Learning Callback
    if curriculum_config.get("enabled", False):
        curriculum_callback = CurriculumCallback(
            start_stage=int(curriculum_config.get("start_stage", 0)),
            end_stage=int(curriculum_config.get("end_stage", 6)),
            epochs_per_stage=int(curriculum_config.get("epochs_per_stage", 10)),
        )
        callbacks.append(curriculum_callback)
        print(
            f"📈 Curriculum Learning: "
            f"stage {curriculum_config.get('start_stage', 0)} -> "
            f"{curriculum_config.get('end_stage', 6)}, "
            f"{curriculum_config.get('epochs_per_stage', 10)} epochs/stage"
        )

    # 5. Visualization Callback
    vis_config = training_config.get("visualization", {})
    if vis_config.get("enabled", True):
        vis_callback = VisualizationCallback(
            num_samples=int(vis_config.get("num_samples", 4)),
            log_metrics=vis_config.get("log_metrics", True),
            log_interval=int(vis_config.get("log_interval", 1)),
            prefix="Validation" if stage_name != "structural" else "Train",
        )
        callbacks.append(vis_callback)
        print(f"🎨 Visualization: {vis_config.get('num_samples', 4)} samples")

    # =========================================================================
    # Trainer 配置
    # =========================================================================

    # 设备配置
    accelerator = device_config.get("accelerator", "auto")
    precision = device_config.get("precision", "16-mixed")

    # 兼容旧配置格式
    if "type" in device_config:
        device_type = device_config["type"]
        if device_type == "cuda":
            accelerator = "gpu"
        elif device_type == "cpu":
            accelerator = "cpu"
            precision = "32"
        elif device_type == "xpu":
            accelerator = "xpu"

    # CPU 模式下使用 32 精度
    if accelerator == "cpu":
        precision = "32"

    trainer = pl.Trainer(
        # 基础配置
        max_epochs=int(training_config.get("epochs", 50)),
        accelerator=accelerator,
        devices="auto",
        precision=precision,
        # 核心: 限制每个 epoch 的 batch 数量
        limit_train_batches=limit_train_batches,
        limit_val_batches=max(1, limit_train_batches // 10),
        # 梯度裁剪
        gradient_clip_val=float(training_config.get("grad_clip", 1.0)),
        # Callbacks & Logger
        callbacks=callbacks,
        logger=logger,
        # 日志频率
        log_every_n_steps=int(logging_config.get("log_interval", 10)),
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


def run_stage(
    config: Dict[str, Any],
    stage_name: str,
    init_from: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> str:
    """
    运行单个训练阶段

    Args:
        config: 全局配置
        stage_name: 阶段名称 ("structural", "dense", "finetune", "debug")
        init_from: 初始化权重路径 (用于迁移学习)
        resume_from: 断点续训 checkpoint 路径

    Returns:
        最佳 checkpoint 路径
    """
    print(f"\n{'=' * 60}")
    print(f"🚀 Starting stage: {stage_name}")
    print(f"{'=' * 60}")

    # 获取阶段配置
    stage_config = get_stage_config(config, stage_name)

    # 验证和打印配置
    validate_config(stage_config, stage_name)
    print_stage_config(stage_config, stage_name)

    training_config = stage_config.get("training", {})
    model_config = stage_config.get("model", {})
    data_config = stage_config.get("data", {})

    # 处理 init_from
    # 优先级：命令行参数 > 阶段配置 > None
    effective_init_from = init_from
    if effective_init_from is None:
        effective_init_from = stage_config.get("init_from")

    # 处理 "auto" 特殊值（由 run_all_stages 填充）
    if effective_init_from == "auto":
        effective_init_from = None  # 后续由 pipeline 处理

    # =========================================================================
    # 创建 DataModule
    # =========================================================================
    datamodule = InkTraceDataModule(
        img_size=int(data_config.get("img_size", 64)),
        batch_size=int(training_config.get("batch_size", 128)),
        epoch_length=int(training_config.get("epoch_length", 10000)),
        curriculum_stage=int(data_config.get("curriculum_stage", 0)),
        num_workers=int(data_config.get("num_workers", 8)),
        rust_threads=data_config.get("rust_threads"),
        pin_memory=data_config.get("pin_memory", True),
        persistent_workers=data_config.get("persistent_workers", True),
        keypoint_sigma=float(data_config.get("keypoint_sigma", 1.5)),
    )

    # =========================================================================
    # 创建模型
    # =========================================================================
    scheduler_config = training_config.get("scheduler", {})

    model = UnifiedTask(
        stage=stage_name if stage_name != "debug" else "dense",
        embed_dim=int(model_config.get("embed_dim", 192)),
        num_layers=int(model_config.get("num_layers", 4)),
        lr=float(training_config.get("lr", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
        loss_weights=training_config.get("loss_weights"),
        mask_ratio=float(model_config.get("mask_ratio", 0.6)),
        mask_strategy=model_config.get("mask_strategy", "block"),
        grad_clip=float(training_config.get("grad_clip", 1.0)),
        scheduler_type=scheduler_config.get("type", "onecycle"),
        warmup_epochs=int(scheduler_config.get("warmup_epochs", 2)),
        pct_start=float(scheduler_config.get("pct_start", 0.1)),
    )

    # 从 checkpoint 初始化权重 (迁移学习)
    if effective_init_from and not resume_from:
        freeze_encoder = stage_config.get("freeze_encoder", False)
        model.load_pretrained_weights(
            effective_init_from, strict=False, freeze_encoder=freeze_encoder
        )

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
        ckpt_path=resume_from,
    )

    # 返回最佳 checkpoint 路径
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\n✅ Stage {stage_name} completed!")
    print(f"   Best checkpoint: {best_ckpt}")

    return best_ckpt


def run_all_stages(config: Dict[str, Any], start_stage: Optional[str] = None) -> str:
    """
    运行完整的多阶段训练流水线

    按 pipeline.order 中定义的顺序执行各阶段
    自动在阶段之间传递权重

    Args:
        config: 全局配置
        start_stage: 从哪个阶段开始 (用于恢复训练)

    Returns:
        最终最优 checkpoint 路径
    """
    pipeline_config = config.get("pipeline", {})
    stage_order = pipeline_config.get("order", ["structural", "dense"])
    auto_transfer = pipeline_config.get("auto_transfer", True)

    print(f"\n{'#' * 60}")
    print("🎯 Multi-Stage Training Pipeline")
    print(f"   Stages: {' -> '.join(stage_order)}")
    print(f"   Auto transfer: {auto_transfer}")
    print(f"{'#' * 60}")

    # 确定起始点
    start_idx = 0
    if start_stage:
        try:
            start_idx = stage_order.index(start_stage)
            print(f"📍 Starting from stage: {start_stage}")
        except ValueError:
            print(f"⚠️  Stage '{start_stage}' not in pipeline, starting from beginning")

    last_ckpt = None

    for idx, stage_name in enumerate(stage_order):
        if idx < start_idx:
            continue

        # 确定 init_from
        init_from = None
        if auto_transfer and last_ckpt:
            init_from = last_ckpt
            print(f"\n🔗 Transferring weights from: {last_ckpt}")

        # 检查阶段配置中的 init_from
        stage_config = get_stage_config(config, stage_name)
        stage_init = stage_config.get("init_from")
        if stage_init and stage_init != "auto":
            init_from = stage_init

        # 运行阶段
        best_ckpt = run_stage(config, stage_name, init_from=init_from)
        last_ckpt = best_ckpt

    print(f"\n{'#' * 60}")
    print("🎉 All stages completed!")
    print(f"   Final checkpoint: {last_ckpt}")
    print(f"{'#' * 60}\n")

    return last_ckpt


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="InkTrace PyTorch Lightning Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single stage training
  python train_pl.py --config configs/default.yaml --stage structural
  python train_pl.py --config configs/default.yaml --stage dense

  # Multi-stage pipeline
  python train_pl.py --config configs/default.yaml --run-all-stages

  # Resume training
  python train_pl.py --config configs/default.yaml --stage dense --resume checkpoints/dense/last.ckpt

  # Transfer learning
  python train_pl.py --config configs/default.yaml --stage dense --init_from checkpoints/structural/best.ckpt

  # Quick debug
  python train_pl.py --config configs/default.yaml --stage debug
        """,
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
        help="训练阶段 (structural, dense, finetune, debug 或其他自定义阶段)",
    )
    parser.add_argument(
        "--run-all-stages",
        action="store_true",
        help="运行 pipeline 中定义的所有阶段",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        help="多阶段训练时从哪个阶段开始 (用于恢复)",
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
    parser.add_argument("--seed", type=int, default=114514, help="随机种子")

    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # CLI 参数覆盖配置
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # 设置随机种子
    pl.seed_everything(args.seed, workers=True)

    # 运行训练
    if args.run_all_stages:
        run_all_stages(config, start_stage=args.start_from)
    elif args.stage:
        run_stage(
            config,
            args.stage,
            init_from=args.init_from,
            resume_from=args.resume,
        )
    else:
        # 默认列出可用阶段
        stages = config.get("stages", {})
        print("\n可用的训练阶段:")
        for name, stage_config in stages.items():
            desc = stage_config.get("description", "")
            print(f"  - {name}: {desc}")
        print("\n使用 --stage <name> 指定阶段，或 --run-all-stages 运行完整流水线\n")


if __name__ == "__main__":
    main()
