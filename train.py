#!/usr/bin/env python3
"""
统一训练脚本 - 替代 train_structural.py 和 train_dense.py

特性:
  - 单一脚本支持所有训练模式
  - YAML 配置文件
  - 自动 checkpoint 管理
  - 多阶段训练
  - 简单的恢复训练

使用方法:
    # 单阶段训练（structural pretrain）
    python train.py --config configs/default.yaml --stage structural

    # 单阶段训练（dense）
    python train.py --config configs/default.yaml --stage dense --init_from checkpoints/structural/checkpoint_best.pth

    # 多阶段自动训练
    python train.py --config configs/default.yaml --run-all-stages

    # 恢复训练（自动检测配置）
    python train.py --resume checkpoints/structural/checkpoint_latest.pth
"""

import argparse
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import ModelFactory, MaskingGenerator, StructuralPretrainLoss
from losses import DenseLoss
from datasets_v2 import DenseInkTraceDataset, collate_dense_batch
from train_lib import Config, BaseTrainer
from visualize_dense import DenseVisualizer


# ============================================================================
# Structural Pretraining Trainer
# ============================================================================


class StructuralTrainer(BaseTrainer):
    """结构预训练训练器"""

    def __init__(self, config: Config, init_from: str = None):
        super().__init__(config, stage_name="structural")

        # 模型
        self.model = ModelFactory.create_unified_model(
            embed_dim=config.model["embed_dim"],
            num_layers=config.model["num_layers"],
            full_heads=False,
            device=self.device,
        )

        # 从 checkpoint 初始化
        if init_from:
            self.load_checkpoint(init_from, load_optimizer=False)

        # 掩码生成器 & 损失函数
        self.mask_gen = MaskingGenerator(
            mask_ratio=config.training.get("mask_ratio", 0.6),
            strategy=config.training.get("mask_strategy", "block"),
        )
        self.criterion = StructuralPretrainLoss()

        # 优化器 & 调度器
        lr = float(config.training["lr"])
        weight_decay = float(config.training.get("weight_decay", 1e-4))
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        steps_per_epoch = math.ceil(
            config.training["epoch_length"] / config.training["batch_size"]
        )
        total_steps = steps_per_epoch * config.training["epochs"]
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
        )

        print(f"\n{'=' * 60}")
        print(f"Structural Pretraining")
        print(
            f"  Model: embed_dim={config.model['embed_dim']}, num_layers={config.model['num_layers']}"
        )
        print(f"  Mask ratio: {config.training.get('mask_ratio', 0.6)}")
        print(
            f"  Training: lr={config.training['lr']}, epochs={config.training['epochs']}"
        )
        print(f"{'=' * 60}\n")

    def train_step(self, batch):
        # 新数据集返回 (imgs, targets) 元组
        imgs, targets = batch
        imgs = imgs.to(self.device)
        gt_skel = targets["skeleton"].to(self.device)
        gt_tan = targets["tangent"].to(self.device)

        # 生成掩码
        masked_imgs, mask = self.mask_gen(imgs)
        mask = mask.to(self.device)

        # 前向传播
        self.optimizer.zero_grad()
        outputs = self.model.pretrain_forward(masked_imgs)
        pred_skel = outputs["skeleton"]
        pred_tan = outputs["tangent"]

        # 损失
        losses = self.criterion(pred_skel, pred_tan, gt_skel, gt_tan, mask)
        loss = losses["total"]

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.training.get("grad_clip", 1.0),
        )
        self.optimizer.step()
        self.scheduler.step()

        return {
            "total": loss.item(),
            "skeleton": losses["loss_skeleton"].item(),
            "tangent": losses["loss_tangent"].item(),
        }


# ============================================================================
# Dense Training Trainer (with Curriculum Learning Support)
# ============================================================================


class DenseTrainer(BaseTrainer):
    """
    Dense 训练器 - 支持渐进式训练 (Curriculum Learning)

    Curriculum Stages (Stage 0-9):
      - Stage 0: 单笔画
      - Stage 1-3: 多独立笔画（递增: 1-3, 2-5, 3-8）
      - Stage 4-6: 多段连续笔画（递增: 2-3, 3-5, 4-8）
      - Stage 7-9: 混合模式（多条多段路径）
    """

    def __init__(self, config: Config, init_from: str = None):
        super().__init__(config, stage_name="dense")

        # 当前 curriculum 阶段
        self.curriculum_stage = config.data.get("curriculum_stage", 0)
        self.dataset = None  # Will be set later

        # Curriculum 配置
        self.curriculum_config = config.training.get("curriculum", {})
        self.curriculum_enabled = self.curriculum_config.get("enabled", False)
        self.curriculum_epochs_per_stage = self.curriculum_config.get(
            "epochs_per_stage", 10
        )
        self.curriculum_start_stage = self.curriculum_config.get("start_stage", 0)
        self.curriculum_end_stage = self.curriculum_config.get("end_stage", 9)

        # 模型
        self.model = ModelFactory.create_unified_model(
            embed_dim=config.model.get("embed_dim", 128),
            num_layers=config.model.get("num_layers", 4),
            full_heads=True,
            device=self.device,
        )

        # 从 checkpoint 初始化 (迁移学习)
        if init_from:
            state = self.load_checkpoint(
                init_from, load_optimizer=False, strict=False, reset_epoch=True
            )
            # 验证配置匹配
            ckpt_embed_dim = state["config"].model.get("embed_dim", 128)
            if ckpt_embed_dim != config.model.get("embed_dim", 128):
                print(f"  Warning: embed_dim mismatch! Checkpoint has {ckpt_embed_dim}")

        # 损失函数
        loss_weights = config.training.get("loss_weights", None)
        self.criterion = DenseLoss(weights=loss_weights)

        # 优化器 & 调度器
        lr = float(config.training["lr"])
        weight_decay = float(config.training.get("weight_decay", 1e-4))
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        steps_per_epoch = math.ceil(
            config.training["epoch_length"] / config.training["batch_size"]
        )
        total_steps = steps_per_epoch * config.training["epochs"]
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=lr, total_steps=total_steps
        )

        self._print_info()

    def _print_info(self):
        """打印训练配置信息"""
        print(f"\n{'=' * 60}")
        print(f"Dense Training (Curriculum Learning)")
        print(
            f"  Model: embed_dim={self.config.model.get('embed_dim', 128)}, "
            f"num_layers={self.config.model.get('num_layers', 4)}"
        )
        print(
            f"  Training: lr={self.config.training['lr']}, "
            f"epochs={self.config.training['epochs']}"
        )
        print(f"  Initial Curriculum Stage: {self.curriculum_stage}")
        if self.curriculum_enabled:
            print(
                f"  Curriculum: ENABLED (stage {self.curriculum_start_stage} -> "
                f"{self.curriculum_end_stage}, {self.curriculum_epochs_per_stage} epochs/stage)"
            )
        else:
            print(f"  Curriculum: DISABLED (fixed stage)")
        print(f"{'=' * 60}\n")

    def set_dataset(self, dataset: DenseInkTraceDataset):
        """设置数据集引用，用于动态调整 curriculum"""
        self.dataset = dataset

    def update_curriculum(self, epoch: int):
        """
        根据 epoch 更新 curriculum 阶段

        Returns:
            bool: 是否发生了阶段切换
        """
        if not self.curriculum_enabled or self.dataset is None:
            return False

        # 计算当前应该处于哪个阶段
        relative_epoch = epoch
        target_stage = self.curriculum_start_stage + (
            relative_epoch // self.curriculum_epochs_per_stage
        )
        target_stage = min(target_stage, self.curriculum_end_stage)

        if target_stage != self.curriculum_stage:
            old_stage = self.curriculum_stage
            self.curriculum_stage = target_stage
            self.dataset.set_curriculum(target_stage)
            print(f"\n📈 Curriculum Update: Stage {old_stage} -> {target_stage}")
            return True
        return False

    def train_step(self, batch):
        imgs, targets = batch
        imgs = imgs.to(self.device)
        targets = {k: v.to(self.device) for k, v in targets.items()}

        # 前向传播
        self.optimizer.zero_grad()
        with torch.amp.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            outputs = self.model(imgs)
            losses = self.criterion(outputs, targets)
            loss = losses["total"]

        # 反向传播
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ NaN/Inf loss detected, skipping batch")
            return {k: 0.0 for k in losses.keys()}

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.training.get("grad_clip", 1.0),
        )
        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() for k, v in losses.items()}

    def _epoch_end(self, epoch: int, avg_losses):
        """重写 epoch 结束处理，添加 curriculum 更新"""
        # 调用父类的 epoch 结束逻辑
        super()._epoch_end(epoch, avg_losses)

        # 更新 curriculum
        self.update_curriculum(epoch + 1)

    def set_dataloader(self, dataloader):
        """设置 dataloader 引用，用于可视化"""
        self.dataloader = dataloader
        # 初始化可视化器
        self.visualizer = DenseVisualizer(
            writer=self.writer,
            device=self.device,
            num_samples=4,
        )

    def evaluate(self):
        """在 TensorBoard 中生成可视化"""
        if not hasattr(self, "visualizer") or not hasattr(self, "dataloader"):
            return

        metrics = self.visualizer.visualize(
            model=self.model,
            dataloader=self.dataloader,
            global_step=self.global_step,
            prefix="Dense",
        )

        # 打印指标
        print(f"  📊 Eval: IoU={metrics['skel_iou']:.3f}, F1={metrics['skel_f1']:.3f}")


# ============================================================================
# 命令行接口
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="统一训练脚本")

    # 配置
    parser.add_argument("--config", type=str, help="YAML 配置文件路径")
    parser.add_argument(
        "--stage", type=str, choices=["structural", "dense"], help="训练阶段"
    )
    parser.add_argument(
        "--run-all-stages", action="store_true", help="自动运行所有阶段"
    )

    # Checkpoint
    parser.add_argument("--resume", type=str, help="从 checkpoint 恢复训练")
    parser.add_argument(
        "--init_from", type=str, help="从 checkpoint 初始化模型（新训练）"
    )

    # 覆盖配置（可选）
    parser.add_argument("--lr", type=float, help="覆盖学习率")
    parser.add_argument("--epochs", type=int, help="覆盖训练轮数")
    parser.add_argument("--batch_size", type=int, help="覆盖批次大小")

    return parser.parse_args()


def create_dataloader(config: Config, stage: str):
    """创建数据加载器

    Args:
        config: 配置对象
        stage: 训练阶段 ('structural' 或 'dense')

    Returns:
        (dataloader, dataset) 元组
    """
    curriculum_stage = config.data.get("curriculum_stage", 0)

    dataset = DenseInkTraceDataset(
        img_size=config.data.get("img_size", 64),
        batch_size=config.training["batch_size"],
        epoch_length=config.training["epoch_length"],
        curriculum_stage=curriculum_stage,
        rust_threads=config.data.get("rust_threads", None),
    )

    num_workers = config.data.get("num_workers", 4)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training["batch_size"],
        num_workers=num_workers,
        collate_fn=collate_dense_batch,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return dataloader, dataset


def run_single_stage(args, config: Config, stage_name: str, init_from: str = None):
    """运行单个阶段

    Args:
        args: 命令行参数
        config: 配置对象
        stage_name: 阶段名称 ('structural' 或 'dense')
        init_from: 初始化权重路径

    Returns:
        最佳 checkpoint 路径
    """

    # 从配置中获取 stage 配置
    stage_config = None
    if config.stages:
        for stage in config.stages:
            if stage["name"] == stage_name:
                stage_config = stage
                break

    if stage_config:
        # 合并 stage 配置
        config = Config(
            model={**config.model, **stage_config.get("model", {})},
            training={**config.training, **stage_config.get("training", {})},
            data={**config.data, **stage_config.get("data", {})},
            logging={**config.logging, **stage_config.get("logging", {})},
            device=config.device,
        )
        if init_from is None:
            init_from = stage_config.get("init_from")

    # 创建 trainer
    if stage_name == "structural":
        trainer = StructuralTrainer(config, init_from=init_from)
        dataloader, _ = create_dataloader(config, stage_name)
    elif stage_name == "dense":
        trainer = DenseTrainer(config, init_from=init_from)
        dataloader, dataset = create_dataloader(config, stage_name)
        # 将 dataset 传递给 trainer，支持动态 curriculum
        trainer.set_dataset(dataset)
        # 设置 dataloader 用于可视化
        trainer.set_dataloader(dataloader)
    else:
        raise ValueError(f"Unknown stage: {stage_name}")

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 训练
    trainer.train(dataloader)

    return trainer.ckpt_manager.save_dir / "checkpoint_best.pth"


def main():
    args = parse_args()

    # 加载配置
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        # 向后兼容：从命令行参数创建配置
        config = Config.from_args(args)

    # 覆盖配置
    if args.lr:
        config.training["lr"] = args.lr
    if args.epochs:
        config.training["epochs"] = args.epochs
    if args.batch_size:
        config.training["batch_size"] = args.batch_size

    # 运行训练
    if args.resume:
        # 恢复训练 - 自动检测 stage
        ckpt_config = Config.from_checkpoint(args.resume)
        stage = ckpt_config.metadata.get("stage", "dense")
        run_single_stage(args, config, stage, init_from=None)

    elif args.run_all_stages:
        # 运行所有阶段
        if not config.stages:
            raise ValueError("No stages defined in config")

        last_ckpt = None
        for stage in config.stages:
            stage_name = stage["name"]
            print(f"\n{'#' * 60}")
            print(f"# Running stage: {stage_name}")
            print(f"{'#' * 60}\n")

            # init_from 指定或使用上一个阶段的 checkpoint
            init_from = stage.get("init_from")
            if init_from and last_ckpt:
                init_from = init_from.replace("*", str(last_ckpt))
            elif not init_from and last_ckpt:
                init_from = str(last_ckpt)

            best_ckpt = run_single_stage(args, config, stage_name, init_from=init_from)
            last_ckpt = best_ckpt

    else:
        # 运行单个阶段
        if not args.stage:
            raise ValueError("--stage or --run-all-stages is required")
        run_single_stage(args, config, args.stage, init_from=args.init_from)


if __name__ == "__main__":
    main()
