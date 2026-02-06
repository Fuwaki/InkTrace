"""
PyTorch Lightning Module for InkTrace

支持两种训练阶段：
- structural: 结构预训练 (遮挡重建)
- dense: 密集预测训练 (多任务学习)

核心改进：
- 使用 self.trainer.estimated_stepping_batches 自动计算 OneCycleLR 步数
- 支持多种学习率调度器 (OneCycleLR, CosineAnnealingLR, Constant)
- 自动处理 Curriculum Learning
- 简化的权重加载和迁移学习
"""

import torch
import torch.optim as optim
import pytorch_lightning as pl
from typing import Dict, Optional, Literal

from models import ModelFactory, MaskingGenerator, StructuralPretrainLoss
from losses import DenseLoss
from vis_core import compute_metrics


class UnifiedTask(pl.LightningModule):
    """
    统一的 Lightning Module，支持 structural 和 dense 两种训练阶段

    Args:
        stage: 训练阶段 ("structural" 或 "dense")
        embed_dim: Encoder embedding 维度
        num_layers: Transformer 层数
        lr: 学习率
        weight_decay: 权重衰减
        loss_weights: Dense Loss 权重配置 (仅 dense 阶段)
        mask_ratio: 遮挡比例 (仅 structural 阶段)
        mask_strategy: 遮挡策略 (仅 structural 阶段)
        grad_clip: 梯度裁剪阈值
        scheduler_type: 学习率调度器类型 ("onecycle", "cosine", "constant")
        warmup_epochs: 预热轮数
        pct_start: OneCycleLR warmup 占比
    """

    def __init__(
        self,
        stage: Literal["structural", "dense"] = "dense",
        embed_dim: int = 192,  # 与 configs/default.yaml 一致
        num_layers: int = 4,   # 与 configs/default.yaml 一致
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_weights: Optional[Dict[str, float]] = None,
        mask_ratio: float = 0.6,
        mask_strategy: str = "block",
        grad_clip: float = 1.0,
        scheduler_type: str = "onecycle",
        warmup_epochs: int = 2,
        pct_start: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.stage = stage
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.pct_start = pct_start

        # 创建模型
        full_heads = stage == "dense"
        self.model = ModelFactory.create_unified_model(
            embed_dim=embed_dim,
            num_layers=num_layers,
            full_heads=full_heads,
            device="cpu",  # Lightning 会处理设备迁移
        )

        # 根据阶段设置 Loss 和辅助组件
        if stage == "structural":
            self.mask_gen = MaskingGenerator(
                mask_ratio=mask_ratio,
                strategy=mask_strategy,
            )
            self.criterion = StructuralPretrainLoss()
        else:  # dense
            self.criterion = DenseLoss(weights=loss_weights)

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """单步训练"""
        imgs, targets = batch

        if self.stage == "structural":
            return self._structural_step(imgs, targets)
        else:
            return self._dense_step(imgs, targets)

    def _structural_step(self, imgs, targets):
        """Structural 预训练步骤"""
        gt_skel = targets["skeleton"]
        gt_tan = targets["tangent"]

        # 生成遮挡
        masked_imgs, mask = self.mask_gen(imgs)

        # 前向传播
        outputs = self.model.pretrain_forward(masked_imgs)
        pred_skel = outputs["skeleton"]
        pred_tan = outputs["tangent"]

        # 计算损失
        losses = self.criterion(pred_skel, pred_tan, gt_skel, gt_tan, mask)

        # 日志记录
        self.log("train/loss", losses["total"], prog_bar=True)
        self.log("train/loss_skeleton", losses["loss_skeleton"])
        self.log("train/loss_tangent", losses["loss_tangent"])

        return losses["total"]

    def _dense_step(self, imgs, targets):
        """Dense 训练步骤"""
        # 前向传播
        outputs = self.model(imgs)
        losses = self.criterion(outputs, targets)

        # 检查 NaN/Inf
        if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
            self.log("train/nan_count", 1.0)
            return None  # 跳过这个 batch

        # 日志记录
        self.log("train/loss", losses["total"], prog_bar=True)
        self.log("train/loss_skel", losses["loss_skel"])
        self.log("train/loss_keys", losses["loss_keys"])
        self.log("train/loss_tan", losses["loss_tan"])
        self.log("train/loss_width", losses["loss_width"])
        self.log("train/loss_off", losses["loss_off"])

        return losses["total"]

    def validation_step(self, batch, batch_idx):
        """验证步骤（可选）"""
        imgs, targets = batch

        if self.stage == "structural":
            masked_imgs, mask = self.mask_gen(imgs)
            outputs = self.model.pretrain_forward(masked_imgs)
            losses = self.criterion(
                outputs["skeleton"],
                outputs["tangent"],
                targets["skeleton"],
                targets["tangent"],
                mask,
            )
        else:
            outputs = self.model(imgs)
            losses = self.criterion(outputs, targets)

        # 记录总 loss
        self.log("val/loss", losses["total"], prog_bar=True, sync_dist=True)

        # Dense 阶段：计算详细评估指标
        if self.stage == "dense":
            metrics = compute_metrics(outputs, targets)

            # 记录到 TensorBoard
            self.log("val/iou", metrics["skel_iou"], sync_dist=True)
            self.log("val/precision", metrics["skel_precision"], sync_dist=True)
            self.log("val/recall", metrics["skel_recall"], sync_dist=True)
            self.log("val/f1", metrics["skel_f1"], sync_dist=True)
            self.log("val/kp_topo_recall", metrics["kp_topo_recall"], sync_dist=True)
            self.log("val/kp_geo_recall", metrics["kp_geo_recall"], sync_dist=True)

        return losses["total"]

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器

        支持多种调度器：
        - onecycle: OneCycleLR (推荐，训练效果最好)
        - cosine: CosineAnnealingLR (适合微调)
        - constant: 固定学习率 (调试用)

        关键点：使用 self.trainer.estimated_stepping_batches 自动计算总步数
        """
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # 使用 Lightning 内置的步数估计
        total_steps = self.trainer.estimated_stepping_batches
        print(f"\n📊 Scheduler: {self.scheduler_type}")
        print(f"   Total steps: {total_steps}")
        print(f"   Learning rate: {self.lr}")

        if self.scheduler_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
                pct_start=self.pct_start,
                anneal_strategy="cos",
                div_factor=25.0,  # 初始 lr = max_lr / 25
                final_div_factor=1e4,  # 最终 lr = max_lr / 1e4
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif self.scheduler_type == "cosine":
            # CosineAnnealingLR 按 epoch 更新
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.lr * 0.01,  # 最终 lr = 1% of initial
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        else:  # constant
            # 固定学习率，不使用调度器
            return optimizer

    def on_train_epoch_start(self):
        """Epoch 开始时的回调"""
        # 记录当前 curriculum stage (如果 DataModule 支持)
        if hasattr(self.trainer, "datamodule") and hasattr(
            self.trainer.datamodule, "curriculum_stage"
        ):
            self.log(
                "curriculum/stage",
                float(self.trainer.datamodule.curriculum_stage),
            )

        # 记录当前 epoch
        self.log("train/epoch", float(self.current_epoch))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """每个 batch 结束时的回调，用于监控训练健康度"""
        # 记录梯度范数 (每 100 步)
        if batch_idx % 100 == 0 and outputs is not None:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            self.log("train/grad_norm", total_norm)

    # =========================================================================
    # 权重加载工具方法
    # =========================================================================

    def load_pretrained_weights(
        self,
        checkpoint_path: str,
        strict: bool = False,
        freeze_encoder: bool = False,
    ):
        """
        从 checkpoint 加载预训练权重

        Args:
            checkpoint_path: checkpoint 文件路径
            strict: 是否严格匹配 (False 允许部分加载，适合迁移学习)
            freeze_encoder: 是否冻结 Encoder 权重
        """
        print(f"📦 Loading weights from: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # 支持多种 checkpoint 格式
        if "state_dict" in ckpt:
            # Lightning checkpoint
            state_dict = ckpt["state_dict"]
            # 移除 "model." 前缀 (如果存在)
            state_dict = {
                k.replace("model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
        elif "model_state_dict" in ckpt:
            # 旧版手动 checkpoint
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # 加载权重
        incompatible = self.model.load_state_dict(state_dict, strict=strict)

        if incompatible.missing_keys:
            print(f"  ⚠️ Missing keys: {len(incompatible.missing_keys)}")
            if len(incompatible.missing_keys) <= 10:
                for k in incompatible.missing_keys:
                    print(f"     - {k}")
        if incompatible.unexpected_keys:
            print(f"  ⚠️ Unexpected keys: {len(incompatible.unexpected_keys)}")

        # 冻结 Encoder
        if freeze_encoder:
            print("  🔒 Freezing encoder weights")
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        print("  ✅ Weights loaded successfully")

    @classmethod
    def load_from_checkpoint_with_stage(
        cls,
        checkpoint_path: str,
        stage: Literal["structural", "dense"],
        strict: bool = False,
        **kwargs,
    ) -> "UnifiedTask":
        """
        从 checkpoint 加载模型，同时支持切换训练阶段

        用于从 structural 迁移到 dense 阶段

        Args:
            checkpoint_path: checkpoint 路径
            stage: 目标训练阶段
            strict: 是否严格匹配
            **kwargs: 传递给 __init__ 的额外参数
        """
        # 加载 checkpoint 获取超参数
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        hparams = ckpt.get("hyper_parameters", {})

        # 从 checkpoint 获取模型配置，但使用新的 stage
        model_kwargs = {
            "embed_dim": hparams.get("embed_dim", 192),  # 与 configs/default.yaml 一致
            "num_layers": hparams.get("num_layers", 4),   # 与 configs/default.yaml 一致
            "stage": stage,  # 使用新的 stage
        }
        model_kwargs.update(kwargs)

        # 创建新模型
        model = cls(**model_kwargs)

        # 加载权重 (非严格模式，因为 head 可能不同)
        model.load_pretrained_weights(checkpoint_path, strict=strict)

        return model


class CurriculumCallback(pl.Callback):
    """
    Curriculum Learning 回调

    根据 epoch 自动更新数据集的 curriculum stage
    """

    def __init__(
        self,
        start_stage: int = 0,
        end_stage: int = 9,
        epochs_per_stage: int = 10,
    ):
        super().__init__()
        self.start_stage = start_stage
        self.end_stage = end_stage
        self.epochs_per_stage = epochs_per_stage
        self.current_stage = start_stage

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """每个 epoch 开始时检查是否需要更新 curriculum"""
        epoch = trainer.current_epoch
        target_stage = self.start_stage + (epoch // self.epochs_per_stage)
        target_stage = min(target_stage, self.end_stage)

        if target_stage != self.current_stage:
            old_stage = self.current_stage
            self.current_stage = target_stage

            # 更新 DataModule 的 curriculum
            if hasattr(trainer, "datamodule") and hasattr(
                trainer.datamodule, "set_curriculum"
            ):
                trainer.datamodule.set_curriculum(target_stage)
                print(f"\n📈 Curriculum Update: Stage {old_stage} -> {target_stage}")

            # 记录到日志
            pl_module.log("curriculum/stage", float(target_stage))
