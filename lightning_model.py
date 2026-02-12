
import torch
import torch.optim as optim
import pytorch_lightning as pl
from typing import Dict, Optional, Literal
from timm.optim.adan import Adan

from models import ModelFactory, MaskingGenerator, StructuralPretrainLoss
from losses import DenseLoss
from vis_core import compute_metrics


class UnifiedTask(pl.LightningModule):
    """

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
        embed_dim: int = 64,           # Encoder embedding 维度
        decoder_mid_channels: Optional[int] = None,  # Decoder 中间层通道数
        num_layers: int = 6,              # Transformer 层数
        num_heads: int = 6,               # Attention heads
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_weights: Optional[Dict[str, float]] = None,
        mask_ratio: float = 0.6,
        mask_strategy: str = "block",
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.stage = stage
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        # 创建模型
        # structural: 只输出 skeleton + tangent (full_heads=False)
        # dense/finetune/debug: 输出全部 5 个头 (full_heads=True)
        full_heads = stage != "structural"
        self.model = ModelFactory.create_unified_model(
            embed_dim=embed_dim,
            decoder_mid_channels=decoder_mid_channels,
            num_heads=num_heads,
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

        优化器: Adan (Adaptive Nesterov Momentum)
        调度器: CosineAnnealingLR (稳定收敛)

        CosineAnnealingLR 优势：
        - 平滑学习率衰减
        - 稳定性好
        - 适合多阶段训练
        """
        optimizer = Adan(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.98, 0.92, 0.99),
            eps=1e-8,
            no_prox=False,
            caution=False,
        )

        total_epochs = self.trainer.max_epochs
        print(f"\n📊 Optimizer: Adan (lr={self.lr}, wd={self.weight_decay})")
        print(f"📊 Scheduler: CosineAnnealingLR")
        print(f"   Total epochs: {total_epochs}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=self.lr * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

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

    # =========================================================================
    # Checkpoint 钩子（保存/加载元数据）
    # =========================================================================

    def on_save_checkpoint(self, checkpoint):
        """
        Lightning 钩子：保存 checkpoint 时添加元数据

        Args:
            checkpoint: Lightning checkpoint 字典
        """
        from datetime import datetime

        # 添加自定义元数据
        checkpoint['metadata'] = {
            'model_version': 'v1.0',
            'save_time': datetime.now().isoformat(),
            'model_type': self.hparams.get('stage', 'unknown'),
        }
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """
        Lightning 钩子：加载 checkpoint 时恢复元数据

        Args:
            checkpoint: Lightning checkpoint 字典
        """
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            print(f"\n{'='*60}")
            print(f"📦 Checkpoint Metadata:")
            print(f"   Version: {metadata.get('model_version', 'unknown')}")
            print(f"   Saved at: {metadata.get('save_time', 'unknown')}")
            print(f"   Model type: {metadata.get('model_type', 'unknown')}")
            print(f"{'='*60}\n")
        return checkpoint

# CurriculumCallback 已移除 - curriculum stage 现在通过配置文件静态设置
# 参考：conf/stage/curriculum_*.yaml 配置文件
