"""
统一的模型接口 (Refactored for V5)

结构:
- Encoder: encoder.py
- Decoder: decoder.py
- Heads: dense_heads.py
- Model Factory: models.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import StrokeEncoder
from decoder import UniversalDecoder

# =============================================================================
# Masking Generator (for Structural Pretraining)
# =============================================================================


class MaskingGenerator:
    """
    向量化优化的遮挡生成器 (High Performance)

    策略：
    - block: 随机遮挡若干个矩形块 (类似 MAE)
    - random: 随机像素遮挡
    """

    def __init__(self, mask_ratio=0.5, strategy="block", block_size=8):
        """
        Args:
            mask_ratio: 遮挡比例 (0.0 ~ 1.0)
            strategy: 遮挡策略 ('block', 'random')
            block_size: block 策略时的块大小
        """
        self.mask_ratio = mask_ratio
        self.strategy = strategy
        self.block_size = block_size

    def __call__(self, imgs):
        """
        Args:
            imgs: [B, 1, H, W] 输入图像
        Returns:
            masked_imgs: [B, 1, H, W] 遮挡后的图像
            mask: [B, 1, H, W] 遮挡掩码 (1 表示被遮挡)
        """
        B, C, H, W = imgs.shape
        device = imgs.device

        if self.strategy == "block":
            mask = self._generate_block_mask_vectorized(B, H, W, device)
        elif self.strategy == "random":
            mask = (torch.rand(B, 1, H, W, device=device) < self.mask_ratio).float()
        else:
            # Default to block
            mask = self._generate_block_mask_vectorized(B, H, W, device)

        # Apply mask: set masked regions to 0 (or background value)
        masked_imgs = imgs * (1 - mask)

        return masked_imgs, mask

    def _generate_block_mask_vectorized(self, B, H, W, device):
        """
        完全向量化的 Block Mask 生成
        原理：在 Grid 层级生成 Mask，然后插值放大
        性能：O(1) Python调用，全部GPU并行，10-100x加速
        """
        # 1. 计算 Grid 尺寸 (例如 64x64 img, 8 block -> 8x8 grid)
        H_grid = H // self.block_size
        W_grid = W // self.block_size
        num_blocks = H_grid * W_grid
        num_masked = int(num_blocks * self.mask_ratio)

        # 2. 生成随机噪声 [B, num_blocks]
        noise = torch.rand(B, num_blocks, device=device)

        # 3. 找出需要 Mask 的索引 (Top-K 逻辑)
        # argsort 的开销远小于几千次 Python 循环
        ids_shuffle = torch.argsort(noise, dim=1)  # [B, num_blocks]
        ids_mask = ids_shuffle[:, :num_masked]     # [B, num_masked] 取前 K 个

        # 4. 创建 Grid Mask [B, num_blocks]
        mask_flat = torch.zeros(B, num_blocks, device=device)
        # scatter_ 是完全并行的 GPU 操作
        mask_flat.scatter_(1, ids_mask, 1.0)

        # 5. Reshape 回 [B, 1, H_grid, W_grid]
        mask_grid = mask_flat.view(B, 1, H_grid, W_grid)

        # 6. 上采样回原图尺寸 (Nearest Neighbor = 完美的 Block 效果)
        # 这一步非常快，是 GPU 的拿手好戏
        mask = F.interpolate(mask_grid, size=(H, W), mode='nearest')

        return mask


# =============================================================================
# Structural Pretraining Loss
# =============================================================================


class StructuralPretrainLoss(nn.Module):
    """
    结构预训练 Loss
    核心目标：让 Encoder 学会从残缺输入推断完整的骨架结构和笔画方向

    L = L_skeleton + λ * L_tangent
    """

    def __init__(self, skeleton_weight=1.0, tangent_weight=1.0):
        super().__init__()
        self.skeleton_weight = skeleton_weight
        self.tangent_weight = tangent_weight

    def _dice_loss(self, pred, target, smooth=1.0):
        pred = pred.float().contiguous()
        target = target.float().contiguous()
        intersection = (pred * target).sum(dim=[2, 3])
        loss = 1 - (
            (2.0 * intersection + smooth)
            / (pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + smooth)
        )
        return loss.mean()

    def _bce_loss(self, pred, target):
        pred = pred.float().clamp(min=1e-6, max=1 - 1e-6)
        target = target.float()
        logits = torch.log(pred / (1 - pred))
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")

    def forward(self, pred_skeleton, pred_tangent, gt_skeleton, gt_tangent, mask=None):
        """
        Args:
            pred_skeleton: [B, 1, H, W] 预测骨架
            pred_tangent: [B, 2, H, W] 预测切向场
            gt_skeleton: [B, 1, H, W] GT 骨架
            gt_tangent: [B, 2, H, W] GT 切向场
            mask: [B, 1, H, W] 遮挡掩码 (可选，用于只在遮挡区域计算 loss)
        Returns:
            losses: dict
        """
        losses = {}

        # 1. Skeleton Loss (BCE + Dice)
        bce_skel = self._bce_loss(pred_skeleton, gt_skeleton)
        dice_skel = self._dice_loss(pred_skeleton, gt_skeleton)
        losses["loss_skeleton"] = self.skeleton_weight * (bce_skel + dice_skel)

        # 2. Tangent Loss (只在骨架区域计算)
        skel_mask = (gt_skeleton > 0.5).float()
        num_fg = skel_mask.sum().clamp(min=1.0)

        # L2 loss on tangent field
        l2_tan = (pred_tangent - gt_tangent) ** 2
        l2_tan = (l2_tan * skel_mask).sum() / num_fg / 2.0
        losses["loss_tangent"] = self.tangent_weight * l2_tan

        losses["total"] = losses["loss_skeleton"] + losses["loss_tangent"]

        return losses


class UnifiedModel(nn.Module):
    """
    统一模型架构 (V5)
    Encoder + UniversalDecoder

    支持两种模式：
    - 预训练模式 (use_skips=False): 强迫 Encoder 学习结构
    - 正式训练模式 (use_skips=True): 利用跳连提高精度
    """

    def __init__(self, encoder, decoder=None, full_heads=True):
        super().__init__()
        self.encoder = encoder

        # Infer embed_dim from encoder
        embed_dim = getattr(encoder, "token_embed", None)
        if embed_dim is not None:
            embed_dim = embed_dim.out_features
        else:
            embed_dim = 128

        # Decoder 默认使用与 encoder 相同的 embed_dim
        # 如果需要不同的中间层通道数，可以显式传入 decoder
        self.decoder = decoder or UniversalDecoder(
            embed_dim=embed_dim, full_heads=full_heads
        )
        self._use_skips = True  # Default to using skips

    @property
    def use_skips(self):
        return self._use_skips

    @use_skips.setter
    def use_skips(self, value):
        self._use_skips = value

    def forward(self, x, use_skips=None):
        """
        Args:
            x: [B, 1, 64, 64] 输入图像
            use_skips: bool, 是否使用跳连 (None 时使用默认值)
        Returns:
            outputs: dict with predictions
            aux_outputs: dict with deep supervision (training only)
        """
        if use_skips is None:
            use_skips = self._use_skips

        # 1. Encoder Pass
        if use_skips:
            features, embeddings = self.encoder(x, return_interm_layers=True)
        else:
            # 只获取 f3，不需要中间层
            features, embeddings = self.encoder(x, return_interm_layers=True)
            # 但我们会告诉 decoder 忽略 f1, f2

        # 2. Decoder Pass
        outputs = self.decoder(features, use_skips=use_skips)

        return outputs

    def pretrain_forward(self, x):
        """预训练专用 forward，强制关闭跳连"""
        return self.forward(x, use_skips=False)

    def finetune_forward(self, x):
        """微调专用 forward，启用跳连"""
        return self.forward(x, use_skips=True)


class ModelFactory:
    """
    模型工厂类
    """

    @staticmethod
    def create_unified_model(
        embed_dim=192,
        decoder_mid_channels=None,
        num_heads=6,
        num_layers=4,
        full_heads=True,
        device="cpu",
        encoder_ckpt=None,
    ):
        """
        创建统一模型 (推荐)

        Args:
            embed_dim: Encoder embedding 维度
            decoder_mid_channels: Decoder 中间层通道数 (None 时使用 embed_dim)
            num_heads: Transformer 注意力头数
            num_layers: Transformer 层数 (与 configs/default.yaml 一致)
            full_heads: 是否输出全部 5 个头
            device: 设备
            encoder_ckpt: 预训练 Encoder 权重路径
        """
        encoder = StrokeEncoder(
            in_channels=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=0.1,
        )

        # Load pretrained encoder if provided
        if encoder_ckpt:
            print(f"Loading pretrained encoder from {encoder_ckpt}")
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            if "encoder_state_dict" in ckpt:
                state_dict = ckpt["encoder_state_dict"]
            elif "model_state_dict" in ckpt:
                state_dict = {
                    k.replace("encoder.", ""): v
                    for k, v in ckpt["model_state_dict"].items()
                    if k.startswith("encoder.")
                }
            else:
                state_dict = ckpt
            encoder.load_state_dict(state_dict, strict=False)
            print("Encoder weights loaded.")

        # Create decoder with config-specified mid_channels
        decoder = UniversalDecoder(
            embed_dim=embed_dim,
            mid_channels=decoder_mid_channels,
            full_heads=full_heads,
        )

        model = UnifiedModel(encoder, decoder=decoder, full_heads=full_heads).to(device)
        return model

    @staticmethod
    def load_unified_model(checkpoint_path, device="cpu", full_heads=True):
        """加载统一模型"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 从 checkpoint 获取配置，或使用默认值
        config = checkpoint.get("config", {})
        embed_dim = config.get("embed_dim", 192)  # 与 configs/default.yaml 一致
        num_layers = config.get("num_layers", 4)   # 与 configs/default.yaml 一致

        model = ModelFactory.create_unified_model(
            embed_dim=embed_dim,
            num_layers=num_layers,
            full_heads=full_heads,
            device=device,
        )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "encoder_state_dict" in checkpoint:
            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            if "decoder_state_dict" in checkpoint:
                model.decoder.load_state_dict(checkpoint["decoder_state_dict"])

        return model
