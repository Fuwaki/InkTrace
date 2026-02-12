import torch
import torch.nn as nn
import torch.nn.functional as F
from RepVit import Conv2d_BN
from dense_heads import DenseHeads
from modules import NeXtBlock, GatedCrossAttention

class UniversalDecoder(nn.Module):
    """
    统一解码器：使用 Gated Cross Attention 自动学习是否使用 skip connection

    核心设计:
    - GatedCrossAttention 内置零初始化门控 (gate)
    - 初始 gate=0 时，完全等价于没有 skip (预训练模式)
    - 训练过程中 gate 自动学习，逐渐开启 skip 信息流
    - 无需 if/else 分支，统一前向路径

    通道配置:
    - F1 (Encoder): 32 通道 (固定)
    - F2 (Encoder): 128 通道 (固定)
    - F3 (Encoder): embed_dim 通道 (可配置，默认 192)
    - Decoder 中间层: mid_channels (可配置，默认与 embed_dim 相同)
    """

    def __init__(self, embed_dim=192, mid_channels=None):
        """
        Args:
            embed_dim: Encoder F3 输出的 embedding 维度 (来自 config)
            mid_channels: Decoder 中间层通道数 (None 时使用 embed_dim)
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 中间层通道数：默认与 embed_dim 相同，保持一致性
        if mid_channels is None:
            mid_channels = embed_dim
        self.mid_channels = mid_channels

        # Project F3 from embed_dim to mid_channels if needed
        self.f3_proj = nn.Identity()
        if embed_dim != mid_channels:
            self.f3_proj = nn.Conv2d(embed_dim, mid_channels, 1)

        # Encoder skip dimensions (from encoder.py)
        # 保存为实例变量，用于运行时维度检查
        self.f2_channels = 128  # 固定值，与 encoder.stem2 输出一致
        self.f1_channels = 32   # 固定值，与 encoder.stem1 输出一致
        f2_channels = self.f2_channels
        f1_channels = self.f1_channels

        # Decoder 输出维度 (heads 之前)
        # 使用渐进式降维: mid_channels -> mid_channels//2 -> 64
        out_channels = max(64, mid_channels // 2)

        # ========== Layer 1: 8x8 -> 16x16 ==========
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Gated Cross Attention: Decoder (Query) + Encoder F2 (Key/Value)
        # gate=0 时退化为恒等映射，等价于无 skip
        self.cross_attn1 = GatedCrossAttention(
            dim_high=mid_channels, dim_skip=f2_channels, num_heads=4
        )
        self.conv1 = NeXtBlock(mid_channels, mid_channels, kernel_size=7)

        # ========== Layer 2: 16x16 -> 32x32 ==========
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Gated Cross Attention: Decoder + Encoder F1
        self.cross_attn2 = GatedCrossAttention(
            dim_high=mid_channels, dim_skip=f1_channels, num_heads=4
        )
        # 降维融合
        self.fusion2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = NeXtBlock(out_channels, out_channels, kernel_size=7)

        # ========== Layer 3: 32x32 -> 64x64 ==========
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = NeXtBlock(out_channels, out_channels, kernel_size=7)

        # ========== Prediction Heads ==========
        self.heads = DenseHeads(in_channels=out_channels, head_channels=64)

    def set_skip_mode(self, mode='frozen'):
        """
        设置 skip gate 的训练模式
        Args:
            mode: 'frozen' (预训练，gate=0且不更新) / 'learnable' (可学习)
        """
        for module in [self.cross_attn1, self.cross_attn2]:
            if mode == 'frozen':
                module.gate.data.zero_()
                module.gate.requires_grad = False
            elif mode == 'learnable':
                module.gate.requires_grad = True
            else:
                raise ValueError(f"Unknown mode: {mode}")

    def forward(self, features, use_skips=True):
        """
        Args:
            features: [f1, f2, f3] from Encoder (list) 或单独 f3 (tensor)
            use_skips: bool (保持兼容，但逻辑上由 gate 自动控制)
        Returns:
            outputs: dict with 'skeleton', 'tangent', and optionally others
        """
        # Handle input format
        if isinstance(features, (list, tuple)):
            f1, f2, f3 = features

            # 运行时维度检查：防止 encoder/decoder 通道不匹配
            if f1 is not None:
                assert f1.shape[1] == self.f1_channels, \
                    f"F1 通道不匹配: 期望 {self.f1_channels}, 实际 {f1.shape[1]}"
            if f2 is not None:
                assert f2.shape[1] == self.f2_channels, \
                    f"F2 通道不匹配: 期望 {self.f2_channels}, 实际 {f2.shape[1]}"
        else:
            f3 = features
            f1, f2 = None, None

        # 如果不使用 skip，强制设为 None (gate 会处理)
        if not use_skips:
            f1, f2 = None, None

        # Project F3 from embed_dim to mid_channels if needed
        f3 = self.f3_proj(f3)

        # ========== Block 1 (8 -> 16) ==========
        d1_up = self.up1(f3)  # [B, mid_channels, 16, 16]
        d1 = self.cross_attn1(d1_up, f2)  # Gated Cross Attention (skip if f2 is None)
        d1 = self.conv1(d1)

        # ========== Block 2 (16 -> 32) ==========
        d2_up = self.up2(d1)  # [B, mid_channels, 32, 32]
        d2 = self.cross_attn2(d2_up, f1)  # Gated Cross Attention
        d2 = self.fusion2(d2)  # 降维到 out_channels
        d2 = self.conv2(d2)

        # ========== Block 3 (32 -> 64) ==========
        d3_up = self.up3(d2)
        d3 = self.conv3(d3_up)  # [B, 64, 64, 64]

        # ========== Prediction ==========
        outputs = self.heads(d3)

        return outputs
