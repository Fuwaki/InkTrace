import torch
import torch.nn as nn
import torch.nn.functional as F
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
    - F2 (Encoder): 64 通道 (timm RepViT)
    - F3 (Encoder): embed_dim 通道 (默认 64，UNet 瓶颈)
    - Decoder 渐进升维: c1(64) -> c2(96) -> c3(128) -> 64
    - 升维后降维，充分利用 skip connection 的高维特征
    """

    def __init__(self, embed_dim=64, mid_channels=None):
        """
        Args:
            embed_dim: Encoder F3 输出的 embedding 维度 (默认 64，UNet 瓶颈)
            mid_channels: Decoder 起始通道数 (None 时使用 embed_dim)
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
        self.f2_channels = 64   # timm RepViT Stage 2 输出
        self.f1_channels = 32   # timm RepViT Stage 1 输出
        f2_channels = self.f2_channels
        f1_channels = self.f1_channels

        # ========== 渐进式升维通道配置 ==========
        # mid_channels (64) -> c1 (64) -> c2 (96) -> c3 (128) -> 64
        # 逐层升维，充分利用 skip connection 的高维特征
        c1 = mid_channels           # 64
        c2 = int(mid_channels * 1.5)  # 96
        c3 = int(mid_channels * 2)    # 128

        # ========== Layer 1: 8x8 -> 16x16 ==========
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # 升维: c1 -> c2
        self.proj1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )
        # Gated Cross Attention: Decoder (Query) + Encoder F2 (Key/Value)
        self.cross_attn1 = GatedCrossAttention(
            dim_high=c2, dim_skip=f2_channels, num_heads=4
        )
        self.conv1 = NeXtBlock(c2, c2, kernel_size=7)

        # ========== Layer 2: 16x16 -> 32x32 ==========
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # 升维: c2 -> c3
        self.proj2 = nn.Sequential(
            nn.Conv2d(c2, c3, 1, bias=False),
            nn.BatchNorm2d(c3)
        )
        # Gated Cross Attention: Decoder + Encoder F1
        self.cross_attn2 = GatedCrossAttention(
            dim_high=c3, dim_skip=f1_channels, num_heads=4
        )
        self.conv2 = NeXtBlock(c3, c3, kernel_size=7)

        # ========== Layer 3: 32x32 -> 64x64 ==========
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # 降维: c3 -> 64
        self.proj3 = nn.Sequential(
            nn.Conv2d(c3, 64, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.conv3 = NeXtBlock(64, 64, kernel_size=7)

        # ========== Prediction Heads ==========
        self.heads = DenseHeads(in_channels=64, head_channels=64)

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
        d1_up = self.up1(f3)  # [B, c1, 16, 16]
        d1 = self.proj1(d1_up)  # 升维: c1 -> c2
        d1 = self.cross_attn1(d1, f2)  # Gated Cross Attention
        d1 = self.conv1(d1)

        # ========== Block 2 (16 -> 32) ==========
        d2_up = self.up2(d1)  # [B, c2, 32, 32]
        d2 = self.proj2(d2_up)  # 升维: c2 -> c3
        d2 = self.cross_attn2(d2, f1)  # Gated Cross Attention
        d2 = self.conv2(d2)

        # ========== Block 3 (32 -> 64) ==========
        d3_up = self.up3(d2)  # [B, c3, 64, 64]
        d3 = self.proj3(d3_up)  # 降维: c3 -> 64
        d3 = self.conv3(d3)  # [B, 64, 64, 64]

        # ========== Prediction ==========
        outputs = self.heads(d3)

        return outputs
