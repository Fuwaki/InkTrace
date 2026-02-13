import torch
import torch.nn as nn
import torch.nn.functional as F
from dense_heads import DenseHeads
from modules import NeXtBlock, GatedCrossAttention

class UniversalDecoder(nn.Module):
    """
    适配 V3 Encoder 的瘦解码器

    核心设计:
    - GatedCrossAttention 内置零初始化门控 (gate)
    - 初始 gate=0 时，完全等价于没有 skip (预训练模式)
    - 训练过程中 gate 自动学习，逐渐开启 skip 信息流
    - 无需 if/else 分支，统一前向路径

    通道配置 (V3 Encoder):
    - F1 (Encoder): 16 通道
    - F2 (Encoder): 32 通道
    - F3 (Encoder): 64 通道
    - F4 (Encoder): embed_dim 通道 (默认 64，UNet 瓶颈，4×4)
    - Decoder 瘦身: c0(48) -> c1(32) -> c2(24) -> c3(16)
    - 从 4×4 逐层上采样到 64×64，共 4 级
    """

    def __init__(self, embed_dim=64, mid_channels=None):
        """
        Args:
            embed_dim: Encoder F4 输出的 embedding 维度 (默认 64，UNet 瓶颈)
            mid_channels: Decoder 起始通道数 (None 时使用 embed_dim，已废弃)
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Encoder skip 通道数 (V3 Encoder)
        self.f1_channels = 16   # Stage 1 输出
        self.f2_channels = 32   # Stage 2 输出
        self.f3_channels = 64   # Stage 3 输出

        # 瘦 Decoder 通道配置
        c0 = 48   # Block 0: 4→8 (与 f3=64 融合，需要一定容量)
        c1 = 32   # Block 1: 8→16
        c2 = 24   # Block 2: 16→32
        c3 = 16   # Block 3: 32→64 (输出前)

        # ========== Block 0: 4×4 → 8×8 (新增) ==========
        # 输入: f4 (embed_dim=64, 4×4)
        # Skip: f3 (64ch, 8×8)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj0 = nn.Sequential(
            nn.Conv2d(embed_dim, c0, 1, bias=False),
            nn.BatchNorm2d(c0),
        )
        self.cross_attn0 = GatedCrossAttention(
            dim_high=c0, dim_skip=self.f3_channels, num_heads=4
        )
        self.conv0 = NeXtBlock(c0, c0, kernel_size=7)

        # ========== Block 1: 8×8 → 16×16 ==========
        # Skip: f2 (32ch, 16×16)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj1 = nn.Sequential(
            nn.Conv2d(c0, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.cross_attn1 = GatedCrossAttention(
            dim_high=c1, dim_skip=self.f2_channels, num_heads=4
        )
        self.conv1 = NeXtBlock(c1, c1, kernel_size=7)

        # ========== Block 2: 16×16 → 32×32 ==========
        # Skip: f1 (16ch, 32×32)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj2 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.cross_attn2 = GatedCrossAttention(
            dim_high=c2, dim_skip=self.f1_channels, num_heads=4
        )
        self.conv2 = NeXtBlock(c2, c2, kernel_size=7)

        # ========== Block 3: 32×32 → 64×64 (无 skip) ==========
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj3 = nn.Sequential(
            nn.Conv2d(c2, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
        )
        self.conv3 = NeXtBlock(c3, c3, kernel_size=7)

        # ========== Prediction Heads ==========
        self.heads = DenseHeads(in_channels=c3, head_channels=c3)

    def set_skip_mode(self, mode='frozen'):
        """
        设置 skip gate 的训练模式
        Args:
            mode: 'frozen' (预训练，gate=0且不更新) / 'learnable' (可学习)
        """
        for module in [self.cross_attn0, self.cross_attn1, self.cross_attn2]:
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
            features: [f1, f2, f3, f4] from Encoder (list) 或单独 f4 (tensor)
            use_skips: bool (保持兼容，但逻辑上由 gate 自动控制)
        Returns:
            outputs: dict with 'skeleton', 'tangent', and optionally others
        """
        # ========== 解包 4 级特征 ==========
        if isinstance(features, (list, tuple)):
            f1, f2, f3, f4 = features

            # 运行时维度检查：防止 encoder/decoder 通道不匹配
            if f1 is not None:
                assert f1.shape[1] == self.f1_channels, \
                    f"F1 通道不匹配: 期望 {self.f1_channels}, 实际 {f1.shape[1]}"
            if f2 is not None:
                assert f2.shape[1] == self.f2_channels, \
                    f"F2 通道不匹配: 期望 {self.f2_channels}, 实际 {f2.shape[1]}"
            if f3 is not None:
                assert f3.shape[1] == self.f3_channels, \
                    f"F3 通道不匹配: 期望 {self.f3_channels}, 实际 {f3.shape[1]}"
        else:
            f4 = features
            f1, f2, f3 = None, None, None

        # 如果不使用 skip，强制设为 None (gate 会处理)
        if not use_skips:
            f1, f2, f3 = None, None, None

        # ========== Block 0: 4→8, skip=f3 ==========
        x = self.up0(f4)              # [B, 64, 8, 8]
        x = self.proj0(x)             # [B, 48, 8, 8]
        x = self.cross_attn0(x, f3)   # Gated fusion with f3
        x = self.conv0(x)             # [B, 48, 8, 8]

        # ========== Block 1: 8→16, skip=f2 ==========
        x = self.up1(x)               # [B, 48, 16, 16]
        x = self.proj1(x)             # [B, 32, 16, 16]
        x = self.cross_attn1(x, f2)   # Gated fusion with f2
        x = self.conv1(x)             # [B, 32, 16, 16]

        # ========== Block 2: 16→32, skip=f1 ==========
        x = self.up2(x)               # [B, 32, 32, 32]
        x = self.proj2(x)             # [B, 24, 32, 32]
        x = self.cross_attn2(x, f1)   # Gated fusion with f1
        x = self.conv2(x)             # [B, 24, 32, 32]

        # ========== Block 3: 32→64, no skip ==========
        x = self.up3(x)               # [B, 24, 64, 64]
        x = self.proj3(x)             # [B, 16, 64, 64]
        x = self.conv3(x)             # [B, 16, 64, 64]

        # ========== Prediction ==========
        outputs = self.heads(x)
        return outputs
