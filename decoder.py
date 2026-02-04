import torch
import torch.nn as nn
from RepVit import Conv2d_BN


class AttentionGate(nn.Module):
    """
    Attention Gate: Filter features from skip connection using signal from current decoder layer.
    """

    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Gate channels (from deeper layer)
            F_l: Skip connection channels (from shallower layer)
            F_int: Intermediate channels
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: Gate signal (Upsampled feature from prev decoder layer)
        x: Skip connection feature (from Encoder)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ConvBlock(nn.Module):
    """
    Standard Conv Block: Conv-BN-ReLU-Conv-BN-ReLU
    Using RepVGG style Conv2d_BN for efficiency if desired, but standard ResBlock is good.
    Here we use a simple residual block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d_BN(in_channels, out_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = Conv2d_BN(out_channels, out_channels, 3, 1, 1)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = Conv2d_BN(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x + res


class PixelDecoder(nn.Module):
    """
    轻量级像素解码器 (Phase 1 专用)
    输入: [B, 64, embed_dim] (来自 Encoder 的序列)
    输出: [B, 1, 64, 64] (重构图像)
    """
    def __init__(self, embed_dim=128):
        super().__init__()

        # 1. 序列转回特征图的预处理
        # 此时输入是序列，我们先把它 reshape 回 8x8
        self.embed_dim = embed_dim

        # 2. 上采样模块 (Upsample Block)
        # 结构: [上采样 -> 卷积 -> BN -> GELU]
        # 目标: 8 -> 16 -> 32 -> 64 (3次上采样)

        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), # 8 -> 16
            nn.Conv2d(embed_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), # 16 -> 32
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), # 32 -> 64
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )

        # 3. 输出层 (Output Layer)
        # 最后一层把通道压缩为 1，并用 Sigmoid 归一化到 0-1
        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, 64, embed_dim]  <-- 来自 Encoder 的输出

        # Step 1: Reshape (序列 -> 图像)
        # [B, 64, embed_dim] -> [B, embed_dim, 64] -> [B, embed_dim, 8, 8]
        B, N, C = x.shape
        H = W = int(N ** 0.5) # 根号64 = 8
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Step 2: 逐级放大
        x = self.layer1(x) # -> [B, 64, 16, 16]
        x = self.layer2(x) # -> [B, 32, 32, 32]
        x = self.layer3(x) # -> [B, 16, 64, 64]

        # Step 3: 输出图像
        img = self.head(x) # -> [B, 1, 64, 64]

        return img


class DenseDecoder(nn.Module):
    """
    Hybrid U-Net with Attention Gates for Dense Prediction Tasks.
    Takes multi-scale features from Encoder and progressively upsamples them.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        
        # Feature channels from encoder (Modified StrokeEncoder)
        # f1: 32 (32x32)
        # f2: 128 (16x16)
        # f3: embed_dim (8x8)

        # Project F3 from embed_dim to 128 if needed
        self.f3_proj = nn.Identity()
        if embed_dim != 128:
            self.f3_proj = nn.Conv2d(embed_dim, 128, 1)

        # Layer 1: 8x8 -> 16x16
        # Input: F3 (128). Skip: F2 (128).
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ag1 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv1 = ConvBlock(128 + 128, 128)  # Cat F3_up, F2_gated

        # Deep Supervision 1 (at 16x16)
        self.ds1_head = nn.Sequential(
            Conv2d_BN(128, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

        # Layer 2: 16x16 -> 32x32
        # Input: D1 (128). Skip: F1 (32).
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ag2 = AttentionGate(F_g=128, F_l=32, F_int=16)
        self.conv2 = ConvBlock(128 + 32, 64)

        # Deep Supervision 2 (at 32x32)
        self.ds2_head = nn.Sequential(
            Conv2d_BN(64, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 1, 1)
        )

        # Layer 3: 32x32 -> 64x64
        # Input: D2 (64). Skip: None
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = ConvBlock(64, 64)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3] from Encoder
              f1: [B, 32, 32, 32]
              f2: [B, 128, 16, 16]
              f3: [B, embed_dim, 8, 8]
        Returns:
            d3: [B, 64, 64, 64] Final Feature Map
            aux_outputs: dict containing 'ds1', 'ds2' if training
        """
        f1, f2, f3 = features

        # Project F3 to 128 channels if needed
        f3 = self.f3_proj(f3)

        # Block 1 (8 -> 16)
        d1_up = self.up1(f3)      # [B, 128, 16, 16]
        f2_gated = self.ag1(d1_up, f2) # [B, 128, 16, 16]
        d1 = torch.cat([d1_up, f2_gated], dim=1)
        d1 = self.conv1(d1)       # [B, 128, 16, 16]

        # Block 2 (16 -> 32)
        d2_up = self.up2(d1)      # [B, 128, 32, 32]
        f1_gated = self.ag2(d2_up, f1) # [B, 32, 32, 32]
        d2 = torch.cat([d2_up, f1_gated], dim=1)
        d2 = self.conv2(d2)       # [B, 64, 32, 32]

        # Block 3 (32 -> 64)
        d3_up = self.up3(d2)      # [B, 64, 64, 64]
        d3 = self.conv3(d3_up)    # [B, 64, 64, 64]

        aux_outputs = {}
        if self.training:
            ds1 = torch.sigmoid(self.ds1_head(d1))
            ds2 = torch.sigmoid(self.ds2_head(d2))
            aux_outputs['aux_skeleton_16'] = ds1
            aux_outputs['aux_skeleton_32'] = ds2
            
        return d3, aux_outputs
