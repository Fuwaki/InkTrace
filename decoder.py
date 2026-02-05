import torch
import torch.nn as nn
from RepVit import Conv2d_BN


# =============================================================================
# Building Blocks
# =============================================================================


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


# =============================================================================
# Universal Decoder (Unified Architecture)
# =============================================================================


class UniversalDecoder(nn.Module):
    """
    统一解码器：通过 use_skips 开关控制是否使用跳连

    - use_skips=False (预训练模式):
        只用 f3，强迫 Encoder 在 bottleneck 编码完整的结构信息
    - use_skips=True (正式训练模式):
        使用 f1, f2 跳连，提高细节精度

    输出头：skeleton, tangent (预训练核心), junction, width, offset (可选)
    """

    def __init__(self, embed_dim=128, full_heads=True):
        """
        Args:
            embed_dim: Encoder 输出的 embedding 维度
            full_heads: 是否输出全部 5 个头 (False 时只输出 skeleton + tangent)
        """
        super().__init__()
        self.full_heads = full_heads

        # Project F3 from embed_dim to 128 if needed
        self.f3_proj = nn.Identity()
        if embed_dim != 128:
            self.f3_proj = nn.Conv2d(embed_dim, 128, 1)

        # ========== Layer 1: 8x8 -> 16x16 ==========
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Attention Gate for skip connection (only used when use_skips=True)
        self.ag1 = AttentionGate(F_g=128, F_l=128, F_int=64)
        # Two versions of conv: with skip (256 in) vs without skip (128 in)
        self.conv1_skip = ConvBlock(128 + 128, 128)
        self.conv1_noskip = ConvBlock(128, 128)

        # ========== Layer 2: 16x16 -> 32x32 ==========
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ag2 = AttentionGate(F_g=128, F_l=32, F_int=16)
        self.conv2_skip = ConvBlock(128 + 32, 64)
        self.conv2_noskip = ConvBlock(128, 64)

        # ========== Layer 3: 32x32 -> 64x64 ==========
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = ConvBlock(64, 64)

        # ========== Prediction Heads ==========
        self.shared_conv = nn.Sequential(Conv2d_BN(64, 64, 3, 1, 1), nn.GELU())

        # Core heads (always present)
        self.skeleton_head = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.tangent_head = nn.Sequential(nn.Conv2d(64, 2, 1), nn.Tanh())

        # Optional heads (for full dense prediction)
        if full_heads:
            # Keypoints: 2ch (Ch0=topo, Ch1=geo)
            self.keypoints_head = nn.Sequential(nn.Conv2d(64, 2, 1), nn.Sigmoid())
            self.width_head = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Softplus())
            self.offset_conv = nn.Conv2d(64, 2, 1)

        # Deep Supervision heads
        self.ds1_head = nn.Sequential(
            Conv2d_BN(128, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 1, 1)
        )
        self.ds2_head = nn.Sequential(
            Conv2d_BN(64, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 1, 1)
        )

    def forward(self, features, use_skips=True):
        """
        Args:
            features:
                if use_skips=True: [f1, f2, f3] from Encoder
                if use_skips=False: f3 only (or [f1, f2, f3] but f1/f2 ignored)
            use_skips: bool, 是否使用跳连
        Returns:
            outputs: dict with 'skeleton', 'tangent', and optionally others
            aux_outputs: dict with deep supervision outputs (training only)
        """
        # Handle input format
        if isinstance(features, (list, tuple)):
            f1, f2, f3 = features
        else:
            # features is just f3
            f3 = features
            f1, f2 = None, None

        # Project F3 to 128 channels if needed
        f3 = self.f3_proj(f3)

        # ========== Block 1 (8 -> 16) ==========
        d1_up = self.up1(f3)  # [B, 128, 16, 16]
        if use_skips and f2 is not None:
            f2_gated = self.ag1(d1_up, f2)
            d1 = torch.cat([d1_up, f2_gated], dim=1)
            d1 = self.conv1_skip(d1)
        else:
            d1 = self.conv1_noskip(d1_up)

        # ========== Block 2 (16 -> 32) ==========
        d2_up = self.up2(d1)  # [B, 128, 32, 32]
        if use_skips and f1 is not None:
            f1_gated = self.ag2(d2_up, f1)
            d2 = torch.cat([d2_up, f1_gated], dim=1)
            d2 = self.conv2_skip(d2)
        else:
            d2 = self.conv2_noskip(d2_up)

        # ========== Block 3 (32 -> 64) ==========
        d3_up = self.up3(d2)
        d3 = self.conv3(d3_up)  # [B, 64, 64, 64]

        # ========== Prediction ==========
        feat = self.shared_conv(d3)

        outputs = {
            "skeleton": self.skeleton_head(feat),
            "tangent": self.tangent_head(feat),
        }

        if self.full_heads:
            outputs["keypoints"] = self.keypoints_head(feat)
            outputs["width"] = self.width_head(feat)
            outputs["offset"] = torch.tanh(self.offset_conv(feat)) * 0.5

        # Deep Supervision
        aux_outputs = {}
        if self.training:
            aux_outputs["aux_skeleton_16"] = torch.sigmoid(self.ds1_head(d1))
            aux_outputs["aux_skeleton_32"] = torch.sigmoid(self.ds2_head(d2))

        return outputs, aux_outputs
