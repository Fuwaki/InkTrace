import torch
import torch.nn as nn
from RepVit import Conv2d_BN
from dense_heads import DenseHeads


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





class NeXtBlock(nn.Module):
    """
    ConvNeXt Style Block for Decoder.
    特点: 7x7 Depthwise Conv + Inverted Bottleneck (1x1 Conv)
    优势: 比普通 3x3 ResBlock 感受野更大，计算量更小，适合捕捉长笔画结构。
    """

    def __init__(self, in_channels, out_channels, expand_ratio=2, kernel_size=7):
        super().__init__()
        # Ensure input fits output dimension if needed for residual connection
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = Conv2d_BN(in_channels, out_channels, 1, 1, 0)

        # 1. Depthwise Conv: Large Kernel (7x7), Spatial mixing
        # We assume input has been projected to 'out_channels' dimension or we handle it inside.
        # Here we follow a design where we match dimensions first if needed,
        # but to keep it clean, let's process 'in_channels' -> 'out_channels' at the start if needed.

        # However, standard ConvNeXt keeps dims constant.
        # Let's do a preliminary projection if in != out, similar to the shortcut.
        self.pre_proj = nn.Identity()
        current_dim = in_channels
        if in_channels != out_channels:
            self.pre_proj = Conv2d_BN(in_channels, out_channels, 1, 1, 0)
            current_dim = out_channels

        # Now standard ConvNeXt block
        self.dwconv = Conv2d_BN(
            current_dim,
            current_dim,
            kernel_size,
            stride=1,
            pad=kernel_size // 2,
            groups=current_dim,
        )

        hidden_dim = int(current_dim * expand_ratio)
        self.pwconv1 = Conv2d_BN(current_dim, hidden_dim, 1, 1, 0)
        self.act = nn.GELU()
        self.pwconv2 = Conv2d_BN(hidden_dim, current_dim, 1, 1, 0)

    def forward(self, x):
        # Shortcut uses original input
        res = self.shortcut(x)

        # Main path
        x = self.pre_proj(x)
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

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

        # [Refactor] Explicit Fusion + Shared Processor
        # Fusion: Merge skip connection (if any) to 128 channels
        self.fusion1_skip = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 1, bias=False), nn.BatchNorm2d(128)
        )
        self.fusion1_noskip = nn.Identity()  # 128 -> 128 identity

        # Shared Block: NeXtBlock (7x7) to process the fused features
        # Crucial: Weights are shared between skip/noskip modes
        self.conv1_shared = NeXtBlock(128, 128, kernel_size=7)

        # ========== Layer 2: 16x16 -> 32x32 ==========
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ag2 = AttentionGate(F_g=128, F_l=32, F_int=16)

        # Fusion: 128+32 -> 64
        self.fusion2_skip = nn.Sequential(
            nn.Conv2d(128 + 32, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        # Fusion: 128 -> 64
        self.fusion2_noskip = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64)
        )

        # Shared Block
        self.conv2_shared = NeXtBlock(64, 64, kernel_size=7)

        # ========== Layer 3: 32x32 -> 64x64 ==========
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = NeXtBlock(64, 64, kernel_size=7)

        # ========== Prediction Heads ==========
        self.heads = DenseHeads(in_channels=64, head_channels=64, full_heads=full_heads)

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
            d1_fused = self.fusion1_skip(torch.cat([d1_up, f2_gated], dim=1))
        else:
            d1_fused = self.fusion1_noskip(d1_up)

        d1 = self.conv1_shared(d1_fused)

        # ========== Block 2 (16 -> 32) ==========
        d2_up = self.up2(d1)  # [B, 128, 32, 32]
        if use_skips and f1 is not None:
            f1_gated = self.ag2(d2_up, f1)
            d2_fused = self.fusion2_skip(torch.cat([d2_up, f1_gated], dim=1))
        else:
            d2_fused = self.fusion2_noskip(d2_up)

        d2 = self.conv2_shared(d2_fused)

        # ========== Block 3 (32 -> 64) ==========
        d3_up = self.up3(d2)
        d3 = self.conv3(d3_up)  # [B, 64, 64, 64]

        # ========== Prediction ==========
        outputs = self.heads(d3)

        # Deep Supervision
        aux_outputs = {}
        if self.training:
            aux_outputs["aux_skeleton_16"] = torch.sigmoid(self.ds1_head(d1))
            aux_outputs["aux_skeleton_32"] = torch.sigmoid(self.ds2_head(d2))

        return outputs, aux_outputs
