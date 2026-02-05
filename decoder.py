import torch
import torch.nn as nn
import torch.nn.functional as F
from RepVit import Conv2d_BN
from dense_heads import DenseHeads


# =============================================================================
# Building Blocks
# =============================================================================


class GroundingBlock(nn.Module):
    """
    Grounding Transformer Block (Inspired by FPT, ECCV 2020)

    Uses high-level semantics (Query from Decoder) to search and refine
    low-level details (Key/Value from Encoder skip connection).

    与 AttentionGate 的区别:
    - AG: 只是"过滤器"，抑制无关区域
    - Grounding: "搜索引擎"，主动在 Encoder 特征中查找相关细节

    对于墨迹重建任务:
    - Decoder 发出查询: "我需要恢复这里的横折钩细节"
    - Encoder 提供资源: 边缘纹理、转角特征等
    - Attention 计算相关度，精准抓取匹配的细节
    """

    def __init__(self, dim_high, dim_low, num_heads=4, ffn_expand=2):
        """
        Args:
            dim_high: Decoder feature channels (Query source)
            dim_low:  Encoder skip feature channels (Key/Value source, also output dim)
            num_heads: Number of attention heads
            ffn_expand: FFN expansion ratio
        """
        super().__init__()
        self.dim_low = dim_low
        self.num_heads = num_heads
        self.head_dim = dim_low // num_heads
        self.scale = self.head_dim**-0.5

        assert dim_low % num_heads == 0, f"dim_low ({dim_low}) must be divisible by num_heads ({num_heads})"

        # Layer Norms (Transformer style, stabilizes training)
        self.norm_high = nn.GroupNorm(1, dim_high)  # Instance Norm equivalent
        self.norm_low = nn.GroupNorm(1, dim_low)

        # Linear projections
        # Q comes from Decoder (High level) -> project to dim_low
        self.to_q = nn.Conv2d(dim_high, dim_low, 1, bias=False)
        # K, V come from Encoder (Low level)
        self.to_k = nn.Conv2d(dim_low, dim_low, 1, bias=False)
        self.to_v = nn.Conv2d(dim_low, dim_low, 1, bias=False)

        # Output projection
        self.proj = nn.Conv2d(dim_low, dim_low, 1, bias=False)

        # Lightweight FFN for enhanced expressiveness
        self.ffn = nn.Sequential(
            nn.Conv2d(dim_low, dim_low * ffn_expand, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim_low * ffn_expand, dim_low, 1, bias=False),
        )
        self.norm_ffn = nn.GroupNorm(1, dim_low)

        # Gating parameter (Init as 0 to start with pure skip, gradual warmup)
        self.gamma_attn = nn.Parameter(torch.zeros(1))
        self.gamma_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, high_feature, low_feature):
        """
        Args:
            high_feature: [B, C_high, H, W] (Decoder / Query)
            low_feature:  [B, C_low,  H, W] (Encoder / Key, Value)
        Returns:
            Refined low_feature with semantic alignment
        """
        B, _, H, W = low_feature.shape
        N = H * W

        # Normalize inputs
        high_norm = self.norm_high(high_feature)
        low_norm = self.norm_low(low_feature)

        # 1. Projections -> [B, heads, N, head_dim]
        q = self.to_q(high_norm).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = self.to_k(low_norm).view(B, self.num_heads, self.head_dim, N)  # [B, heads, head_dim, N]
        v = self.to_v(low_norm).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # 2. Cross Attention: Q @ K^T
        # Query (high-level) searches in Key (low-level details)
        attn = (q @ k) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)

        # 3. Aggregation: Attn @ V
        out = (attn @ v)  # [B, heads, N, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, self.dim_low, H, W)

        # 4. Output projection
        out = self.proj(out)

        # 5. Residual with learnable gate (attention path)
        x = low_feature + self.gamma_attn * out

        # 6. FFN with residual (enhances local processing after global attention)
        x = x + self.gamma_ffn * self.ffn(self.norm_ffn(x))

        return x

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

    Skip Connection 策略:
    - 使用 GroundingBlock (Cross-Attention) 替代简单的 AttentionGate
    - Decoder 特征作为 Query，在 Encoder 特征中"搜索"相关细节
    - 实现语义对齐，而非简单的门控过滤

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
        # Grounding Block: Cross-Attention for semantic alignment
        # dim_high=128 (Decoder), dim_low=128 (Encoder F2)
        self.grounding1 = GroundingBlock(dim_high=128, dim_low=128, num_heads=4)

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
        # Grounding Block: dim_high=128 (Decoder), dim_low=32 (Encoder F1)
        self.grounding2 = GroundingBlock(dim_high=128, dim_low=32, num_heads=4)

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
            # Grounding: Decoder queries Encoder for relevant details
            f2_grounded = self.grounding1(d1_up, f2)
            d1_fused = self.fusion1_skip(torch.cat([d1_up, f2_grounded], dim=1))
        else:
            d1_fused = self.fusion1_noskip(d1_up)

        d1 = self.conv1_shared(d1_fused)

        # ========== Block 2 (16 -> 32) ==========
        d2_up = self.up2(d1)  # [B, 128, 32, 32]
        if use_skips and f1 is not None:
            # Grounding: Decoder queries Encoder for fine details
            f1_grounded = self.grounding2(d2_up, f1)
            d2_fused = self.fusion2_skip(torch.cat([d2_up, f1_grounded], dim=1))
        else:
            d2_fused = self.fusion2_noskip(d2_up)

        d2 = self.conv2_shared(d2_fused)

        # ========== Block 3 (32 -> 64) ==========
        d3_up = self.up3(d2)
        d3 = self.conv3(d3_up)  # [B, 64, 64, 64]

        # ========== Prediction ==========
        outputs = self.heads(d3)

        return outputs
