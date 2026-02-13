from torch import nn
import torch
import math
from timm.models.repvit import ConvNorm
class AddCoords(nn.Module):
    """
    自动添加坐标通道，支持高频（Fourier）坐标注入
    输入: [B, C, H, W]
    输出: [B, C + added_channels, H, W]
    """

    def __init__(self, num_freqs=2, height=64, width=64):
        super().__init__()
        self.num_freqs = num_freqs
        self.height = height
        self.width = width
        self.added_channels = 2 + (self.num_freqs * 4)

        # 执行预计算
        self._precompute_coords()

    def _precompute_coords(self):
        # 1. 使用 meshgrid 生成完整的 HxW 网格 (更直观，且自动处理广播)
        # indexing='ij' 保证 y 在前 (H), x 在后 (W)
        yy = torch.linspace(-1, 1, self.height)
        xx = torch.linspace(-1, 1, self.width)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        # grid_y, grid_x 形状均为 [H, W]
        # 扩展维度以便拼接: [H, W] -> [1, H, W]
        grid_x = grid_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(0)

        # 2. 收集所有坐标特征
        coords_list = [grid_x, grid_y]  # 基础线性坐标

        # 3. 生成 Fourier 特征 (所有特征都必须是 [1, H, W] 形状)
        for i in range(self.num_freqs):
            freq = (2.0**i) * math.pi
            coords_list.extend(
                [
                    torch.sin(freq * grid_x),
                    torch.cos(freq * grid_x),
                    torch.sin(freq * grid_y),
                    torch.cos(freq * grid_y),
                ]
            )

        # 4. 拼接所有特征通道
        # 结果形状: [added_channels, H, W]
        full_coords = torch.cat(coords_list, dim=0)

        # 5. 增加 Batch 维度并注册为 Buffer
        # 最终形状: [1, added_channels, H, W]
        self.register_buffer("cached_coords", full_coords.unsqueeze(0))

    def forward(self, x):
        B, _, H, W = x.shape

        # 简单的尺寸检查
        if H != self.height or W != self.width:
            # 如果尺寸变了（动态输入），这里需要重新计算或者报错
            # 为保证鲁棒性，建议报错，或者在这里动态生成（会慢一点）
            raise ValueError(
                f"Input size ({H}, {W}) doesn't match precomputed size ({self.height}, {self.width})"
            )

        # 直接 Expand 并拼接，效率最高
        # cached_coords: [1, C_add, H, W] -> [B, C_add, H, W]
        coords = self.cached_coords.expand(B, -1, -1, -1)

        return torch.cat([x, coords], dim=1)
class SpatialModulation(nn.Module):
    """
    空间坐标调制：用 Fourier 坐标特征对卷积特征做逐通道仿射变换

    y(h,w,c) = x(h,w,c) · (1 + γ(h,w,c)) + β(h,w,c)

    关键特性：
    1. 不增加主干通道数（γ,β 与输入同维）
    2. 乘法注入，不会被 BN/LN 洗掉
    3. 初始化为恒等映射（1+0=1），训练稳定
    4. 可自由使用高频 Fourier 特征，不担心信号淹没
    """

    def __init__(self, channels, height, width, num_freqs=2, reduction=4):
        super().__init__()
        coord_dim = 2 + num_freqs * 4  # 线性xy + Fourier sin/cos
        hidden = max(channels // reduction, 8)

        # 坐标 → 调制参数的映射网络 (1×1 conv = per-pixel MLP)
        self.modulation_net = nn.Sequential(
            nn.Conv2d(coord_dim, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels * 2, 1, bias=True),  # ×2: γ 和 β
        )

        # 关键：最后一层初始化为 0 → 初始时 γ=0, β=0 → y = x·(1+0)+0 = x
        # 这保证模块初始时是恒等映射，训练从"无调制"开始，逐渐学习
        nn.init.zeros_(self.modulation_net[-1].weight)
        nn.init.zeros_(self.modulation_net[-1].bias)

        # 预计算坐标特征（固定，不随输入变化）
        self._build_coords(height, width, num_freqs)

    def _build_coords(self, H, W, num_freqs):
        yy = torch.linspace(-1, 1, H)
        xx = torch.linspace(-1, 1, W)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")

        feats = [gx.unsqueeze(0), gy.unsqueeze(0)]
        for i in range(num_freqs):
            freq = (2.0**i) * math.pi
            feats.extend(
                [
                    torch.sin(freq * gx).unsqueeze(0),
                    torch.cos(freq * gx).unsqueeze(0),
                    torch.sin(freq * gy).unsqueeze(0),
                    torch.cos(freq * gy).unsqueeze(0),
                ]
            )

        # [1, coord_dim, H, W] — 注册为 buffer，不参与梯度但跟随设备
        self.register_buffer("coord_feats", torch.cat(feats).unsqueeze(0))

    def forward(self, x):
        B = x.shape[0]

        # 坐标特征扩展到 batch 维度
        coords = self.coord_feats.expand(B, -1, -1, -1)

        # 生成调制参数
        params = self.modulation_net(coords)  # [B, 2C, H, W]
        gamma, beta = params.chunk(2, dim=1)  # 各 [B, C, H, W]

        # 残差式调制（初始为恒等）
        return x * (1.0 + gamma) + beta

class GatedCrossAttention(nn.Module):
    """
    带门控的残差交叉注意力模块 (Gated Cross Attention)

    结构:
        x = x + gate * MultiHeadAttention(Q=x, K=skip, V=skip)

    特点:
    1. 纯粹的 Cross Attention，逻辑简单，易于 debug
    2. Zero-init Gate: 初始状态下完全等价于没有 skip connection
    3. 广泛用于 Transformer Decoder 和 Diffusion Model 中
    """

    def __init__(self, dim_high, dim_skip, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_high // num_heads
        self.scale = self.dim_head**-0.5

        # 维度对齐：将 skip 特征投影到 decoder 维度
        self.skip_proj = nn.Conv2d(dim_skip, dim_high, 1, bias=False)
        self.norm_skip = nn.GroupNorm(1, dim_high)

        # Attention 投影
        self.to_q = nn.Conv2d(dim_high, dim_high, 1, bias=False)
        self.to_k = nn.Conv2d(dim_high, dim_high, 1, bias=False)
        self.to_v = nn.Conv2d(dim_high, dim_high, 1, bias=False)
        self.proj = nn.Conv2d(dim_high, dim_high, 1, bias=False)

        # Norm for Query
        self.norm_high = nn.GroupNorm(1, dim_high)

        # 核心：零初始化门控
        # 初始为 0，保证起步时完全切断 skip 信号
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, high_feat, skip_feat=None):
        """
        Args:
            high_feat: Decoder feature [B, C, H, W]
            skip_feat: Encoder feature [B, C_skip, H, W], 可为 None
        Returns:
            Gated residual output [B, C, H, W]
        """
        # 无 skip 时直接返回主干
        if skip_feat is None:
            return high_feat

        B, C, H, W = high_feat.shape
        residual = high_feat

        # 准备 Q (来自 Decoder)
        high_norm = self.norm_high(high_feat)
        q = (
            self.to_q(high_norm)
            .view(B, self.num_heads, self.dim_head, -1)
            .permute(0, 1, 3, 2)
        )

        # 准备 K, V (来自 Encoder Skip)
        skip_proj = self.skip_proj(skip_feat)
        skip_proj = self.norm_skip(skip_proj)

        k = self.to_k(skip_proj).view(B, self.num_heads, self.dim_head, -1)
        v = (
            self.to_v(skip_proj)
            .view(B, self.num_heads, self.dim_head, -1)
            .permute(0, 1, 3, 2)
        )

        # Attention 计算
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # Output Projection
        out = self.proj(out)

        # Gated Residual: residual + gate * out
        return residual + self.gate * out


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
            self.shortcut = ConvNorm(in_channels, out_channels, 1, 1, 0)

        # 1. Depthwise Conv: Large Kernel (7x7), Spatial mixing
        # We assume input has been projected to 'out_channels' dimension or we handle it inside.
        # Here we follow a design where we match dimensions first if needed,
        # but to keep it clean, let's process 'in_channels' -> 'out_channels' at the start if needed.

        # However, standard ConvNeXt keeps dims constant.
        # Let's do a preliminary projection if in != out, similar to the shortcut.
        self.pre_proj = nn.Identity()
        current_dim = in_channels
        if in_channels != out_channels:
            self.pre_proj = ConvNorm(in_channels, out_channels, 1, 1, 0)
            current_dim = out_channels

        # Now standard ConvNeXt block
        self.dwconv = ConvNorm(
            current_dim,
            current_dim,
            kernel_size,
            stride=1,
            pad=kernel_size // 2,
            groups=current_dim,
        )

        hidden_dim = int(current_dim * expand_ratio)
        self.pwconv1 = ConvNorm(current_dim, hidden_dim, 1, 1, 0)
        self.act = nn.GELU()
        self.pwconv2 = ConvNorm(hidden_dim, current_dim, 1, 1, 0)

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


class LargeKernelBlock(nn.Module):
    """单层大核卷积，类似 ConvNeXt 风格的轻量大感受野块"""

    def __init__(self, dim, kernel_size=11):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x):
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv(out)
        return x + out
