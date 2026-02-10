import torch
import torch.nn as nn
import math
from RepVit import RepViTBlock, Conv2d_BN, _make_divisible


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


class StrokeEncoder(nn.Module):
    """编码器：使用修改后的 RepViT 提取特征，Transformer 处理序列"""

    def __init__(
        self,
        in_channels=1,  # 单通道灰度图
        embed_dim=192,  # embedding 维度 (与 configs/default.yaml 一致)
        num_heads=6,  # Transformer 注意力头数
        num_layers=4,  # Transformer 层数 (与 configs/default.yaml 一致)
        dropout=0.1,
    ):
        super().__init__()

        # --- 修改点 1: 引入坐标生成器 ---
        self.add_coords = AddCoords(height=64, width=64)  # 预计算 64x64 坐标

        # --- 修改点 2: Stem 接收动态通道数 (1 Gray + added_coords) ---
        # 这样网络第一层就能感知绝对几何位置
        input_channels_with_coords = in_channels + self.add_coords.added_channels

        self.stem1 = nn.Sequential(
            Conv2d_BN(input_channels_with_coords, 32, 3, 2, 1),  # 3 -> 32
            nn.GELU(),
        )
        self.stem2 = nn.Sequential(
            Conv2d_BN(32, 64, 3, 2, 1),  # 32->64, 32x32 -> 16x16
            nn.GELU(),
            Conv2d_BN(64, 128, 3, 1, 1),  # 64->128, 16x16 (保持分辨率)
        )

        # 2. RepViT 特征提取块 (优化版)
        # cfg: [k, t, c, SE, HS, s]
        # 重要：stride=1 时要求 inp == oup
        # 策略：只在必要时用 stride=2 改变通道，其余用 stride=1 提取特征
        # 最终保持在 8x8 = 64 tokens
        cfgs = [
            [3, 2, 128, 0, 0, 2],  # 128->128, stride=2, 16x16->8x8
            [3, 2, 128, 1, 0, 1],  # 128->128, stride=1, 8x8
            [3, 2, 128, 0, 1, 1],  # 128->128, stride=1, 8x8
            [3, 2, 128, 1, 1, 1],  # 128->128, stride=1, 8x8
            [3, 2, 128, 0, 1, 1],  # 128->128, stride=1, 8x8
            [3, 2, 128, 1, 1, 1],  # 128->128, stride=1, 8x8
        ]

        self.features = nn.ModuleList()
        input_channel = 128  # stem 已经输出 128 通道

        for k, t, c, use_se, use_hs, s in cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            block = RepViTBlock(
                input_channel, exp_size, output_channel, k, s, use_se, use_hs
            )
            self.features.append(block)
            input_channel = output_channel

        # 最终通道数和空间尺寸
        self.feature_dim = input_channel  # 128
        self.spatial_size = 8  # 8x8 特征图 = 64 tokens

        # 4. 特征图 -> Token 序列
        # 将 128x8x8 展平为 64x128 的 token 序列
        self.token_embed = nn.Linear(self.feature_dim, embed_dim)

        # 5. 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 6. Transformer Encoder (增加层数以充分利用 token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 7. Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, return_interm_layers=False):
        """
        Args:
            x: [B, 1, 64, 64] 输入图像
            return_interm_layers: bool, 是否返回中间特征层 (F1, F2, F3)
        Returns:
            if return_interm_layers=False:
                embeddings: [B, 64, embed_dim] embedding 序列 (F3_Enhanced flattened)
            if return_interm_layers=True:
                (features, embeddings)
                features: [f1, f2, f3_enhanced] (spatially organized)
                  - f1: [B, 32, 32, 32]
                  - f2: [B, 128, 16, 16]
                  - f3_enhanced: [B, 128, 8, 8]
                embeddings: [B, 64, embed_dim] (same as default return)
        """
        B = x.shape[0]

        # --- 1. 注入坐标信息 ---
        # x 变成了 [B, 3, 64, 64]
        x = self.add_coords(x)

        # 2. Stem 层 (Split for F1, F2)
        x = self.stem1(x)
        f1 = x  # [B, 32, 32, 32]

        x = self.stem2(x)
        f2 = x  # [B, 128, 16, 16]

        # 2. RepViT 特征提取
        for feature in self.features:
            x = feature(x)  # [B, 128, 8, 8]

        # 3. 展平为 token 序列
        x = x.flatten(2)  # [B, 128, 64]
        x = x.transpose(1, 2)  # [B, 64, 128]

        # 4. Embedding 投影
        x = self.token_embed(x)  # [B, 64, embed_dim]

        # 5. 可学习位置编码
        x = x + self.pos_embed

        # 6. Transformer 处理
        x_trans = self.transformer(x)  # [B, 64, embed_dim]

        # 7. Layer Norm
        embeddings = self.norm(x_trans)  # [B, 64, embed_dim]

        if return_interm_layers:
            # Reshape embeddings back to spatial [B, C, H, W] for F3
            # embeddings: [B, 64, embed_dim] -> [B, embed_dim, 8, 8]
            # But decoder expects 128 channels, so we need a projection
            H = W = self.spatial_size  # 8
            # Project back to feature_dim (128) if embed_dim != 128
            f3_enhanced = embeddings.transpose(1, 2).reshape(
                B, -1, H, W
            )  # [B, embed_dim, 8, 8]
            return [f1, f2, f3_enhanced], embeddings

        return embeddings
