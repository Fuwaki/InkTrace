import torch
import torch.nn as nn
import math
from RepVit import RepViTBlock, Conv2d_BN, _make_divisible
from modules import AddCoords


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
