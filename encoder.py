import torch
import torch.nn as nn
import math
from timm.models.repvit import (
    ConvNorm, RepViTBlock, RepVitDownsample
)
from timm.layers import SqueezeExcite
from modules import AddCoords


class StrokeEncoder(nn.Module):
    """编码器：使用修改后的 RepViT 提取特征，Transformer 处理序列"""

    def __init__(
        self,
        in_channels=1,  # 单通道灰度图
        embed_dim=64,  # embedding 维度 (UNet 瓶颈: stem 128 -> bottleneck 64)
        num_heads=4,  # Transformer 注意力头数 (64/4=16 per head)
        num_layers=2,  # Transformer 层数 (64 tokens 不需要太深)
        dropout=0.1,
    ):
        super().__init__()

        # --- 修改点 1: 引入坐标生成器 ---
        self.add_coords = AddCoords(height=64, width=64)  # 预计算 64x64 坐标

        # --- 修改点 2: Stem 接收动态通道数 (1 Gray + added_coords) ---
        # 这样网络第一层就能感知绝对几何位置
        input_channels_with_coords = in_channels + self.add_coords.added_channels

        # ========== Minimal Stem (使用 timm ConvNorm) ==========
        # 仅做通道投影：3ch → 16ch，不降采样
        self.stem = nn.Sequential(
            ConvNorm(input_channels_with_coords, 16, 3, 1, 1),  # 3→16, 保持 64×64
            nn.GELU(),
        )

        # ========== Stage 1: 16→32, 64×64→32×32 ==========
        self.stage1_downsample = RepVitDownsample(
            16, 2, 32, 3, nn.GELU
        )
        self.stage1_blocks = nn.ModuleList([
            RepViTBlock(32, 2, 3, use_se=False, act_layer=nn.GELU),
        ])

        # ========== Stage 2: 32→64, 32×32→16×16 ==========
        self.stage2_downsample = RepVitDownsample(
            32, 2, 64, 3, nn.GELU
        )
        self.stage2_blocks = nn.ModuleList([
            RepViTBlock(64, 2, 3, use_se=True, act_layer=nn.GELU),
            RepViTBlock(64, 2, 3, use_se=False, act_layer=nn.GELU),
        ])

        # ========== Stage 3: 64→128, 16×16→8×8 ==========
        self.stage3_downsample = RepVitDownsample(
            64, 2, 128, 3, nn.GELU
        )
        self.stage3_blocks = nn.ModuleList([
            RepViTBlock(128, 2, 3, use_se=True, act_layer=nn.GELU),
            RepViTBlock(128, 2, 3, use_se=False, act_layer=nn.GELU),
        ])

        # 最终通道数和空间尺寸
        self.feature_dim = 128  # Stage 3 输出
        self.spatial_size = 8   # 8×8 特征图 = 64 tokens

        # Skip 通道数（供 Decoder 读取）
        self.f1_channels = 32   # Stage 1 输出
        self.f2_channels = 64   # Stage 2 输出

        # 4. 特征图 -> Token 序列
        # 将 128x8x8 展平为 64x128 的 token 序列
        self.token_embed = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.Dropout(0.1),
        )

        # 5. 可学习位置编码 (使用 xavier_uniform 更稳定)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))
        nn.init.xavier_uniform_(self.pos_embed)

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

        # 8. Bottleneck Dropout (强制模型学习高级特征)
        self.bottleneck_dropout = nn.Dropout(0.15)

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
                  - f2: [B, 64, 16, 16]
                  - f3_enhanced: [B, embed_dim, 8, 8] (embed_dim 可配置, 例如 64)
                embeddings: [B, 64, embed_dim] (same as default return)
        """
        B = x.shape[0]

        # --- 1. 注入坐标信息 ---
        # x 变成了 [B, 3, 64, 64]
        x = self.add_coords(x)

        # --- 2. Minimal Stem: 通道投影 ---
        x = self.stem(x)  # [B, 16, 64, 64]

        # --- 3. Stage 1: 16→32, 64→32 (F1) ---
        x = self.stage1_downsample(x)
        for block in self.stage1_blocks:
            x = block(x)
        f1 = x  # [B, 32, 32, 32]

        # --- 4. Stage 2: 32→64, 32→16 (F2) ---
        x = self.stage2_downsample(x)
        for block in self.stage2_blocks:
            x = block(x)
        f2 = x  # [B, 64, 16, 16]

        # --- 5. Stage 3: 64→128, 16→8 ---
        x = self.stage3_downsample(x)
        for block in self.stage3_blocks:
            x = block(x)
        # x: [B, 128, 8, 8]

        # --- 6. 展平为 token 序列 ---
        x = x.flatten(2)  # [B, 128, 64]
        x = x.transpose(1, 2)  # [B, 64, 128]

        # --- 7. Embedding 投影 ---
        x = self.token_embed(x)  # [B, 64, embed_dim]

        # --- 8. 可学习位置编码 ---
        x = x + self.pos_embed

        # --- 9. Transformer 处理 ---
        x_trans = self.transformer(x)  # [B, 64, embed_dim]

        # --- 10. Layer Norm ---
        embeddings = self.norm(x_trans)  # [B, 64, embed_dim]

        # --- 11. Bottleneck Dropout ---
        embeddings = self.bottleneck_dropout(embeddings)

        if return_interm_layers:
            # Reshape embeddings back to spatial [B, C, H, W] for F3
            # embeddings: [B, 64, embed_dim] -> [B, embed_dim, 8, 8]
            H = W = self.spatial_size  # 8
            f3_enhanced = embeddings.transpose(1, 2).reshape(
                B, -1, H, W
            )  # [B, embed_dim, 8, 8]
            return [f1, f2, f3_enhanced], embeddings

        return embeddings
