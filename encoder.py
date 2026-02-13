import torch
import torch.nn as nn
import math
from timm.models.repvit import ConvNorm, RepViTBlock, RepVitDownsample
from modules import AddCoords, SpatialModulation


class StrokeEncoder(nn.Module):
    """
    坐标调制沙漏编码器（最终版）

    信息流: 32768 -> 16384 -> 8192 -> 4096 -> 2048 -> 1024 (瓶颈)
    返回：embeddings [B, num_tokens, embed_dim] 或者 ([f1,f2,f3,f4], embeddings)
    """

    def __init__(
        self,
        in_channels=1,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        # ========== 输入坐标：仅线性 xy (2通道) ==========
        self.add_coords = AddCoords(num_freqs=0, height=64, width=64)
        input_ch = in_channels + 2  # 1 + 2 = 3

        # ========== Stem: 3→8, 保持 64×64 ==========
        self.stem = nn.Sequential(
            ConvNorm(input_ch, 8, 3, 1, 1),
            nn.GELU(),
        )

        # ========== Stage 1: 8→16, 64→32 ==========
        self.stage1_downsample = RepVitDownsample(8, 2, 16, 3, nn.GELU)
        self.stage1_blocks = nn.ModuleList(
            [
                RepViTBlock(16, 2, 3, use_se=False, act_layer=nn.GELU),
            ]
        )
        self.stage1_mod = SpatialModulation(16, 32, 32, num_freqs=2)
        self.f1_channels = 16

        # ========== Stage 2: 16→32, 32→16 ==========
        self.stage2_downsample = RepVitDownsample(16, 2, 32, 3, nn.GELU)
        self.stage2_blocks = nn.ModuleList(
            [
                RepViTBlock(32, 2, 3, use_se=True, act_layer=nn.GELU),
            ]
        )
        self.stage2_mod = SpatialModulation(32, 16, 16, num_freqs=1)
        self.f2_channels = 32

        # ========== Stage 3: 32→64, 16→8 ==========
        self.stage3_downsample = RepVitDownsample(32, 2, 64, 3, nn.GELU)
        self.stage3_blocks = nn.ModuleList(
            [
                RepViTBlock(64, 2, 3, use_se=True, act_layer=nn.GELU),
                RepViTBlock(64, 2, 3, use_se=False, act_layer=nn.GELU),
            ]
        )
        self.stage3_mod = SpatialModulation(64, 8, 8, num_freqs=1)
        self.f3_channels = 64

        # ========== Stage 4: 64→128, 8→4 (瓶颈前) ==========
        self.stage4_downsample = RepVitDownsample(64, 2, 128, 3, nn.GELU)
        self.stage4_blocks = nn.ModuleList(
            [
                RepViTBlock(128, 2, 3, use_se=True, act_layer=nn.GELU),
            ]
        )
        self.stage4_mod = SpatialModulation(128, 4, 4, num_freqs=0)

        # ========== Transformer 瓶颈 ==========
        self.feature_dim = 128
        self.spatial_size = 4
        self.num_tokens = 16  # 4×4

        self.token_embed = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.bottleneck_dropout = nn.Dropout(0.15)

    def forward(self, x, return_interm_layers=False):
        """
        Args:
            x: [B, 1, 64, 64]
            return_interm_layers: if True, return ([f1,f2,f3,f4], embeddings)
        Returns:
            embeddings: [B, 16, embed_dim]
        """
        B = x.shape[0]

        # 输入 + 线性坐标
        x = self.add_coords(x)  # [B, 3, 64, 64]
        x = self.stem(x)  # [B, 8, 64, 64]

        # Stage 1
        x = self.stage1_downsample(x)
        for block in self.stage1_blocks:
            x = block(x)
        x = self.stage1_mod(x)
        f1 = x  # [B, 16, 32, 32]

        # Stage 2
        x = self.stage2_downsample(x)
        for block in self.stage2_blocks:
            x = block(x)
        x = self.stage2_mod(x)
        f2 = x  # [B, 32, 16, 16]

        # Stage 3
        x = self.stage3_downsample(x)
        for block in self.stage3_blocks:
            x = block(x)
        x = self.stage3_mod(x)
        f3 = x  # [B, 64, 8, 8]

        # Stage 4
        x = self.stage4_downsample(x)
        for block in self.stage4_blocks:
            x = block(x)
        x = self.stage4_mod(x)  # [B, 128, 4, 4]

        # Flatten → Transformer
        x = x.flatten(2).transpose(1, 2)  # [B, 16, 128]
        x = self.token_embed(x)  # [B, 16, embed_dim]
        x = x + self.pos_embed
        x = self.transformer(x)  # [B, 16, embed_dim]
        embeddings = self.norm(x)
        embeddings = self.bottleneck_dropout(embeddings)

        if return_interm_layers:
            H = W = self.spatial_size  # 4
            f4 = embeddings.transpose(1, 2).reshape(B, -1, H, W)
            return [f1, f2, f3, f4], embeddings

        return embeddings
