"""
DETR 风格的矢量化解码器

核心特性：
1. Set Prediction：预测无序的笔画集合
2. 每个 Slot 预测一个完整的独立笔画（P0, P1, P2, P3, w_start, w_end）
3. 配合 Hungarian Matching Loss 使用
"""

import torch
import torch.nn as nn


class DETRVectorDecoder(nn.Module):
    """
    DETR 风格的矢量化解码器

    架构：
    - Learnable Slot Queries：每个 Slot 负责预测一个笔画
    - Transformer Decoder：Slots attend to Encoder features
    - Prediction Head：输出笔画参数（P0, P1, P2, P3, w_start, w_end）

    输出格式：
    - strokes: [B, num_slots, 10]
      - 前 8 维：P0, P1, P2, P3 的坐标（归一化到 [0, 1]）
      - 后 2 维：w_start, w_end（归一化到 [0, 10]）
    - validity: [B, num_slots, 1] 有效性概率
    """

    def __init__(
        self,
        embed_dim=128,
        num_slots=8,      # 最多预测 8 个笔画
        num_layers=3,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_slots = num_slots

        # 1. Learnable Slot Queries
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, embed_dim) * 0.02)

        # 2. 位置编码（给 Slots 加上位置先验）
        self.register_buffer('position_ids', torch.arange(num_slots).expand((1, -1)))
        self.position_embeddings = nn.Embedding(num_slots, embed_dim)

        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. 笔画参数预测头
        self.stroke_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 10)
        )

        # 5. 有效性预测头
        self.validity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

        # 6. Layer Norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: [B, seq_len, embed_dim]

        Returns:
            strokes: [B, num_slots, 10] 笔画参数
            validity: [B, num_slots, 1] 有效性
        """
        B = encoder_features.shape[0]
        device = encoder_features.device

        # 1. 初始化 Slot Queries + 位置编码
        slots = self.slot_queries.expand(B, -1, -1)  # [B, num_slots, embed_dim]
        pos_embeds = self.position_embeddings(self.position_ids).expand(B, -1, -1)
        slots = slots + pos_embeds

        # 2. Transformer Decoder
        decoded_slots = self.decoder(
            tgt=slots,               # [B, num_slots, embed_dim]
            memory=encoder_features  # [B, 64, embed_dim]
        )  # [B, num_slots, embed_dim]

        # 3. Layer Norm
        decoded_slots = self.norm(decoded_slots)

        # 4. 预测笔画参数
        strokes = self.stroke_head(decoded_slots)  # [B, num_slots, 10]

        # 归一化坐标到 [0, 1]
        strokes[..., :8] = torch.sigmoid(strokes[..., :8])
        # 归一化宽度到 [0, 10]
        strokes[..., 8:10] = torch.sigmoid(strokes[..., 8:10]) * 10.0

        # 5. 预测有效性
        validity_logits = self.validity_head(decoded_slots)
        validity = torch.sigmoid(validity_logits)  # [B, num_slots, 1]

        return strokes, validity
