"""
DETR 风格的矢量化解码器 (V2 改进版)

核心特性：
1. Set Prediction：预测无序的笔画集合
2. 每个 Slot 预测一个完整的独立笔画（P0, P1, P2, P3, w_start, w_end）
3. 配合 Hungarian Matching Loss 使用

改进点：
- Slot 间交互：Self-Attention 让 Slots 互相感知，避免预测重复
- 空间先验：基于 2D 位置的 Slot 初始化，让不同 Slot 负责不同区域
- 扩展输出范围：支持画面外的控制点
- 迭代优化：多轮 Decoder 逐步精化预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DETRVectorDecoder(nn.Module):
    """
    DETR 风格的矢量化解码器 (V2 改进版)

    架构改进：
    - 2D 空间位置先验：Slot 按网格分布，天然负责不同区域
    - Slot Self-Attention：Slots 之间互相感知，减少重复预测
    - 迭代式解码：多轮优化，逐步精化

    输出格式：
    - strokes: [B, num_slots, 10]
      - 前 8 维：P0, P1, P2, P3 的坐标（范围 [-0.5, 1.5]）
      - 后 2 维：w_start, w_end（范围 [0, 10]）
    - validity: [B, num_slots, 1] 有效性概率
    """

    def __init__(
        self,
        embed_dim=128,
        num_slots=8,  # 最多预测 8 个笔画
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        num_refine_steps=2,  # 迭代优化次数
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.num_refine_steps = num_refine_steps

        # 1. Learnable Slot Queries (可学习的内容查询)
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, embed_dim) * 0.02)

        # 2. 2D 空间位置先验 (让 Slot 天然负责不同区域)
        # 将 8 个 Slots 按 4x2 网格分布
        self._init_spatial_prior()
        self.spatial_embed = nn.Linear(2, embed_dim)

        # 3. Slot Self-Attention (让 Slots 互相感知)
        self.slot_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.slot_self_attn_norm = nn.LayerNorm(embed_dim)

        # 4. Cross-Attention Decoder (Slots attend to Encoder features)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 5. 笔画参数预测头 (更深的网络)
        self.stroke_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 10),
        )

        # 6. 有效性预测头 (独立的深度网络)
        self.validity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )

        # 7. 迭代优化：将上一轮预测编码回 Slot
        self.refine_encoder = nn.Linear(10, embed_dim)

        # 8. Layer Norm
        self.norm = nn.LayerNorm(embed_dim)

    def _init_spatial_prior(self):
        """初始化 2D 空间位置先验"""
        # 将 num_slots 个 Slot 按网格分布
        # 例如 8 个 Slot -> 4x2 网格
        grid_h = int(math.ceil(math.sqrt(self.num_slots)))
        grid_w = int(math.ceil(self.num_slots / grid_h))

        positions = []
        for i in range(self.num_slots):
            row = i // grid_w
            col = i % grid_w
            # 归一化到 [0, 1]
            y = (row + 0.5) / grid_h
            x = (col + 0.5) / grid_w
            positions.append([x, y])

        self.register_buffer(
            "spatial_positions", torch.tensor(positions, dtype=torch.float32)
        )  # [num_slots, 2]

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

        # 1. 初始化 Slot Queries + 空间位置先验
        slots = self.slot_queries.expand(B, -1, -1)  # [B, num_slots, embed_dim]
        spatial_embeds = self.spatial_embed(
            self.spatial_positions
        )  # [num_slots, embed_dim]
        slots = slots + spatial_embeds.unsqueeze(0)  # [B, num_slots, embed_dim]

        # 2. Slot Self-Attention (让 Slots 互相感知)
        slots_residual = slots
        slots, _ = self.slot_self_attn(slots, slots, slots)
        slots = self.slot_self_attn_norm(slots + slots_residual)

        # 3. 迭代式解码
        for step in range(self.num_refine_steps):
            # Cross-Attention: Slots attend to Encoder features
            decoded_slots = self.decoder(
                tgt=slots,  # [B, num_slots, embed_dim]
                memory=encoder_features,  # [B, 64, embed_dim]
            )  # [B, num_slots, embed_dim]

            # Layer Norm
            decoded_slots = self.norm(decoded_slots)

            # 如果不是最后一轮，将预测结果编码回 Slot 进行优化
            if step < self.num_refine_steps - 1:
                # 预测当前轮的笔画参数
                strokes_raw = self.stroke_head(decoded_slots)
                # 将预测编码回 Slot
                slots = slots + self.refine_encoder(strokes_raw)
                # 再次 Self-Attention
                slots_residual = slots
                slots, _ = self.slot_self_attn(slots, slots, slots)
                slots = self.slot_self_attn_norm(slots + slots_residual)

        # 4. 最终预测
        strokes = self.stroke_head(decoded_slots)  # [B, num_slots, 10]

        # 归一化坐标到 [-0.5, 1.5] (Sigmoid * 2.0 - 0.5) 以覆盖画面外控制点
        strokes = strokes.clone()
        strokes[..., :8] = torch.sigmoid(strokes[..., :8]) * 2.0 - 0.5
        # 归一化宽度到 [0, 10]
        strokes[..., 8:10] = torch.sigmoid(strokes[..., 8:10]) * 10.0

        # 5. 预测有效性
        validity_logits = self.validity_head(decoded_slots)
        validity = torch.sigmoid(validity_logits)  # [B, num_slots, 1]

        return strokes, validity
