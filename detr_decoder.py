"""
DETR 风格的矢量化解码器 (V3 训练优化版)

核心特性：
1. Set Prediction：预测无序的笔画集合
2. 每个 Slot 预测一个完整的独立笔画（P0, P1, P2, P3, w_start, w_end）
3. 配合 Hungarian Matching Loss 使用

V3 改进点 (针对训练稳定性):
- 辅助输出：返回每个 refine step 的中间预测，用于 Deep Supervision
- 去噪训练支持：可接收带噪声的 GT Query，加速收敛 (DN-DETR 思想)
- 更深的预测头：增强表达能力
- 参考点机制：Slot 初始化包含可学习的参考点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DETRVectorDecoder(nn.Module):
    """
    DETR 风格的矢量化解码器 (V3 训练优化版)

    架构改进：
    - 2D 空间位置先验：Slot 按网格分布，天然负责不同区域
    - Slot Self-Attention：Slots 之间互相感知，减少重复预测
    - 迭代式解码：多轮优化，逐步精化
    - 辅助输出：每层都输出预测，用于 Deep Supervision
    - 去噪训练：可选的 DN-DETR 风格加速收敛

    输出格式：
    - strokes: [B, num_slots, 10]
      - 前 8 维：P0, P1, P2, P3 的坐标（范围 [-0.5, 1.5]）
      - 后 2 维：w_start, w_end（范围 [0, 10]）
    - pen_state: [B, num_slots, 3] 笔画状态 logits
      - 0: Null (无效)
      - 1: New/Start (新笔画起点)
      - 2: Continue (延续上一笔)
    - aux_outputs: List[Dict] 每个 refine step 的中间输出 (用于 Aux Loss)
    """

    def __init__(
        self,
        embed_dim=128,
        num_slots=8,  # 最多预测 8 个笔画
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        num_refine_steps=3,  # 迭代优化次数 (增加到3)
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.num_refine_steps = num_refine_steps

        # 1. Learnable Slot Queries (可学习的内容查询)
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, embed_dim) * 0.02)

        # 2. 2D 空间位置先验 (让 Slot 天然负责不同区域)
        self._init_spatial_prior()
        self.spatial_embed = nn.Linear(2, embed_dim)

        # 3. 可学习的参考点 (每个 Slot 初始化一个空间位置)
        self.reference_points = nn.Parameter(torch.zeros(1, num_slots, 2))
        self._init_reference_points()

        # 4. Slot Self-Attention (让 Slots 互相感知)
        self.slot_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.slot_self_attn_norm = nn.LayerNorm(embed_dim)

        # 5. Cross-Attention Decoder (Slots attend to Encoder features)
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

        # 6. 笔画参数预测头 (更深的网络，带残差)
        self.stroke_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 10),
        )

        # 7. 笔画状态预测头 (3分类: Null, New, Continue)
        self.pen_state_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 3),
        )

        # 8. 迭代优化：将上一轮预测编码回 Slot
        self.refine_encoder = nn.Linear(10, embed_dim)

        # 9. 去噪训练：将带噪声的 GT 编码为 Query
        self.dn_query_encoder = nn.Sequential(
            nn.Linear(10, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 10. Layer Norm
        self.norm = nn.LayerNorm(embed_dim)

    def _init_reference_points(self):
        """初始化参考点为网格分布"""
        grid_h = int(math.ceil(math.sqrt(self.num_slots)))
        grid_w = int(math.ceil(self.num_slots / grid_h))

        for i in range(self.num_slots):
            row = i // grid_w
            col = i % grid_w
            y = (row + 0.5) / grid_h
            x = (col + 0.5) / grid_w
            self.reference_points.data[0, i, 0] = x
            self.reference_points.data[0, i, 1] = y

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

    def _predict_from_slots(self, decoded_slots):
        """
        从 decoded slots 预测笔画参数和状态

        Args:
            decoded_slots: [B, num_slots, embed_dim]

        Returns:
            strokes: [B, num_slots, 10] 归一化后的笔画参数
            pen_state_logits: [B, num_slots, 3] 状态 logits (未 softmax)
        """
        # 笔画参数预测
        strokes_raw = self.stroke_head(decoded_slots)  # [B, num_slots, 10]

        # 归一化坐标到 [-0.5, 1.5] (Sigmoid * 2.0 - 0.5) 以覆盖画面外控制点
        strokes = strokes_raw.clone()
        strokes[..., :8] = torch.sigmoid(strokes_raw[..., :8]) * 2.0 - 0.5
        # 归一化宽度到 [0, 10]
        strokes[..., 8:10] = torch.sigmoid(strokes_raw[..., 8:10]) * 10.0

        # 笔画状态预测 (返回 logits，不做 softmax)
        pen_state_logits = self.pen_state_head(decoded_slots)  # [B, num_slots, 3]

        return strokes, pen_state_logits

    def forward(self, encoder_features, dn_queries=None, return_aux=True):
        """
        Args:
            encoder_features: [B, seq_len, embed_dim]
            dn_queries: Optional[Tensor] 去噪训练的带噪声 GT Query
                       形状 [B, num_dn, 10] (坐标 + 宽度)
            return_aux: bool 是否返回辅助输出 (用于 Deep Supervision)

        Returns:
            strokes: [B, num_slots, 10] 最终笔画参数
            pen_state_logits: [B, num_slots, 3] 笔画状态 logits
            aux_outputs: List[Dict] 每个 refine step 的中间输出 (仅当 return_aux=True)
                        每个 Dict 包含 {'strokes': ..., 'pen_state_logits': ...}
        """
        B = encoder_features.shape[0]
        device = encoder_features.device

        # 1. 初始化 Slot Queries + 空间位置先验
        slots = self.slot_queries.expand(B, -1, -1)  # [B, num_slots, embed_dim]
        spatial_embeds = self.spatial_embed(
            self.spatial_positions
        )  # [num_slots, embed_dim]
        slots = slots + spatial_embeds.unsqueeze(0)  # [B, num_slots, embed_dim]

        # 2. 添加参考点编码
        ref_embeds = self.spatial_embed(
            self.reference_points.expand(B, -1, -1)
        )  # [B, num_slots, embed_dim]
        slots = slots + ref_embeds

        # 3. 处理去噪 Query (如果有)
        num_dn = 0
        if dn_queries is not None:
            num_dn = dn_queries.shape[1]
            dn_embeds = self.dn_query_encoder(dn_queries)  # [B, num_dn, embed_dim]
            # 拼接到 slots 前面
            slots = torch.cat(
                [dn_embeds, slots], dim=1
            )  # [B, num_dn + num_slots, embed_dim]

        # 4. Slot Self-Attention (让 Slots 互相感知)
        slots_residual = slots
        slots, _ = self.slot_self_attn(slots, slots, slots)
        slots = self.slot_self_attn_norm(slots + slots_residual)

        # 5. 迭代式解码 (收集辅助输出)
        aux_outputs = []

        for step in range(self.num_refine_steps):
            # Cross-Attention: Slots attend to Encoder features
            decoded_slots = self.decoder(
                tgt=slots,  # [B, num_dn + num_slots, embed_dim]
                memory=encoder_features,  # [B, 64, embed_dim]
            )

            # Layer Norm
            decoded_slots = self.norm(decoded_slots)

            # 预测当前轮的笔画参数
            strokes, pen_state_logits = self._predict_from_slots(decoded_slots)

            # 收集辅助输出 (除了最后一轮)
            if return_aux and step < self.num_refine_steps - 1:
                # 分离去噪部分和正常部分
                if num_dn > 0:
                    aux_outputs.append(
                        {
                            "strokes": strokes[:, num_dn:],  # [B, num_slots, 10]
                            "pen_state_logits": pen_state_logits[:, num_dn:],
                            "dn_strokes": strokes[:, :num_dn],  # [B, num_dn, 10]
                            "dn_pen_state_logits": pen_state_logits[:, :num_dn],
                        }
                    )
                else:
                    aux_outputs.append(
                        {
                            "strokes": strokes,
                            "pen_state_logits": pen_state_logits,
                        }
                    )

            # 如果不是最后一轮，将预测结果编码回 Slot 进行优化
            if step < self.num_refine_steps - 1:
                # 将预测编码回 Slot
                slots = slots + self.refine_encoder(strokes)
                # 再次 Self-Attention
                slots_residual = slots
                slots, _ = self.slot_self_attn(slots, slots, slots)
                slots = self.slot_self_attn_norm(slots + slots_residual)

        # 6. 分离最终输出
        if num_dn > 0:
            final_strokes = strokes[:, num_dn:]  # [B, num_slots, 10]
            final_pen_state = pen_state_logits[:, num_dn:]  # [B, num_slots, 3]
            dn_strokes = strokes[:, :num_dn]
            dn_pen_state = pen_state_logits[:, :num_dn]

            if return_aux:
                return (
                    final_strokes,
                    final_pen_state,
                    aux_outputs,
                    dn_strokes,
                    dn_pen_state,
                )
            else:
                return final_strokes, final_pen_state, dn_strokes, dn_pen_state
        else:
            if return_aux:
                return strokes, pen_state_logits, aux_outputs
            else:
                return strokes, pen_state_logits

    def forward_inference(self, encoder_features):
        """
        推理模式：简化输出，只返回最终预测

        Args:
            encoder_features: [B, seq_len, embed_dim]

        Returns:
            strokes: [B, num_slots, 10]
            pen_state: [B, num_slots, 3] softmax 概率
        """
        strokes, pen_state_logits = self.forward(
            encoder_features, dn_queries=None, return_aux=False
        )
        pen_state = torch.softmax(pen_state_logits, dim=-1)
        return strokes, pen_state
