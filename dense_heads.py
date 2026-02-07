import torch
import torch.nn as nn
import torch.nn.functional as F
from RepVit import Conv2d_BN


# =============================================================================
# Multi-Scale Context Modules
# =============================================================================


class SkeletonMSCA(nn.Module):
    """
    Multi-Scale Context Aggregation for Skeleton Structures

    使用 Strip Conv (条形卷积) 捕捉长距离笔画结构
    相比 ASPP 的优势：针对长条形结构更高效
    """

    def __init__(self, dim):
        super().__init__()

        # 1. Local feature (5x5)
        self.conv_local = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 2. Medium-range strip conv (11px) - 捕捉中等长度笔画片段
        self.conv_med_h = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv_med_v = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        # 3. Long-range strip conv (21px) - 捕捉长笔画和全局结构
        self.conv_long_h = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv_long_v = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        # 4. Feature fusion with gating
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),  # Gating mechanism
        )

        # 5. Residual projection
        self.res_proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Enhanced features with long-range context
        """
        identity = x

        # Local feature
        attn = self.conv_local(x)

        # Multi-scale strip convolutions
        attn_med = self.conv_med_v(self.conv_med_h(attn))
        attn_long = self.conv_long_v(self.conv_long_h(attn))

        # Aggregate multi-scale features
        attn = attn + attn_med + attn_long

        # Gating (soft attention)
        gate = self.fusion(attn)

        # Output with residual connection
        return identity * gate + self.res_proj(attn)


class DenseHeads(nn.Module):
    """
    Hybrid Dense Prediction Heads

    架构设计：
    - Stage 1 (Parallel): Skeleton, Tangent, Width, Offset - 同时输出
    - Stage 2 (Cascaded): Keypoints - 利用 Skeleton 作为 attention 引导

    参数：
        detach_skel: 是否断开 Skeleton 的梯度
            - True: 训练稳定，但 Keypoints 无法修正 Skeleton
            - False: 端到端优化，但可能梯度冲突
    """

    def __init__(self, in_channels, head_channels=64, full_heads=True, detach_skel=True):
        super().__init__()
        self.full_heads = full_heads
        self.detach_skel = detach_skel

        # ==========================================
        # Shared Stem
        # ==========================================
        self.shared_conv = nn.Sequential(
            Conv2d_BN(in_channels, head_channels, 3, 1, 1), nn.GELU()
        )

        # ==========================================
        # Stage 1: Pixel-Level Tasks (Parallel)
        # ==========================================
        # 1. Skeleton Map (1ch, Sigmoid)
        self.skeleton = nn.Sequential(nn.Conv2d(head_channels, 1, 1), nn.Sigmoid())

        # 2. Tangent Field (2ch, Tanh) - cos2θ, sin2θ
        self.tangent = nn.Sequential(nn.Conv2d(head_channels, 2, 1), nn.Tanh())

        if self.full_heads:
            # 3. Width Map (1ch, Softplus)
            self.width = nn.Sequential(nn.Conv2d(head_channels, 1, 1), nn.Softplus())

            # 4. Offset Map (2ch, scaled Tanh)
            self.offset_conv = nn.Conv2d(head_channels, 2, 1)

            # ==========================================
            # Stage 2: Topological Task (Cascaded)
            # ==========================================
            # Keypoints 需要知道：
            # - 骨架位置 (哪里有关键点)
            # - 切线方向 (什么类型的关键点：端点/交叉/拐点)
            # - 绝对坐标 (空间位置)

            # Fusion: Stem(64) + Skeleton(1) + Tangent(2) + Coords(2) = 69 → 64
            self.keypoint_fusion = nn.Sequential(
                nn.Conv2d(head_channels + 1 + 2 + 2, head_channels, 1, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True),
            )

            # MSCA for long-range dependency
            self.geo_msca = SkeletonMSCA(dim=head_channels)

            # 5. Keypoints Map (2ch, Sigmoid)
            #    Ch0: Topological nodes (endpoints, junctions) - MUST break
            #    Ch1: Geometric anchors (sharp turns, inflections) - SHOULD break
            self.keypoints = nn.Sequential(
                nn.Conv2d(head_channels, head_channels // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(head_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels // 2, 2, 1),
                nn.Sigmoid(),
            )

        self._init_weights()

    def forward(self, x):
        """
        Args:
            x: [B, 64, 64, 64] Feature map from Decoder
        Returns:
            dict with skeleton, tangent, width, offset, keypoints
        """
        B, _, H, W = x.shape

        # ==========================================
        # Stage 1: Parallel Predictions
        # ==========================================
        feat_stem = self.shared_conv(x)

        skel_pred = self.skeleton(feat_stem)
        tan_pred = self.tangent(feat_stem)

        outputs = {
            "skeleton": skel_pred,
            "tangent": tan_pred,
        }

        if self.full_heads:
            # Pixel tasks (parallel)
            out_width = self.width(feat_stem)
            out_offset = torch.tanh(self.offset_conv(feat_stem)) * 0.5

            # ==========================================
            # Stage 2: Cascaded Keypoints
            # ==========================================
            # 生成坐标网格 [-1, 1]
            y_grid = (
                torch.linspace(-1, 1, H, device=x.device)
                .view(1, 1, H, 1)
                .expand(B, 1, H, W)
            )
            x_grid = (
                torch.linspace(-1, 1, W, device=x.device)
                .view(1, 1, 1, W)
                .expand(B, 1, H, W)
            )

            # 准备 Skeleton 引导（可选 detach）
            # Tangent 不需要 detach，因为它的几何信息对 keypoints 很重要
            skel_guide = skel_pred.detach() if self.detach_skel else skel_pred

            # 拼接：原始特征 + 骨架引导 + 切线方向 + 位置编码
            fusion_input = torch.cat(
                [feat_stem, skel_guide, tan_pred, x_grid, y_grid], dim=1
            )  # [B, 64+1+2+2, H, W] = [B, 69, H, W]

            # 特征融合 + 多尺度上下文聚合
            feat_key = self.keypoint_fusion(fusion_input)
            feat_key = self.geo_msca(feat_key)

            # Keypoints 预测
            out_keys = self.keypoints(feat_key)

            outputs.update(
                {
                    "width": out_width,
                    "offset": out_offset,
                    "keypoints": out_keys,  # [B, 2, H, W]
                }
            )

        return outputs

    def _init_weights(self):
        """
        Tip A: Sigmoid Bias Initialization
        对于极度稀疏的任务（Skeleton, Keypoints 只有 <5% 前景），
        将最后一层 bias 初始化为 -4.59 (ln(0.01/0.99))。
        不仅加速收敛，还能防止 loss 在训练初期爆炸。
        """
        # Skeleton: Sparse foreground
        if isinstance(self.skeleton[0], nn.Conv2d):
            nn.init.constant_(self.skeleton[0].bias, -4.59)

        # Keypoints: Extremely sparse (Conv2d before Sigmoid)
        # BUG FIX: keypoints[-1] 是 Sigmoid (无 bias)，需要用 [-2] 获取最后的 Conv2d
        if self.full_heads:
            last_conv = self.keypoints[-2]  # Sigmoid 前的 Conv2d
            if isinstance(last_conv, nn.Conv2d):
                nn.init.constant_(last_conv.bias, -4.59)
