import torch
import torch.nn as nn
import torch.nn.functional as F
from RepVit import Conv2d_BN


# =============================================================================
# Multi-Scale Context Modules
# =============================================================================


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

    def __init__(self, in_channels, head_channels=64, detach_guidance=True):
        super().__init__()
        # 始终启用所有 heads（移除 full_heads 参数）
        self.detach_guidance = detach_guidance

        # ==========================================
        # Shared Stem
        # ==========================================
        self.shared_conv = nn.Sequential(
            Conv2d_BN(in_channels, head_channels, 3, 1, 1), nn.GELU()
        )

        # ==========================================
        # Stage 1: Pixel-Level Tasks (Parallel)
        # ==========================================
        # Per-head refinement layers (gradient buffering + non-linear transform)
        self.skel_refine = nn.Sequential(
            Conv2d_BN(head_channels, head_channels, 3, 1, 1),
            nn.GELU(),
        )
        self.tan_refine = nn.Sequential(
            Conv2d_BN(head_channels, head_channels, 3, 1, 1),
            nn.GELU(),
        )

        # 1. Skeleton Map: keep a sigmoid wrapper for compatibility,
        # but expose raw logits via `self.skel_conv` for later use
        self.skel_conv = nn.Conv2d(head_channels, 1, 1)
        self.skeleton = nn.Sequential(self.skel_conv, nn.Sigmoid())

        # 2. Tangent Field (2ch) - will be normalized to unit vectors (unit circle)
        # 输出为未归一化的 2 通道向量，前向中会归一化到单位圆
        self.tangent = nn.Conv2d(head_channels, 2, 1)

        # 3. Width Map (1ch, Softplus)
        self.width_refine = nn.Sequential(
            Conv2d_BN(head_channels, head_channels, 3, 1, 1),
            nn.GELU(),
        )
        self.width = nn.Sequential(nn.Conv2d(head_channels, 1, 1), nn.Softplus())

        # 4. Offset Map (2ch, scaled Tanh)
        self.offset_refine = nn.Sequential(
            Conv2d_BN(head_channels, head_channels, 3, 1, 1),
            nn.GELU(),
        )
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

        # Large-kernel block for long-range dependency
        self.geo_msca = LargeKernelBlock(dim=head_channels, kernel_size=11)

        # 5. Keypoints Map (2ch, Sigmoid)
        #    Ch0: Topological nodes (endpoints, junctions) - MUST break
        #    Ch1: Geometric anchors (sharp turns, inflections) - SHOULD break
        # Keypoint head produces logits; sigmoid applied in forward
        self.keypoints = nn.Sequential(
            nn.Conv2d(head_channels, head_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels // 2, 2, 1),
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

        feat_stem = self.shared_conv(x)

        # 使用 per-head refinement 层
        skel_feat = self.skel_refine(feat_stem)
        skel_logits = self.skel_conv(skel_feat)
        skel_pred = torch.sigmoid(skel_logits)

        # 原始切线向量，随后归一化到单位圆以确保长度为 1
        tan_feat = self.tan_refine(feat_stem)
        tan_raw = self.tangent(tan_feat)
        # 强制投影到单位圆
        tan_pred = F.normalize(tan_raw, dim=1, eps=1e-6)

        outputs = {
            "skeleton": skel_pred,
            "tangent": tan_pred,
        }

        # Pixel tasks (parallel)
        width_feat = self.width_refine(feat_stem)
        out_width = self.width(width_feat)

        offset_feat = self.offset_refine(feat_stem)
        out_offset = torch.tanh(self.offset_conv(offset_feat)) * 0.5

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

        # 准备 skeleton/tangent 引导（可选 detach）
        if self.detach_guidance:
            skel_guide = skel_logits.detach()
            tan_guide = tan_raw.detach()
        else:
            skel_guide = skel_logits
            tan_guide = tan_raw

        # 拼接：原始特征 + 骨架引导(logits) + 切线原始向量 + 位置编码
        fusion_input = torch.cat(
            [feat_stem, skel_guide, tan_guide, x_grid, y_grid], dim=1
        )

        # 特征融合 + 多尺度上下文聚合
        feat_key = self.keypoint_fusion(fusion_input)
        feat_key = self.geo_msca(feat_key)

        # Keypoints 预测 (logits -> sigmoid)
        kp_logits = self.keypoints(feat_key)
        kp_pred = torch.sigmoid(kp_logits)

        outputs.update({"width": out_width, "offset": out_offset, "keypoints": kp_pred})

        return outputs

    def _init_weights(self):
        """
        Tip A: Sigmoid Bias Initialization
        对于极度稀疏的任务（Skeleton, Keypoints 只有 <5% 前景），
        将最后一层 bias 初始化为 -4.59 (ln(0.01/0.99))。
        不仅加速收敛，还能防止 loss 在训练初期爆炸。
        """
        # Skeleton: Sparse foreground (initialize logits bias)
        if hasattr(self, "skel_conv") and isinstance(self.skel_conv, nn.Conv2d):
            if self.skel_conv.bias is not None:
                nn.init.constant_(self.skel_conv.bias, -4.59)

        # Keypoints: Extremely sparse (last conv outputs logits)
        last_conv = self.keypoints[-1]
        if isinstance(last_conv, nn.Conv2d) and last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, -4.59)
