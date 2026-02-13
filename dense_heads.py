import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import LargeKernelBlock


class DenseHeads(nn.Module):
    """
    密集预测头（调制架构适配版）

    改动:
    1. 移除 AddCoords（Encoder SpatialModulation 已覆盖位置信息）
    2. 保留足够的 head_channels 容量（默认48）
    3. 共享主干做强特征展开，各任务分支有独立增强能力
    """

    def __init__(self, in_channels=16, head_channels=48, detach_guidance=False):
        super().__init__()
        self.detach_guidance = detach_guidance

        # ==========================================
        # Shared Stem: 展开通道 (16→48)
        # 从 Decoder 的窄通道展开到任务所需的表达空间
        # 类似 FPN 的 "展开" 阶段
        # ==========================================
        self.shared_stem = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.GELU(),
            LargeKernelBlock(head_channels, kernel_size=5),
            LargeKernelBlock(head_channels, kernel_size=5),
        )

        # ==========================================
        # Stage 1: Pixel-Level Tasks (Parallel)
        # ==========================================

        # 1. Skeleton Head (独立增强)
        self.skel_enhance = LargeKernelBlock(head_channels, kernel_size=5)
        self.skel_conv = nn.Conv2d(head_channels, 1, 1)

        # 2. Tangent Head (直接投影)
        self.tangent = nn.Conv2d(head_channels, 2, 1)

        # 3. Width Head (直接投影)
        self.width_conv = nn.Conv2d(head_channels, 1, 1)
        self.width_act = nn.Softplus()

        # 4. Offset Head (直接投影)
        self.offset_conv = nn.Conv2d(head_channels, 2, 1)
        self.offset_scale = 0.5

        # ==========================================
        # Stage 2: Keypoints (级联融合)
        # shared_feat(48) + 原始输入(16) = 64 → 融合 → 增强
        # ==========================================
        fusion_channels = head_channels + in_channels  # 48 + 16 = 64

        self.keypoint_fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )
        self.kp_enhance = nn.Sequential(
            LargeKernelBlock(head_channels, kernel_size=5),
            LargeKernelBlock(head_channels, kernel_size=5),
        )
        self.keypoints = nn.Conv2d(head_channels, 2, 1)

        self._init_weights()

    def forward(self, x):
        """
        Args:
            x: [B, 16, 64, 64] 来自 Decoder
        Returns:
            dict: skeleton, tangent, width, offset, keypoints
        """
        # 共享特征展开
        feat = self.shared_stem(x)  # [B, 48, 64, 64]

        # ----- 骨架 -----
        skel_feat = self.skel_enhance(feat)
        skel_logits = self.skel_conv(skel_feat)
        skel_pred = torch.sigmoid(skel_logits)

        # ----- 切线 -----
        tan_raw = self.tangent(feat)
        tan_pred = F.normalize(tan_raw, dim=1, eps=1e-6)

        # ----- 宽度 -----
        width_logits = self.width_conv(feat)
        width_pred = self.width_act(width_logits)

        # ----- 偏移 -----
        offset_raw = self.offset_conv(feat)
        offset_pred = torch.tanh(offset_raw) * self.offset_scale

        # ----- 关键点 -----
        fusion_input = torch.cat([feat, x], dim=1)  # [B, 64, 64, 64]
        feat_fuse = self.keypoint_fusion(fusion_input)
        feat_fuse = self.kp_enhance(feat_fuse)
        kp_logits = self.keypoints(feat_fuse)
        kp_pred = torch.sigmoid(kp_logits)

        return {
            "skeleton": skel_pred,
            "tangent": tan_pred,
            "width": width_pred,
            "offset": offset_pred,
            "keypoints": kp_pred,
        }

    def _init_weights(self):
        sparse_bias = -2.19

        if hasattr(self, "skel_conv") and isinstance(self.skel_conv, nn.Conv2d):
            nn.init.xavier_normal_(self.skel_conv.weight, gain=0.1)
            if self.skel_conv.bias is not None:
                nn.init.constant_(self.skel_conv.bias, sparse_bias)

        if hasattr(self, "keypoints") and isinstance(self.keypoints, nn.Conv2d):
            nn.init.xavier_normal_(self.keypoints.weight, gain=0.1)
            if self.keypoints.bias is not None:
                nn.init.constant_(self.keypoints.bias, sparse_bias)

        for conv in [self.tangent, self.width_conv, self.offset_conv]:
            if isinstance(conv, nn.Conv2d):
                nn.init.xavier_normal_(conv.weight, gain=0.1)
                if conv.bias is not None:
                    nn.init.constant_(conv.bias, 0)

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                module.momentum = 0.01
