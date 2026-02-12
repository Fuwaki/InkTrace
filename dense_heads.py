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
    混合密集预测头（坐标注入共享主干版）

    设计特点：
    - 坐标网格在进入共享主干前即拼接，所有任务共享位置先验
    - 强共享特征 + 骨架独立微增强 + 其他头直接投影
    - 关键点头利用共享特征及骨架/切线引导，但不再重复注入坐标

    参数：
        in_channels : 解码器输入特征通道数
        head_channels: 共享特征通道数（默认64）
        detach_guidance: 是否断开骨架/切线到关键点的梯度（默认False，端到端）
    """

    def __init__(self, in_channels, head_channels=64, detach_guidance=False):
        super().__init__()
        self.detach_guidance = detach_guidance

        # ==========================================
        # Shared Stem with Early Coordinate Injection
        # ==========================================
        # 输入特征 + 2通道坐标网格 -> 调整通道 -> 强特征提取
        self.coord_conv = nn.Sequential(
            Conv2d_BN(in_channels + 2, head_channels, 3, 1, 1),  # 含 BN
            nn.GELU(),
        )
        self.shared_blocks = nn.Sequential(
            LargeKernelBlock(head_channels, kernel_size=5),
            LargeKernelBlock(head_channels, kernel_size=5),
        )

        # ==========================================
        # Stage 1: Pixel-Level Tasks (Parallel)
        # ==========================================
        # ----- 1. Skeleton Head (Enhanced) -----
        self.skel_enhance = LargeKernelBlock(head_channels, kernel_size=5)
        self.skel_conv = nn.Conv2d(head_channels, 1, 1)  # logits

        # ----- 2. Tangent Head (Direct Projection) -----
        self.tangent = nn.Conv2d(head_channels, 2, 1)  # raw vectors

        # ----- 3. Width Head (Direct Projection) -----
        self.width_conv = nn.Conv2d(head_channels, 1, 1)
        self.width_act = nn.Softplus()

        # ----- 4. Offset Head (Direct Projection) -----
        self.offset_conv = nn.Conv2d(head_channels, 2, 1)
        self.offset_scale = 0.5

        # ==========================================
        # Stage 2: Topological Task (Cascaded Keypoints)
        # ==========================================
        # 融合特征：共享特征 + 骨架logits + 切线原始向量 (不再包含坐标网格)
        fusion_channels = head_channels + 1 + 2
        self.keypoint_fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )

        # 关键点特征增强：2×LKB(5) 替代原单一大核11x11，更精细
        self.kp_enhance = nn.Sequential(
            LargeKernelBlock(head_channels, kernel_size=5),
            LargeKernelBlock(head_channels, kernel_size=5),
        )

        # 关键点输出：直接1×1卷积到2通道logits
        self.keypoints = nn.Conv2d(head_channels, 2, 1)

        self._init_weights()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 来自解码器的特征图（例如 64×64）
        Returns:
            dict: skeleton, tangent, width, offset, keypoints
        """
        B, _, H, W = x.shape

        # ----- 生成坐标网格（[-1, 1] 归一化）-----
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
        coord = torch.cat([x_grid, y_grid], dim=1)  # [B, 2, H, W]

        # ----- 坐标注入 + 共享特征提取 -----
        x_with_coord = torch.cat([x, coord], dim=1)  # [B, C+2, H, W]
        feat_stem = self.coord_conv(x_with_coord)  # [B, head_channels, H, W]
        feat_stem = self.shared_blocks(feat_stem)  # 2×LKB

        # ----- 骨架预测（增强）-----
        skel_feat = self.skel_enhance(feat_stem)  # 1×LKB
        skel_logits = self.skel_conv(skel_feat)  # [B, 1, H, W]
        skel_pred = torch.sigmoid(skel_logits)

        # ----- 切线预测（直接投影 + 单位圆归一化）-----
        tan_raw = self.tangent(feat_stem)  # [B, 2, H, W]
        tan_pred = F.normalize(tan_raw, dim=1, eps=1e-6)

        # ----- 宽度预测（直接投影 + Softplus）-----
        width_logits = self.width_conv(feat_stem)
        width_pred = self.width_act(width_logits)

        # ----- 偏移预测（直接投影 + Tanh缩放）-----
        offset_raw = self.offset_conv(feat_stem)
        offset_pred = torch.tanh(offset_raw) * self.offset_scale

        # ----- 引导特征（梯度控制）-----
        if self.detach_guidance:
            skel_guide = skel_logits.detach()
            tan_guide = tan_raw.detach()
        else:
            skel_guide = skel_logits
            tan_guide = tan_raw

        # ----- 关键点融合与预测（不再拼接坐标）-----
        fusion_input = torch.cat([feat_stem, skel_guide, tan_guide], dim=1)
        feat_fuse = self.keypoint_fusion(fusion_input)
        feat_fuse = self.kp_enhance(feat_fuse)  # 2×LKB
        kp_logits = self.keypoints(feat_fuse)  # [B, 2, H, W]
        kp_pred = torch.sigmoid(kp_logits)

        return {
            "skeleton": skel_pred,
            "tangent": tan_pred,
            "width": width_pred,
            "offset": offset_pred,
            "keypoints": kp_pred,
        }

    def _init_weights(self):
        """稀疏任务偏置初始化 (logits bias = -4.59)"""
        if hasattr(self, "skel_conv") and isinstance(self.skel_conv, nn.Conv2d):
            if self.skel_conv.bias is not None:
                nn.init.constant_(self.skel_conv.bias, -4.59)
        if hasattr(self, "keypoints") and isinstance(self.keypoints, nn.Conv2d):
            if self.keypoints.bias is not None:
                nn.init.constant_(self.keypoints.bias, -4.59)