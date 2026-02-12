import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import LargeKernelBlock, AddCoords

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
        self.add_coords = AddCoords(height=64, width=64, num_freqs=1)  # num_freqs=1 adds 6 channels

        # ==========================================
        # Shared Stem with Early Coordinate Injection
        # ==========================================
        # 输入特征 + 6通道坐标网格 (num_freqs=1) -> 调整通道 -> 强特征提取
        coord_in_channels = in_channels + self.add_coords.added_channels
        self.coord_conv = nn.Sequential(
            nn.Conv2d(coord_in_channels, head_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(head_channels),
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
        # 融合特征：共享特征 + 原始输入 x（保留详细信息）
        # 不使用 skel_guide/tan_guide（因为只是简单线性投影，信息量有限）
        fusion_channels = head_channels + in_channels
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

        x_with_coord = self.add_coords(x)  # [B, C+6, H, W]
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

        # ----- 关键点融合与预测 -----
        # 使用共享特征 + 原始输入，保留详细信息
        fusion_input = torch.cat([feat_stem, x], dim=1)
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
        """
        完整初始化策略:
        1. 稀疏任务偏置初始化 (logits bias = -2.19, sigmoid ≈ 0.1)
        2. 所有输出卷积用小 gain 初始化 (0.1)
        3. BatchNorm 权重初始化为 1
        """
        # 稀疏任务偏置初始化 (更保守的 -2.19)
        sparse_bias = -2.19  # sigmoid ≈ 0.1

        if hasattr(self, "skel_conv") and isinstance(self.skel_conv, nn.Conv2d):
            nn.init.xavier_normal_(self.skel_conv.weight, gain=0.1)
            if self.skel_conv.bias is not None:
                nn.init.constant_(self.skel_conv.bias, sparse_bias)

        if hasattr(self, "keypoints") and isinstance(self.keypoints, nn.Conv2d):
            nn.init.xavier_normal_(self.keypoints.weight, gain=0.1)
            if self.keypoints.bias is not None:
                nn.init.constant_(self.keypoints.bias, sparse_bias)

        # 其他输出卷积也用小 gain 初始化
        for conv in [self.tangent, self.width_conv, self.offset_conv]:
            if isinstance(conv, nn.Conv2d):
                nn.init.xavier_normal_(conv.weight, gain=0.1)
                if conv.bias is not None:
                    nn.init.constant_(conv.bias, 0)

        # BatchNorm 权重初始化为 1
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                module.momentum = 0.01  # 更小的 momentum，更稳定