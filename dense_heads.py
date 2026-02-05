import torch
import torch.nn as nn
import torch.nn.functional as F
from RepVit import Conv2d_BN


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for 64x64 feature maps.

    为什么在 Head 层需要 ASPP？
    - Transformer 在 8×8 提供全局上下文
    - 但经过 3 次上采样后，全局信息被稀释
    - ASPP 在最终分辨率重新聚合多尺度上下文
    - 对端点/交叉点检测特别有用（需要理解邻域结构）
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 64x64 适用的膨胀率
        dilations = [1, 2, 4, 6]

        # Branch 1: 1x1 Conv (点级特征)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Branch 2-4: 3x3 Atrous Conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilations[3],
                dilation=dilations[3],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Branch 5: Global Average Pooling (全局上下文)
        self.branch5_pool = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Fusion: 5 branches -> out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        feat5 = F.interpolate(
            self.branch5_conv(self.branch5_pool(x)),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        return self.fusion(torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1))


class DenseHeads(nn.Module):
    """
    Asymmetric Dense Prediction Heads

    Branch 1 (Pixel Tasks): skeleton, tangent, width, offset
        - 需要高频细节，简单卷积即可
        - 轻量级，复用预训练权重

    Branch 2 (Geometric Tasks): keypoints
        - 需要理解邻域拓扑结构
        - 使用 ASPP 聚合多尺度上下文
        - 使用 CoordConv 提供精确位置（Encoder 的坐标信息经过多层已模糊）
    """

    def __init__(self, in_channels, head_channels=64):
        super().__init__()

        # ==========================================
        # Branch 1: Pixel Tasks (轻量级)
        # ==========================================
        self.shared_conv = nn.Sequential(
            Conv2d_BN(in_channels, head_channels, 3, 1, 1), nn.GELU()
        )

        # 1. Skeleton Map (1ch, Sigmoid)
        self.skeleton = nn.Sequential(nn.Conv2d(head_channels, 1, 1), nn.Sigmoid())

        # 2. Tangent Field (2ch, Tanh) - cos2θ, sin2θ
        self.tangent = nn.Sequential(nn.Conv2d(head_channels, 2, 1), nn.Tanh())

        # 3. Width Map (1ch, Softplus)
        self.width = nn.Sequential(nn.Conv2d(head_channels, 1, 1), nn.Softplus())

        # 4. Offset Map (2ch, scaled Tanh)
        self.offset_conv = nn.Conv2d(head_channels, 2, 1)

        # ==========================================
        # Branch 2: Geometric Tasks (ASPP + Coord)
        # ==========================================
        # Input: in_channels + 2 (X, Y coords)
        # ASPP output: 32 channels (节省参数)
        self.geo_aspp = ASPP(in_channels=in_channels + 2, out_channels=32)

        # 5. Keypoints Map (2ch, Sigmoid)
        #    Ch0: Topological nodes (endpoints, junctions) - MUST break
        #    Ch1: Geometric anchors (sharp turns, inflections) - SHOULD break
        self.keypoints = nn.Sequential(nn.Conv2d(32, 2, 1), nn.Sigmoid())

        self._init_weights()

    def forward(self, x):
        """
        Args:
            x: [B, 64, 64, 64] Feature map from Decoder
        Returns:
            dict with skeleton, tangent, width, offset, keypoints
        """
        # --- Branch 1: Pixel Tasks ---
        feat_pixel = self.shared_conv(x)

        out_skel = self.skeleton(feat_pixel)
        out_tan = self.tangent(feat_pixel)
        out_width = self.width(feat_pixel)
        out_offset = torch.tanh(self.offset_conv(feat_pixel)) * 0.5

        # --- Branch 2: Geometric Tasks ---
        B, _, H, W = x.shape

        # 动态生成坐标网格 (不存储，节省显存)
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

        # Concat: [B, 64+2, 64, 64]
        x_geo = torch.cat([x, x_grid, y_grid], dim=1)

        # ASPP + Keypoints
        feat_geo = self.geo_aspp(x_geo)
        out_keys = self.keypoints(feat_geo)

        return {
            "skeleton": out_skel,
            "tangent": out_tan,
            "width": out_width,
            "offset": out_offset,
            "keypoints": out_keys,  # [B, 2, H, W]
        }

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

        # Keypoints: Extremely sparse
        if isinstance(self.keypoints[0], nn.Conv2d):
            nn.init.constant_(self.keypoints[0].bias, -4.59)
