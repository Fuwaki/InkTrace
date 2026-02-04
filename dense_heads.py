import torch
import torch.nn as nn
from RepVit import Conv2d_BN


class DenseHeads(nn.Module):
    """
    Collection of 5 Dense Prediction Heads
    """

    def __init__(self, in_channels, head_channels=64):
        super().__init__()

        self.shared_conv = nn.Sequential(
            Conv2d_BN(in_channels, head_channels, 3, 1, 1), nn.GELU()
        )

        # 1. Skeleton Map (1ch, Sigmoid)
        self.skeleton = nn.Sequential(nn.Conv2d(head_channels, 1, 1), nn.Sigmoid())

        # 2. Junction Map (1ch, Sigmoid)
        self.junction = nn.Sequential(nn.Conv2d(head_channels, 1, 1), nn.Sigmoid())

        # 3. Tangent Field (2ch, Tanh) - cos2t, sin2t
        self.tangent = nn.Sequential(nn.Conv2d(head_channels, 2, 1), nn.Tanh())

        # 4. Width Map (1ch, Softplus for smooth positive values)
        # GT width is in pixel units (typically 0.5 ~ 10)
        # Softplus is smoother than ReLU and avoids dead gradients
        self.width = nn.Sequential(nn.Conv2d(head_channels, 1, 1), nn.Softplus())

        # 5. Offset Map (2ch, scaled Tanh)
        # GT offset is in [-0.5, 0.5], so we scale Tanh output by 0.5
        self.offset_conv = nn.Conv2d(head_channels, 2, 1)

    def forward(self, x):
        feat = self.shared_conv(x)
        # Offset: scale Tanh from [-1,1] to [-0.5, 0.5]
        offset = torch.tanh(self.offset_conv(feat)) * 0.5
        return {
            "skeleton": self.skeleton(feat),
            "junction": self.junction(feat),
            "tangent": self.tangent(feat),
            "width": self.width(feat),
            "offset": offset,
        }
