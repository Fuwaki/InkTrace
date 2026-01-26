import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    """
    轻量级像素解码器 (Phase 1 专用)
    输入: [B, 64, embed_dim] (来自 Encoder 的序列)
    输出: [B, 1, 64, 64] (重构图像)
    """
    def __init__(self, embed_dim=128):
        super().__init__()

        # 1. 序列转回特征图的预处理
        # 此时输入是序列，我们先把它 reshape 回 8x8
        self.embed_dim = embed_dim

        # 2. 上采样模块 (Upsample Block)
        # 结构: [上采样 -> 卷积 -> BN -> GELU]
        # 目标: 8 -> 16 -> 32 -> 64 (3次上采样)

        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), # 8 -> 16
            nn.Conv2d(embed_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), # 16 -> 32
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), # 32 -> 64
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )

        # 3. 输出层 (Output Layer)
        # 最后一层把通道压缩为 1，并用 Sigmoid 归一化到 0-1
        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, 64, embed_dim]  <-- 来自 Encoder 的输出

        # Step 1: Reshape (序列 -> 图像)
        # [B, 64, embed_dim] -> [B, embed_dim, 64] -> [B, embed_dim, 8, 8]
        B, N, C = x.shape
        H = W = int(N ** 0.5) # 根号64 = 8
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Step 2: 逐级放大
        x = self.layer1(x) # -> [B, 64, 16, 16]
        x = self.layer2(x) # -> [B, 32, 32, 32]
        x = self.layer3(x) # -> [B, 16, 64, 64]

        # Step 3: 输出图像
        img = self.head(x) # -> [B, 1, 64, 64]

        return img


class ReconstructionModel(nn.Module):
    """完整的重建模型：Encoder + Decoder"""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """
        Args:
            x: [B, 1, 64, 64] 输入图像
        Returns:
            reconstructed: [B, 1, 64, 64] 重建图像
            embeddings: [B, 64, embed_dim] 中间表示
        """
        # 编码
        embeddings = self.encoder(x)

        # 解码重建
        reconstructed = self.decoder(embeddings)

        return reconstructed, embeddings
