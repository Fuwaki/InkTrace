"""
统一的模型接口

提供完整的模型类，组合 Encoder 和不同的 Decoder
支持灵活的训练和推理模式
"""

import torch
import torch.nn as nn


class StrokeEncoder(nn.Module):
    """
    笔画编码器（RepViT + Transformer）

    这里的实现与 model.py 中的 StrokeEncoder 相同
    但为了更好的封装，我们直接导入
    """
    pass  # 实际从 model.py 导入


class ReconstructionModel(nn.Module):
    """
    重建模型（Phase 1, 1.5, 1.6 使用）

    Encoder + PixelDecoder
    用于图像重建任务
    """

    def __init__(self, encoder, pixel_decoder):
        super().__init__()
        self.encoder = encoder
        self.pixel_decoder = pixel_decoder

    def forward(self, x):
        """
        Args:
            x: [B, 1, 64, 64] 输入图像

        Returns:
            reconstructed: [B, 1, 64, 64] 重建图像
            embeddings: [B, 64, embed_dim] 中间表示
        """
        embeddings = self.encoder(x)
        reconstructed = self.pixel_decoder(embeddings)
        return reconstructed, embeddings

    def get_embeddings(self, x):
        """只获取 embeddings，不重建"""
        return self.encoder(x)


class VectorizationModel(nn.Module):
    """
    矢量化模型（Phase 2 使用）

    Encoder + DETR Decoder + (可选的) Pixel Decoder

    支持多种模式：
    - 'vectorize': 只输出矢量
    - 'reconstruct': 只输出重建（需要 pixel_decoder）
    - 'both': 同时输出矢量和重建
    """

    def __init__(self, encoder, detr_decoder, pixel_decoder=None):
        super().__init__()
        self.encoder = encoder
        self.detr_decoder = detr_decoder
        self.pixel_decoder = pixel_decoder  # 可选，用于计算重建损失

    def forward(self, x, mode='vectorize'):
        """
        Args:
            x: [B, 1, 64, 64] 输入图像
            mode: 'vectorize' | 'reconstruct' | 'both'

        Returns:
            根据 mode 返回不同的结果
        """
        embeddings = self.encoder(x)

        if mode == 'vectorize':
            # 只输出矢量
            strokes, validity = self.detr_decoder(embeddings)
            return strokes, validity, None

        elif mode == 'reconstruct':
            # 只输出重建（需要 pixel_decoder）
            if self.pixel_decoder is None:
                raise ValueError("pixel_decoder is required for reconstruction mode")
            reconstructed = self.pixel_decoder(embeddings)
            return None, None, reconstructed

        elif mode == 'both':
            # 同时输出矢量和重建
            strokes, validity = self.detr_decoder(embeddings)
            reconstructed = None
            if self.pixel_decoder is not None:
                reconstructed = self.pixel_decoder(embeddings)
            return strokes, validity, reconstructed

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_embeddings(self, x):
        """只获取 embeddings"""
        return self.encoder(x)

    def freeze_encoder(self):
        """冻结 Encoder"""
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """解冻 Encoder"""
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_detr_decoder(self):
        """冻结 DETR Decoder"""
        self.detr_decoder.eval()
        for param in self.detr_decoder.parameters():
            param.requires_grad = False

    def unfreeze_detr_decoder(self):
        """解冻 DETR Decoder"""
        self.detr_decoder.train()
        for param in self.detr_decoder.parameters():
            param.requires_grad = True


class ModelFactory:
    """
    模型工厂类

    提供统一的模型创建和加载接口
    """

    @staticmethod
    def create_reconstruction_model(embed_dim=128, device='cpu'):
        """
        创建重建模型

        Returns:
            ReconstructionModel
        """
        from model import StrokeEncoder
        from pixel_decoder import PixelDecoder

        encoder = StrokeEncoder(
            in_channels=1,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=6,
            dropout=0.1
        ).to(device)

        pixel_decoder = PixelDecoder(embed_dim=embed_dim).to(device)

        return ReconstructionModel(encoder, pixel_decoder)

    @staticmethod
    def create_vectorization_model(embed_dim=128, num_slots=8, device='cpu',
                                  include_pixel_decoder=True):
        """
        创建矢量化模型

        Args:
            include_pixel_decoder: 是否包含 Pixel Decoder（用于计算重建损失）

        Returns:
            VectorizationModel
        """
        from model import StrokeEncoder
        from pixel_decoder import PixelDecoder
        from detr_decoder import DETRVectorDecoder

        encoder = StrokeEncoder(
            in_channels=1,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=6,
            dropout=0.1
        ).to(device)

        detr_decoder = DETRVectorDecoder(
            embed_dim=embed_dim,
            num_slots=num_slots,
            num_layers=3,
            num_heads=4,
            dropout=0.1
        ).to(device)

        pixel_decoder = None
        if include_pixel_decoder:
            pixel_decoder = PixelDecoder(embed_dim=embed_dim).to(device)

        return VectorizationModel(encoder, detr_decoder, pixel_decoder)

    @staticmethod
    def load_reconstruction_model(checkpoint_path, device='cpu'):
        """
        加载重建模型

        Returns:
            ReconstructionModel
        """
        from model import StrokeEncoder
        from pixel_decoder import PixelDecoder

        checkpoint = torch.load(checkpoint_path, map_location=device)

        encoder = StrokeEncoder(
            in_channels=1, embed_dim=128, num_heads=4, num_layers=6, dropout=0.1
        ).to(device)
        pixel_decoder = PixelDecoder(embed_dim=128).to(device)

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        pixel_decoder.load_state_dict(checkpoint['decoder_state_dict'])

        encoder.eval()
        pixel_decoder.eval()

        return ReconstructionModel(encoder, pixel_decoder)

    @staticmethod
    def load_vectorization_model(checkpoint_path, device='cpu',
                                  include_pixel_decoder=True):
        """
        加载矢量化模型

        Returns:
            VectorizationModel
        """
        from model import StrokeEncoder
        from pixel_decoder import PixelDecoder
        from detr_decoder import DETRVectorDecoder

        checkpoint = torch.load(checkpoint_path, map_location=device)

        encoder = StrokeEncoder(
            in_channels=1, embed_dim=128, num_heads=4, num_layers=6, dropout=0.1
        ).to(device)
        detr_decoder = DETRVectorDecoder(
            embed_dim=128, num_slots=8, num_layers=3, num_heads=4, dropout=0.1
        ).to(device)

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        detr_decoder.load_state_dict(checkpoint['detr_decoder_state_dict'])

        # 尝试加载 pixel_decoder（如果存在）
        pixel_decoder = None
        if include_pixel_decoder and 'pixel_decoder_state_dict' in checkpoint:
            pixel_decoder = PixelDecoder(embed_dim=128).to(device)
            pixel_decoder.load_state_dict(checkpoint['pixel_decoder_state_dict'])
            pixel_decoder.eval()

        encoder.eval()
        detr_decoder.eval()

        return VectorizationModel(encoder, detr_decoder, pixel_decoder)

    @staticmethod
    def save_model(model, save_path, epoch, loss, optimizer=None):
        """
        保存模型

        支持保存 ReconstructionModel 和 VectorizationModel
        """
        if isinstance(model, ReconstructionModel):
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.pixel_decoder.state_dict(),
                'loss': loss,
            }
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        elif isinstance(model, VectorizationModel):
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'detr_decoder_state_dict': model.detr_decoder.state_dict(),
                'loss': loss,
            }
            if model.pixel_decoder is not None:
                checkpoint['pixel_decoder_state_dict'] = model.pixel_decoder.state_dict()
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        else:
            raise ValueError(f"Unknown model type: {type(model)}")

        torch.save(checkpoint, save_path)
