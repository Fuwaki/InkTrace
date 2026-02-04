"""
统一的模型接口 (Refactored)

结构:
- Encoder: encoder.py
- Decoder: decoder.py
- Heads: dense_heads.py / detr_decoder.py
- Model Factory: models.py

"""

import torch
import torch.nn as nn
from encoder import StrokeEncoder
from decoder import PixelDecoder, DenseDecoder
from dense_heads import DenseHeads

# Optional imports for other phases
try:
    from detr_decoder import DETRVectorDecoder
except ImportError:
    DETRVectorDecoder = None


class ReconstructionModel(nn.Module):
    """
    重建模型 (Phase 1)
    Encoder + PixelDecoder
    """

    def __init__(self, encoder, pixel_decoder):
        super().__init__()
        self.encoder = encoder
        self.pixel_decoder = pixel_decoder

    def forward(self, x):
        embeddings = self.encoder(x)
        reconstructed = self.pixel_decoder(embeddings)
        return reconstructed, embeddings


class DenseVectorModel(nn.Module):
    """
    InkTrace V4 (Final Paradigm): Dense Prediction Model
    Encoder + DenseDecoder + DenseHeads
    """

    def __init__(self, encoder, decoder=None, heads=None):
        super().__init__()
        self.encoder = encoder

        # Infer embed_dim from encoder for default decoder initialization
        if decoder is None:
            embed_dim = getattr(encoder, "token_embed", None)
            if embed_dim is not None:
                embed_dim = embed_dim.out_features
            else:
                embed_dim = 128
            self.decoder = DenseDecoder(embed_dim=embed_dim)
        else:
            self.decoder = decoder

        self.heads = heads or DenseHeads(in_channels=64)

    def forward(self, x):
        # 1. Encoder Pass (Get intermediate layers)
        features, _ = self.encoder(x, return_interm_layers=True)
        # features = [f1, f2, f3]

        # 2. Decoder Pass
        d3, aux_outputs = self.decoder(features)

        # 3. Heads
        outputs = self.heads(d3)

        # 4. Deep Supervision (Auxiliary Skeleton Heads)
        if self.training:
            outputs.update(aux_outputs)

        return outputs


class ModelFactory:
    """
    模型工厂类
    """

    @staticmethod
    def create_reconstruction_model(
        embed_dim=128, num_heads=4, num_layers=6, device="cpu"
    ):
        """创建 Reconstruction Model (Phase 1)"""
        encoder = StrokeEncoder(
            in_channels=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=0.1,
        )
        pixel_decoder = PixelDecoder(embed_dim=embed_dim)
        model = ReconstructionModel(encoder, pixel_decoder).to(device)
        return model

    @staticmethod
    def create_dense_model(embed_dim=128, device="cpu", encoder_ckpt=None):
        """创建 Dense Vector Model (Phase V4)"""
        encoder = StrokeEncoder(in_channels=1, embed_dim=embed_dim)

        # Load Pretrained Encoder if path provided
        if encoder_ckpt:
            print(f"Loading pretrained encoder from {encoder_ckpt}")
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            if "encoder_state_dict" in ckpt:
                state_dict = ckpt["encoder_state_dict"]
            elif "model_state_dict" in ckpt:
                state_dict = {
                    k.replace("encoder.", ""): v
                    for k, v in ckpt["model_state_dict"].items()
                    if k.startswith("encoder.")
                }
            else:
                state_dict = ckpt

            # Load roughly
            encoder.load_state_dict(state_dict, strict=False)
            print("Encoder weights loaded.")

        model = DenseVectorModel(encoder).to(device)
        return model

    @staticmethod
    def load_reconstruction_model(checkpoint_path, device="cpu"):
        """加载重建模型"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Assuming embed_dim=128 for now, ideally save config in checkpoint
        model = ModelFactory.create_reconstruction_model(embed_dim=128, device=device)

        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        if "decoder_state_dict" in checkpoint:
            model.pixel_decoder.load_state_dict(checkpoint["decoder_state_dict"])
        elif "pixel_decoder_state_dict" in checkpoint:
            model.pixel_decoder.load_state_dict(checkpoint["pixel_decoder_state_dict"])

        return model

    @staticmethod
    def load_dense_model(checkpoint_path, device="cpu"):
        """加载 Dense Model"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Assuming embed_dim=128
        model = ModelFactory.create_dense_model(embed_dim=128, device=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Fallback for separate keys if saved differently
            if "encoder_state_dict" in checkpoint:
                model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            # ... handle decoder loading if split

        return model
