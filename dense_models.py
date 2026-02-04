import torch
import torch.nn as nn
import torch.nn.functional as F
from RepVit import Conv2d_BN


class AttentionGate(nn.Module):
    """
    Attention Gate: Filter features from skip connection using signal from current decoder layer.
    """

    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Gate channels (from deeper layer)
            F_l: Skip connection channels (from shallower layer)
            F_int: Intermediate channels
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: Gate signal (Upsampled feature from prev decoder layer)
        x: Skip connection feature (from Encoder)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ConvBlock(nn.Module):
    """
    Standard Conv Block: Conv-BN-ReLU-Conv-BN-ReLU
    Using RepVGG style Conv2d_BN for efficiency if desired, but standard ResBlock is good.
    Here we use a simple residual block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d_BN(in_channels, out_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = Conv2d_BN(out_channels, out_channels, 3, 1, 1)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = Conv2d_BN(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x + res


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


class DenseVectorNet(nn.Module):
    """
    InkTrace V4 (Final Paradigm): Dense Prediction Model
    Encoder: RepViT + Transformer (StrokeEncoder)
    Decoder: Hybrid U-Net with Attention Gates
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        # Feature channels from encoder (Modified StrokeEncoder)
        # F1: 32 (32x32)
        # F2: 128 (16x16)
        # F3: embed_dim (8x8) - need projection if embed_dim != 128

        self.dims = [32, 128, 128]

        # Get embed_dim from encoder
        embed_dim = getattr(encoder, "token_embed", None)
        if embed_dim is not None:
            embed_dim = embed_dim.out_features
        else:
            embed_dim = 128  # fallback

        # Project F3 from embed_dim to 128 if needed
        self.f3_proj = nn.Identity()
        if embed_dim != 128:
            self.f3_proj = nn.Conv2d(embed_dim, 128, 1)

        # Decoder
        # Layer 1: 8x8 -> 16x16
        # Input: F3 (128). Skip: F2 (128).
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ag1 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv1 = ConvBlock(128 + 128, 128)  # Cat F3_up, F2_gated

        # Deep Supervision 1 (at 16x16)
        self.ds1_head = nn.Sequential(
            Conv2d_BN(128, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                32, 1, 1
            ),  # No Sigmoid here if using BCEWithLogits, but we use Sigmoid in Heads usually.
            # Let's stick to consistent output.
            # Wait, user spec said Skeleton Map uses Sigmoid.
        )

        # Layer 2: 16x16 -> 32x32
        # Input: D1 (128). Skip: F1 (32).
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ag2 = AttentionGate(F_g=128, F_l=32, F_int=16)
        self.conv2 = ConvBlock(128 + 32, 64)

        # Deep Supervision 2 (at 32x32)
        self.ds2_head = nn.Sequential(
            Conv2d_BN(64, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 1, 1)
        )

        # Layer 3: 32x32 -> 64x64
        # Input: D2 (64). Skip: None (Input image? No, usually just upsample)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = ConvBlock(64, 64)

        # Final Heads
        self.heads = DenseHeads(64, head_channels=64)

    def forward(self, x):
        # 1. Encoder Pass
        # Returns: [f1(32), f2(128), f3(embed_dim)], embedding
        features, _ = self.encoder(x, return_interm_layers=True)
        f1, f2, f3 = features

        # Project F3 to 128 channels if needed
        f3 = self.f3_proj(f3)

        # 2. Decoder Pass

        # Block 1 (8 -> 16)
        d1_up = self.up1(f3)  # [B, 128, 16, 16]
        f2_gated = self.ag1(d1_up, f2)  # [B, 128, 16, 16] (Gate=d1_up, Skip=f2)
        d1 = torch.cat([d1_up, f2_gated], dim=1)
        d1 = self.conv1(d1)  # [B, 128, 16, 16]

        # Block 2 (16 -> 32)
        d2_up = self.up2(d1)  # [B, 128, 32, 32]
        f1_gated = self.ag2(d2_up, f1)  # [B, 32, 32, 32]
        d2 = torch.cat([d2_up, f1_gated], dim=1)
        d2 = self.conv2(d2)  # [B, 64, 32, 32]

        # Block 3 (32 -> 64)
        d3_up = self.up3(d2)  # [B, 64, 64, 64]
        d3 = self.conv3(d3_up)  # [B, 64, 64, 64]

        # 3. Heads
        outputs = self.heads(d3)

        # 4. Deep Supervision (Auxiliary Skeleton Heads)
        if self.training:
            ds1 = torch.sigmoid(self.ds1_head(d1))  # 16x16
            ds2 = torch.sigmoid(self.ds2_head(d2))  # 32x32
            outputs["aux_skeleton_16"] = ds1
            outputs["aux_skeleton_32"] = ds2

        return outputs


class DenseLoss(nn.Module):
    """
    Multi-task loss for InkTrace V4
    L = L_skel + L_junc + L_tan + L_width + L_offset
    """

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            "skeleton": 10.0,
            "junction": 5.0,
            "tangent": 1.0,
            "width": 1.0,
            "offset": 1.0,
            "aux": 0.5,  # Weight for deep supervision
        }

    def _dice_loss(self, pred, target, smooth=1.0):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=[2, 3])
        loss = 1 - (
            (2.0 * intersection + smooth)
            / (pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + smooth)
        )
        return loss.mean()

    def forward(self, outputs, targets):
        """
        targets: dict of tensors
        """
        losses = {}

        # 1. Skeleton Loss (BCE + Dice)
        pred_skel = outputs["skeleton"]
        tgt_skel = targets["skeleton"]

        bce_skel = F.binary_cross_entropy(pred_skel, tgt_skel)
        dice_skel = self._dice_loss(pred_skel, tgt_skel)
        losses["loss_skel"] = self.weights["skeleton"] * (bce_skel + dice_skel)

        # 2. Junction Loss (MSE/BCE for Heatmap)
        # If junctions are splatted gaussians (0-1), MSE is good.
        # If binary points, BCE/Focal is better.
        # Current generator makes them binary or near binary.
        # Let's use MSE as per spec.
        pred_junc = outputs["junction"]
        tgt_junc = targets["junction"]
        losses["loss_junc"] = self.weights["junction"] * F.mse_loss(pred_junc, tgt_junc)

        # Masking for regression tasks
        # Only compute loss where skeleton is present (in GT)
        mask = (tgt_skel > 0.5).float()
        num_fg = mask.sum() + 1e-6

        # 3. Tangent Loss (Cosine Similarity or L2)
        # Targets are unit vectors (cos, sin). Pred is Tanh.
        # We should encourage pred to be unit vector?
        # Or just L2 loss on components.
        pred_tan = outputs["tangent"]  # [B, 2, H, W]
        tgt_tan = targets["tangent"]

        # L2 Loss on masked region
        l2_tan = (pred_tan - tgt_tan) ** 2
        l2_tan = (l2_tan * mask.unsqueeze(1)).sum() / num_fg / 2.0
        losses["loss_tan"] = self.weights["tangent"] * l2_tan

        # 4. Width Loss (L1)
        pred_width = outputs["width"]
        tgt_width = targets["width"]
        l1_width = torch.abs(pred_width - tgt_width)
        l1_width = (l1_width * mask).sum() / num_fg
        losses["loss_width"] = self.weights["width"] * l1_width

        # 5. Offset Loss (L1)
        pred_off = outputs["offset"]
        tgt_off = targets["offset"]
        l1_off = torch.abs(pred_off - tgt_off)
        l1_off = (l1_off * mask.unsqueeze(1)).sum() / num_fg / 2.0
        losses["loss_off"] = self.weights["offset"] * l1_off

        # 6. Aux Losses (Skeleton only)
        if "aux_skeleton_16" in outputs:
            # Downsample target
            tgt_16 = F.interpolate(tgt_skel, size=16, mode="bilinear")
            tgt_16 = (tgt_16 > 0.5).float()  # Binarize

            aux_pred = outputs["aux_skeleton_16"]
            aux_loss = F.binary_cross_entropy(aux_pred, tgt_16)
            losses["loss_aux_16"] = self.weights["aux"] * aux_loss

        if "aux_skeleton_32" in outputs:
            # Downsample target
            tgt_32 = F.interpolate(tgt_skel, size=32, mode="bilinear")
            tgt_32 = (tgt_32 > 0.5).float()

            aux_pred = outputs["aux_skeleton_32"]
            aux_loss = F.binary_cross_entropy(aux_pred, tgt_32)
            losses["loss_aux_32"] = self.weights["aux"] * aux_loss

        losses["total"] = sum(losses.values())
        return losses
