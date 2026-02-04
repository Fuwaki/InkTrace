import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLoss(nn.Module):
    """
    Multi-task loss for InkTrace V4
    L = L_skel + L_junc + L_tan + L_width + L_offset

    Note: Uses float32 casting for numerical stability under AMP.
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
        # Cast to float32 for numerical stability
        pred = pred.float().contiguous()
        target = target.float().contiguous()
        intersection = (pred * target).sum(dim=[2, 3])
        loss = 1 - (
            (2.0 * intersection + smooth)
            / (pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + smooth)
        )
        return loss.mean()

    def _safe_bce(self, pred, target):
        """
        AMP-safe BCE loss using F.binary_cross_entropy_with_logits.
        We convert sigmoid outputs back to logits for numerical stability.
        """
        # Cast to float32 and clamp for stability
        pred = pred.float().clamp(min=1e-6, max=1 - 1e-6)
        target = target.float()
        # Convert probability to logits: logit = log(p / (1-p))
        logits = torch.log(pred / (1 - pred))
        return F.binary_cross_entropy_with_logits(logits, target, reduction='mean')

    def forward(self, outputs, targets):
        """
        targets: dict of tensors
        """
        losses = {}

        # 1. Skeleton Loss (BCE + Dice)
        pred_skel = outputs["skeleton"]
        tgt_skel = targets["skeleton"]

        bce_skel = self._safe_bce(pred_skel, tgt_skel)
        dice_skel = self._dice_loss(pred_skel, tgt_skel)
        losses["loss_skel"] = self.weights["skeleton"] * (bce_skel + dice_skel)

        # 2. Junction Loss (MSE for Heatmap)
        pred_junc = outputs["junction"].float()
        tgt_junc = targets["junction"].float()
        losses["loss_junc"] = self.weights["junction"] * F.mse_loss(pred_junc, tgt_junc)

        # Masking for regression tasks
        # Only compute loss where skeleton is present (in GT)
        mask = (tgt_skel > 0.5).float()
        num_fg = mask.sum().clamp(min=1.0)  # At least 1 to avoid div by zero

        # 3. Tangent Loss (L2 on masked region)
        pred_tan = outputs["tangent"].float()  # [B, 2, H, W]
        tgt_tan = targets["tangent"].float()

        l2_tan = (pred_tan - tgt_tan) ** 2
        l2_tan = (l2_tan * mask.unsqueeze(1)).sum() / num_fg / 2.0
        losses["loss_tan"] = self.weights["tangent"] * l2_tan

        # 4. Width Loss (L1)
        pred_width = outputs["width"].float()
        tgt_width = targets["width"].float()
        l1_width = torch.abs(pred_width - tgt_width)
        l1_width = (l1_width * mask).sum() / num_fg
        losses["loss_width"] = self.weights["width"] * l1_width

        # 5. Offset Loss (L1)
        pred_off = outputs["offset"].float()
        tgt_off = targets["offset"].float()
        l1_off = torch.abs(pred_off - tgt_off)
        l1_off = (l1_off * mask.unsqueeze(1)).sum() / num_fg / 2.0
        losses["loss_off"] = self.weights["offset"] * l1_off

        # 6. Aux Losses (Skeleton only)
        if "aux_skeleton_16" in outputs:
            # Downsample target
            tgt_16 = F.interpolate(tgt_skel, size=16, mode="bilinear")
            tgt_16 = (tgt_16 > 0.5).float()  # Binarize

            aux_pred = outputs["aux_skeleton_16"]
            aux_loss = self._safe_bce(aux_pred, tgt_16)
            losses["loss_aux_16"] = self.weights["aux"] * aux_loss

        if "aux_skeleton_32" in outputs:
            # Downsample target
            tgt_32 = F.interpolate(tgt_skel, size=32, mode="bilinear")
            tgt_32 = (tgt_32 > 0.5).float()

            aux_pred = outputs["aux_skeleton_32"]
            aux_loss = self._safe_bce(aux_pred, tgt_32)
            losses["loss_aux_32"] = self.weights["aux"] * aux_loss

        losses["total"] = sum(losses.values())
        return losses
