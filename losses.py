import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Gaussian Focal Loss (CenterNet Style)
# =============================================================================


def gaussian_focal_loss(pred, target, alpha=2.0, beta=4.0):
    """
    Modified Focal Loss for Gaussian Heatmap (CenterNet Style)

    软标签交叉熵损失，适用于高斯热力图回归。
    与硬标签不同，高斯热力图的 GT 是连续值 [0, 1]。

    损失公式：
      - 正样本区域 (target > 0): -target * (1-p)^alpha * log(p)
      - 负样本区域 (target = 0): -(1-target)^beta * p^alpha * log(1-p)

    Args:
        pred: [B, C, H, W] 预测热力图 (sigmoid 输出)
        target: [B, C, H, W] 高斯热力图 GT (0~1, 中心为1)
        alpha: Focal Loss 指数 (default=2.0)
        beta: 负样本衰减指数 (default=4.0)

    Returns:
        loss: scalar
    """
    pred = pred.float().clamp(min=1e-6, max=1 - 1e-6)
    target = target.float()

    # 软标签交叉熵：每个像素都参与，权重 = target 值
    # 正样本区域：target 值越大，权重越高
    # 负样本区域：target 值越小，权重越低
    pos_weights = target

    # 负样本权重: (1 - target)^beta
    neg_weights = torch.pow(1 - target, beta)

    # Focal Loss with soft labels
    # 使用 target 作为软标签，而不是硬 0/1
    pos_loss = pos_weights * torch.pow(1 - pred, alpha) * torch.log(pred)

    # 负样本损失
    neg_loss = neg_weights * torch.pow(pred, alpha) * torch.log(1 - pred)

    # 总损失
    loss = -(pos_loss.sum() + neg_loss.sum())

    # 归一化：除以像素数量（稳定的归一化）
    num_pixels = pred.numel()
    loss = loss / num_pixels

    return loss



class DenseLoss(nn.Module):
    """
    Multi-task loss for InkTrace V5
    L = L_skel + L_keypoints + L_tan + L_width + L_offset

    Keypoints 是 2 通道:
      - Ch0: Topological nodes (endpoints, junctions) - MUST break
      - Ch1: Geometric anchors (sharp turns) - SHOULD break

    Note: Uses float32 casting for numerical stability under AMP.
    """

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            "skeleton": 10.0,
            "keypoints": 5.0,  # renamed from junction
            "tangent": 1.0,
            "width": 1.0,
            "offset": 1.0,
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
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")

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

        # 2. Keypoints Loss (Gaussian Focal Loss, 2 channels)
        #    Ch0: Topological (endpoints, junctions)
        #    Ch1: Geometric (sharp turns, inflections)
        #    GT: Dataset 已转换为高斯热力图 (datasets_v2.py)
        #    推理时: 使用 keypoint_nms() + extract_keypoints() 提取离散点
        pred_keys = outputs["keypoints"].float()  # [B, 2, H, W]
        tgt_keys = targets["keypoints"].float()  # [B, 2, H, W] Gaussian heatmap
        losses["loss_keys"] = self.weights["keypoints"] * gaussian_focal_loss(
            pred_keys, tgt_keys, alpha=2.0, beta=4.0
        )

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

        losses["total"] = sum(losses.values())
        return losses
