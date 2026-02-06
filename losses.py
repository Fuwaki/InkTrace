import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Gaussian Focal Loss (CenterNet Style)
# =============================================================================


def gaussian_focal_loss(pred, target, alpha=2.0, beta=4.0, reduction="mean"):
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
        alpha: Focal Loss 指数 (default=2.0)，控制难样本权重
        beta: 负样本衰减指数 (default=4.0)，控制远离中心点的负样本权重
        reduction: "mean" 或 "sum"

    Returns:
        loss: scalar
    """
    pred = pred.float().clamp(min=1e-6, max=1 - 1e-6)
    target = target.float()

    # 正样本权重：target 值越大，权重越高
    pos_weights = target

    # 负样本权重: (1 - target)^beta
    # beta 越大，远离中心点的负样本权重越小（避免过度惩罚背景）
    neg_weights = torch.pow(1 - target, beta)

    # Focal Loss with soft labels
    # 正样本：难样本（预测低）权重高
    pos_loss = pos_weights * torch.pow(1 - pred, alpha) * torch.log(pred)

    # 负样本：难样本（预测高）权重高
    neg_loss = neg_weights * torch.pow(pred, alpha) * torch.log(1 - pred)

    # 总损失
    loss = -(pos_loss + neg_loss)

    if reduction == "mean":
        # 按正样本数量归一化（比像素数更稳定）
        num_pos = pos_weights.sum().clamp(min=1.0)
        return loss.sum() / num_pos
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


# =============================================================================
# Dice Loss (分割任务)
# =============================================================================


def dice_loss(pred, target, smooth=1.0):
    """
    Dice Loss for binary segmentation

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred: [B, 1, H, W] 预测概率图 (sigmoid 输出)
        target: [B, 1, H, W] 二值 GT
        smooth: 平滑因子，避免除零

    Returns:
        loss: 1 - Dice
    """
    pred = pred.float().contiguous()
    target = target.float().contiguous()

    # 在 H, W 维度上计算 intersection
    intersection = (pred * target).sum(dim=[2, 3])

    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (
        pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + smooth
    )

    return (1 - dice).mean()


class DenseLoss(nn.Module):
    """
    Multi-task loss for InkTrace V5
    L = L_skel + L_keypoints + L_tan + L_width + L_offset

    Keypoints 是 2 通道:
      - Ch0: Topological nodes (endpoints, junctions) - MUST break
      - Ch1: Geometric anchors (sharp turns) - SHOULD break

    Note: Uses float32 casting for numerical stability under AMP.

    改进点：
    1. 使用新的 dice_loss 函数
    2. 更好的数值稳定性
    3. 支持动态权重调整
    """

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            "skeleton": 10.0,
            "keypoints": 5.0,
            "tangent": 2.0,  # 切向场对拟合很重要
            "width": 1.0,
            "offset": 1.0,
        }

    def _safe_bce(self, pred, target):
        """
        AMP-safe BCE loss using F.binary_cross_entropy_with_logits.
        We convert sigmoid outputs back to logits for numerical stability.
        """
        pred = pred.float().clamp(min=1e-6, max=1 - 1e-6)
        target = target.float()
        # Convert probability to logits: logit = log(p / (1-p))
        logits = torch.log(pred / (1 - pred))
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")

    def forward(self, outputs, targets):
        """
        计算多任务损失

        Args:
            outputs: dict with model predictions
                - skeleton: [B, 1, H, W]
                - keypoints: [B, 2, H, W]
                - tangent: [B, 2, H, W]
                - width: [B, 1, H, W]
                - offset: [B, 2, H, W]
            targets: dict with ground truth

        Returns:
            losses: dict with individual losses and total
        """
        losses = {}

        # 1. Skeleton Loss (BCE + Dice)
        # Skeleton 是核心任务，使用两种 loss 确保精度
        pred_skel = outputs["skeleton"]
        tgt_skel = targets["skeleton"]

        bce_skel = self._safe_bce(pred_skel, tgt_skel)
        dice_skel = dice_loss(pred_skel, tgt_skel)
        losses["loss_skel"] = self.weights["skeleton"] * (bce_skel + dice_skel)

        # 2. Keypoints Loss (Gaussian Focal Loss, 2 channels)
        # GT: 高斯热力图 (由 datasets_v2.py 转换)
        pred_keys = outputs["keypoints"].float()
        tgt_keys = targets["keypoints"].float()
        losses["loss_keys"] = self.weights["keypoints"] * gaussian_focal_loss(
            pred_keys, tgt_keys, alpha=2.0, beta=4.0
        )

        # Masking for regression tasks
        # 只在 skeleton 区域计算回归 loss
        mask = (tgt_skel > 0.5).float()
        num_fg = mask.sum().clamp(min=1.0)

        # 3. Tangent Loss (L2 on masked region)
        # 使用 L2 而非 cosine similarity，因为我们需要精确的角度
        pred_tan = outputs["tangent"].float()
        tgt_tan = targets["tangent"].float()

        l2_tan = (pred_tan - tgt_tan) ** 2
        l2_tan = (l2_tan * mask.unsqueeze(1)).sum() / num_fg / 2.0
        losses["loss_tan"] = self.weights["tangent"] * l2_tan

        # 4. Width Loss (Smooth L1)
        # Smooth L1 比纯 L1 更稳定，对异常值不敏感
        pred_width = outputs["width"].float()
        tgt_width = targets["width"].float()
        smooth_l1_width = F.smooth_l1_loss(
            pred_width * mask,
            tgt_width * mask,
            reduction="sum",
            beta=0.5,  # 比默认的 1.0 更早切换到 L1
        )
        losses["loss_width"] = self.weights["width"] * smooth_l1_width / num_fg

        # 5. Offset Loss (Smooth L1)
        pred_off = outputs["offset"].float()
        tgt_off = targets["offset"].float()
        smooth_l1_off = F.smooth_l1_loss(
            pred_off * mask.unsqueeze(1),
            tgt_off * mask.unsqueeze(1),
            reduction="sum",
            beta=0.5,
        )
        losses["loss_off"] = self.weights["offset"] * smooth_l1_off / num_fg / 2.0

        losses["total"] = sum(losses.values())
        return losses
