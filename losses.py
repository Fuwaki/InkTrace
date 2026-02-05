import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Gaussian Focal Loss (CenterNet Style)
# =============================================================================


def gaussian_focal_loss(pred, target, alpha=2.0, beta=4.0):
    """
    Modified Focal Loss for Gaussian Heatmap (CenterNet Style)

    对于高斯热力图：
    - 正样本 (target=1): Focal Loss 惩罚难分类样本
    - 负样本 (target<1): 根据高斯值加权，离中心越近惩罚越轻

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

    # 正样本 mask: 高斯中心 (target == 1)
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()

    # 负样本权重: (1 - target)^beta，离中心越近权重越低
    neg_weights = torch.pow(1 - target, beta)

    # Focal Loss
    # 正样本: -((1-p)^alpha) * log(p)
    pos_loss = torch.pow(1 - pred, alpha) * torch.log(pred) * pos_mask

    # 负样本: -((1-t)^beta) * (p^alpha) * log(1-p)
    neg_loss = neg_weights * torch.pow(pred, alpha) * torch.log(1 - pred) * neg_mask

    # 归一化: 除以正样本数量
    num_pos = pos_mask.sum().clamp(min=1.0)
    loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos

    return loss


# =============================================================================
# NMS for Keypoint Extraction (Inference)
# =============================================================================


def keypoint_nms(heatmap, kernel_size=3, threshold=0.1):
    """
    Non-Maximum Suppression for heatmap keypoint extraction.

    Args:
        heatmap: [B, C, H, W] 预测热力图
        kernel_size: NMS 窗口大小
        threshold: 置信度阈值

    Returns:
        nms_heatmap: [B, C, H, W] 只保留局部最大值的热力图
    """
    pad = (kernel_size - 1) // 2

    # Max pooling 找局部最大值
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=pad)

    # 只保留等于局部最大值的点
    keep = (hmax == heatmap).float()

    # 应用阈值
    return heatmap * keep * (heatmap > threshold).float()


def extract_keypoints(heatmap, kernel_size=3, threshold=0.1, topk=100):
    """
    从热力图提取关键点坐标。

    Args:
        heatmap: [B, C, H, W] 预测热力图
        kernel_size: NMS 窗口大小
        threshold: 置信度阈值
        topk: 每个通道最多提取的点数

    Returns:
        keypoints: List[List[Tensor]] 每个 batch、每个通道的关键点
                   每个 Tensor 形状为 [N, 3] (y, x, score)
    """
    B, C, H, W = heatmap.shape

    # NMS
    nms_heat = keypoint_nms(heatmap, kernel_size, threshold)

    results = []
    for b in range(B):
        batch_kps = []
        for c in range(C):
            heat = nms_heat[b, c]  # [H, W]

            # 找非零点
            scores, indices = heat.flatten().topk(min(topk, H * W))
            valid = scores > threshold
            scores = scores[valid]
            indices = indices[valid]

            if len(scores) > 0:
                ys = (indices // W).float()
                xs = (indices % W).float()
                kps = torch.stack([ys, xs, scores], dim=1)  # [N, 3]
            else:
                kps = torch.zeros(0, 3, device=heatmap.device)

            batch_kps.append(kps)
        results.append(batch_kps)

    return results


# =============================================================================
# Gaussian Heatmap Rendering (for GT generation)
# =============================================================================


def render_gaussian_heatmap(keypoints_onehot, sigma=1.5):
    """
    将独热点 GT 转换为高斯热力图 GT。

    Args:
        keypoints_onehot: [B, C, H, W] 独热点 GT (0/1)
        sigma: 高斯标准差 (对于 64x64，建议 1.0~2.0)

    Returns:
        gaussian_heatmap: [B, C, H, W] 高斯热力图 (0~1)
    """
    B, C, H, W = keypoints_onehot.shape
    device = keypoints_onehot.device

    # 创建高斯核
    # 核大小 = 6*sigma (覆盖 99.7% 的分布)
    kernel_size = int(6 * sigma + 1) | 1  # 确保为奇数
    half = kernel_size // 2

    # 生成 2D 高斯核
    y = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    x = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    gaussian_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.max()  # 归一化到 0~1

    # 将核扩展为 conv2d 所需格式: [C_out, C_in, H, W]
    # 对每个通道独立卷积
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    # 对每个通道进行卷积
    result = []
    for c in range(C):
        channel = keypoints_onehot[:, c : c + 1, :, :]  # [B, 1, H, W]
        # 使用 padding='same' 保持尺寸
        blurred = F.conv2d(channel, gaussian_kernel, padding=half)
        result.append(blurred)

    gaussian_heatmap = torch.cat(result, dim=1)

    # Clamp to [0, 1]
    return gaussian_heatmap.clamp(0, 1)


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
        #    GT should be Gaussian heatmaps, not one-hot
        pred_keys = outputs["keypoints"].float()  # [B, 2, H, W]
        tgt_keys = targets["keypoints"].float()  # [B, 2, H, W]
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
