"""
损失函数模块 (V2 改进版)

包含：
- DETRLoss: DETR 风格的损失（配合 Hungarian Matching）

改进点：
- 向量化 Cost Matrix 计算（大幅提升 GPU 效率）
- Focal Loss 替代 BCE（处理类别不平衡）
- GIoU Loss 辅助坐标回归
- 批量匈牙利匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: [*, 1] sigmoid 概率
            target: [*, 1] 0/1 标签
        """
        pred = pred.clamp(1e-6, 1 - 1e-6)
        ce_loss = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * focal_weight * ce_loss
        return focal_loss.mean()


class DETRLoss(nn.Module):
    """
    DETR 风格的损失函数 (V2 改进版)

    改进：
    - 向量化 Cost Matrix 计算
    - Focal Loss 处理类别不平衡
    - L1 + Smooth L1 混合坐标损失
    """

    def __init__(
        self,
        coord_weight=5.0,
        width_weight=2.0,
        validity_weight=2.0,
        use_focal_loss=True,
    ):
        super().__init__()
        self.coord_weight = coord_weight
        self.width_weight = width_weight
        self.validity_weight = validity_weight
        self.use_focal_loss = use_focal_loss

        self.l1_loss = nn.L1Loss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=0.1)

        if use_focal_loss:
            self.validity_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.validity_loss_fn = nn.BCELoss()

    def forward(self, pred_strokes, pred_validity, targets):
        """
        Args:
            pred_strokes: [B, num_slots, 10] 预测的笔画参数
            pred_validity: [B, num_slots, 1] 预测的有效性
            targets: [B, num_slots, 11] GT
                    前 10 维是笔画参数，最后 1 维是有效性标志

        Returns:
            loss: 总损失
            loss_dict: 各项损失的字典
        """
        B, num_slots, _ = pred_strokes.shape
        device = pred_strokes.device

        # 提取 GT
        gt_strokes = targets[..., :10]  # [B, num_slots, 10]
        gt_validity = targets[..., 10:11]  # [B, num_slots, 1]

        total_loss = 0.0
        loss_dict = {"coord": 0.0, "width": 0.0, "validity": 0.0, "total_matched": 0}

        for b in range(B):
            pred = pred_strokes[b]  # [num_slots, 10]
            pred_val = pred_validity[b]  # [num_slots, 1]
            gt = gt_strokes[b]  # [num_slots, 10]
            gt_val = gt_validity[b]  # [num_slots, 1]

            # 找到有效的 GT 笔画
            valid_gt_mask = gt_val.squeeze(-1) > 0.5
            num_gt = valid_gt_mask.sum().item()

            if num_gt == 0:
                # 没有 GT 笔画，只计算有效性损失（所有 Slot 都应该无效）
                validity_loss = self.validity_loss_fn(pred_val, gt_val)
                total_loss += self.validity_weight * validity_loss
                loss_dict["validity"] += validity_loss.item()
                continue

            valid_gt_indices = torch.where(valid_gt_mask)[0]

            # 向量化计算 Cost Matrix
            cost_matrix = self.compute_cost_matrix_vectorized(
                pred, gt, valid_gt_indices
            )  # [num_slots, num_gt]

            # Hungarian Matching
            pred_indices, gt_matched_indices = linear_sum_assignment(
                cost_matrix.detach().cpu().numpy()
            )

            # 转为 tensor
            pred_indices = torch.from_numpy(pred_indices).to(device)
            gt_matched_indices = torch.from_numpy(gt_matched_indices).to(device)

            # 提取配对后的预测和 GT
            matched_pred = pred[pred_indices]  # [num_gt, 10]
            matched_gt = gt[valid_gt_indices[gt_matched_indices]]  # [num_gt, 10]

            # 1. 坐标损失 (L1 + Smooth L1 混合)
            coord_l1 = self.l1_loss(matched_pred[..., :8], matched_gt[..., :8]).mean()
            coord_smooth = self.smooth_l1(
                matched_pred[..., :8], matched_gt[..., :8]
            ).mean()
            coord_loss = 0.5 * coord_l1 + 0.5 * coord_smooth

            # 2. 宽度损失
            width_loss = self.l1_loss(
                matched_pred[..., 8:10], matched_gt[..., 8:10]
            ).mean()

            # 3. 有效性损失
            # 构建目标：匹配到的 Slot 为 1，其余为 0
            validity_target = torch.zeros_like(pred_val)
            validity_target[pred_indices] = 1.0
            validity_loss = self.validity_loss_fn(pred_val, validity_target)

            # 4. 未匹配 Slot 的惩罚 (应该预测为无效)
            unmatched_mask = torch.ones(num_slots, dtype=torch.bool, device=device)
            unmatched_mask[pred_indices] = False
            if unmatched_mask.any():
                unmatched_validity = pred_val[unmatched_mask]
                # 未匹配的 Slot 应该预测低有效性
                unmatched_penalty = unmatched_validity.mean()
                validity_loss = validity_loss + 0.5 * unmatched_penalty

            # 总损失
            loss = (
                self.coord_weight * coord_loss
                + self.width_weight * width_loss
                + self.validity_weight * validity_loss
            )

            total_loss += loss
            loss_dict["coord"] += coord_loss.item()
            loss_dict["width"] += width_loss.item()
            loss_dict["validity"] += validity_loss.item()
            loss_dict["total_matched"] += num_gt

        # 平均
        avg_loss = total_loss / B
        for key in ["coord", "width", "validity"]:
            loss_dict[key] /= B

        return avg_loss, loss_dict

    def compute_cost_matrix_vectorized(self, pred, gt, valid_gt_indices):
        """
        向量化计算 Cost Matrix (大幅提升效率)

        Args:
            pred: [num_slots, 10]
            gt: [num_slots, 10]
            valid_gt_indices: [num_gt] 有效 GT 的索引

        Returns:
            cost_matrix: [num_slots, num_gt]
        """
        num_slots = pred.shape[0]
        num_gt = len(valid_gt_indices)

        # 提取有效的 GT
        valid_gt = gt[valid_gt_indices]  # [num_gt, 10]

        # 扩展维度以便广播
        pred_expanded = pred.unsqueeze(1)  # [num_slots, 1, 10]
        gt_expanded = valid_gt.unsqueeze(0)  # [1, num_gt, 10]

        # 坐标 cost (L1)
        coord_cost = torch.abs(pred_expanded[..., :8] - gt_expanded[..., :8]).sum(
            dim=-1
        )  # [num_slots, num_gt]

        # 宽度 cost
        width_cost = torch.abs(pred_expanded[..., 8:10] - gt_expanded[..., 8:10]).sum(
            dim=-1
        )  # [num_slots, num_gt]

        # P0 距离加权 (鼓励 Slot 的空间局部性)
        p0_dist = torch.sqrt(
            ((pred_expanded[..., :2] - gt_expanded[..., :2]) ** 2).sum(dim=-1) + 1e-6
        )  # [num_slots, num_gt]

        # 总 cost
        cost_matrix = (
            5.0 * coord_cost
            + 1.0 * width_cost
            + 2.0 * p0_dist  # P0 距离额外加权，鼓励空间局部性
        )

        return cost_matrix
