"""
损失函数模块

包含：
- DETRLoss: DETR 风格的损失（配合 Hungarian Matching）
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class DETRLoss(nn.Module):
    """
    DETR 风格的损失函数

    流程：
    1. Hungarian Matching：找到最优的 Slot-GT 配对
    2. 计算配对后的损失
    """

    def __init__(self, coord_weight=5.0, width_weight=1.0, validity_weight=1.0):
        super().__init__()
        self.coord_weight = coord_weight
        self.width_weight = width_weight
        self.validity_weight = validity_weight

        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()

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

        # 提取 GT
        gt_strokes = targets[..., :10]  # [B, num_slots, 10]
        gt_validity = targets[..., 10:11]  # [B, num_slots, 1]

        total_loss = 0.0
        loss_dict = {
            'coord': 0.0,
            'width': 0.0,
            'validity': 0.0,
            'matching_cost': 0.0
        }

        for b in range(B):
            pred = pred_strokes[b]  # [num_slots, 10]
            pred_val = pred_validity[b]  # [num_slots, 1]
            gt = gt_strokes[b]  # [num_slots, 10]
            gt_val = gt_validity[b]  # [num_slots, 1]

            # 找到有效的 GT 笔画
            valid_gt_indices = torch.where(gt_val.squeeze(-1) > 0.5)[0]
            num_gt = len(valid_gt_indices)

            if num_gt == 0:
                # 没有 GT 笔画，只计算有效性损失
                validity_loss = self.bce_loss(pred_val, gt_val)
                total_loss += validity_loss
                loss_dict['validity'] += validity_loss.item()
                continue

            # 计算 Cost Matrix
            cost_matrix = self.compute_cost_matrix(
                pred, gt, pred_val, gt_val, valid_gt_indices
            )  # [num_slots, num_gt]

            # Hungarian Matching
            pred_indices, gt_indices = linear_sum_assignment(
                cost_matrix.detach().cpu().numpy()
            )

            # 转为 tensor
            pred_indices = torch.from_numpy(pred_indices).to(pred.device)
            gt_indices = torch.from_numpy(gt_indices).to(gt.device)

            # 提取配对后的预测和 GT
            matched_pred = pred[pred_indices]
            matched_gt = gt[gt_indices]
            matched_pred_val = pred_val[pred_indices]
            matched_gt_val = gt_val[gt_indices]

            # 1. 坐标损失
            coord_loss = self.l1_loss(matched_pred[..., :8], matched_gt[..., :8])

            # 2. 宽度损失
            width_loss = self.l1_loss(matched_pred[..., 8:10], matched_gt[..., 8:10])

            # 3. 有效性损失
            global_validity_loss = self.bce_loss(pred_val, gt_val)

            # 总损失
            loss = (
                self.coord_weight * coord_loss +
                self.width_weight * width_loss +
                self.validity_weight * global_validity_loss
            )

            total_loss += loss
            loss_dict['coord'] += coord_loss.item()
            loss_dict['width'] += width_loss.item()
            loss_dict['validity'] += global_validity_loss.item()
            loss_dict['matching_cost'] += cost_matrix[pred_indices, gt_indices].sum().item()

        # 平均
        avg_loss = total_loss / B
        for key in loss_dict:
            loss_dict[key] /= B

        return avg_loss, loss_dict

    def compute_cost_matrix(self, pred, gt, pred_val, gt_val, valid_gt_indices):
        """
        计算预测和 GT 之间的 cost matrix

        Cost = 5 * L1(坐标) + 1 * L1(宽度)
        """
        num_slots = pred.shape[0]
        num_gt = len(valid_gt_indices)

        # 提取有效的 GT
        valid_gt = gt[valid_gt_indices]

        # 初始化 cost matrix
        cost_matrix = torch.zeros(num_slots, num_gt, device=pred.device)

        for i in range(num_slots):
            for j in range(num_gt):
                # 坐标 cost
                coord_cost = torch.abs(pred[i, :8] - valid_gt[j, :8]).sum()

                # 宽度 cost
                width_cost = torch.abs(pred[i, 8:10] - valid_gt[j, 8:10]).sum()

                # 总 cost
                cost_matrix[i, j] = (
                    5.0 * coord_cost +
                    1.0 * width_cost
                )

        return cost_matrix
