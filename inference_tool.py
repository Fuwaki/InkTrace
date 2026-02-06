"""
推理后处理工具 (Inference Post-processing)

包含从模型输出到最终结果的转换：
- 热力图 → NMS → 离散关键点
- 骨架图 → 矢量化路径
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


# =============================================================================
# Keypoint Extraction (热力图 → 离散点)
# =============================================================================


def keypoint_nms(heatmap: torch.Tensor, kernel_size: int = 3, threshold: float = 0.1):
    """
    Non-Maximum Suppression for heatmap keypoint extraction.

    抑制非局部最大值，保留每个局部邻域的最大响应点。

    Args:
        heatmap: [B, C, H, W] 预测热力图
        kernel_size: NMS 窗口大小 (通常 3×3)
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


def extract_keypoints(
    heatmap: torch.Tensor,
    kernel_size: int = 3,
    threshold: float = 0.1,
    topk: int = 100,
) -> List[List[torch.Tensor]]:
    """
    从热力图提取关键点坐标。

    流程：
      1. NMS 去重
      2. 阈值过滤
      3. Top-K 选择
      4. 返回坐标 (y, x, score)

    Args:
        heatmap: [B, C, H, W] 预测热力图（Sigmoid 输出）
        kernel_size: NMS 窗口大小
        threshold: 置信度阈值
        topk: 每个通道最多提取的点数

    Returns:
        keypoints: List[List[Tensor]] 每个 batch、每个通道的关键点
                   每个 Tensor 形状为 [N, 3] = (y, x, score)
                   keypoints[b][c] → 第 b 个样本的第 c 个通道的关键点
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


def extract_keypoints_batch(
    heatmap: torch.Tensor,
    kernel_size: int = 3,
    threshold: float = 0.1,
    topk: int = 100,
) -> torch.Tensor:
    """
    批量提取关键点（向量化版本，返回固定大小 tensor）

    适合需要批量处理和堆叠结果的场景。

    Args:
        heatmap: [B, C, H, W] 预测热力图
        kernel_size: NMS 窗口大小
        threshold: 置信度阈值
        topk: 每个通道最多提取的点数

    Returns:
        keypoints: [B, C, topk, 3] 固定大小的关键点 tensor
                   填充值用 (0, 0, 0)
    """
    B, C, H, W = heatmap.shape

    # NMS
    nms_heat = keypoint_nms(heatmap, kernel_size, threshold)

    # 扁平化
    nms_flat = nms_heat.view(B, C, -1)  # [B, C, H*W]

    # Top-K
    topk_scores, topk_indices = nms_flat.topk(topk, dim=-1)  # [B, C, topk]

    # 过滤低置信度
    valid_mask = topk_scores > threshold

    # 计算坐标
    ys = (topk_indices // W).float()  # [B, C, topk]
    xs = (topk_indices % W).float()

    # 堆叠结果
    keypoints = torch.stack([ys, xs, topk_scores], dim=-1)  # [B, C, topk, 3]

    # 无效位置置零
    keypoints[~valid_mask] = 0.0

    return keypoints


