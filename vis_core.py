#!/usr/bin/env python3
"""
核心可视化渲染层 (Core Rendering Layer)

职责：
- 纯粹的绘图工具库
- 不依赖 PyTorch Lightning 或具体模型类
- 仅依赖 numpy, matplotlib, PIL, torch

功能：
- 张量到 numpy 的转换
- 切向场 HSV 可视化
- 批量网格对比图生成
- 单样本详细可视化
"""

import io
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import torch
import torchmetrics.functional as tmF


# =============================================================================
# 工具函数
# =============================================================================


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    鲁棒地将 PyTorch 张量转换为 numpy 数组

    处理：
    - GPU/CPU 设备迁移
    - 梯度 detach
    - 维度压缩 ([1, C, H, W] -> [C, H, W] -> [H, W])

    Args:
        tensor: PyTorch 张量，任意形状

    Returns:
        numpy 数组
    """
    # Detach and move to CPU
    arr = tensor.detach().cpu()

    # Squeeze batch/channel dimensions if present
    # Handle: [B, C, H, W] or [C, H, W] or [H, W]
    while arr.dim() > 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)

    return arr.numpy()


def visualize_tangent(tangent_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    将切向场转换为 HSV RGB 图像

    切向场格式：[2, H, W]
    - channel 0: cos(2θ)
    - channel 1: sin(2θ)

    Args:
        tangent_map: [2, H, W] 切向场数组
        mask: [H, W] 骨架掩码 (可选，用于遮掩背景)

    Returns:
        [H, W, 3] RGB 图像
    """
    # 计算角度 (0-360 度)
    angle = np.degrees(np.arctan2(tangent_map[1], tangent_map[0]))
    angle[angle < 0] += 360

    # 构建 HSV
    # H: 角度 (0-1)
    # S: 饱和度 (全为1)
    # V: 亮度 (mask 区域为1，否则为0)
    hsv = np.stack(
        [
            angle / 360.0,
            np.ones_like(angle),
            np.where(mask > 0.5, 1.0, 0.0),
        ],
        axis=-1,
    )

    return hsv_to_rgb(hsv)


def compute_metrics(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    使用 torchmetrics 计算评估指标

    支持分布式训练，自动处理多 GPU 场景

    Args:
        outputs: 模型输出字典
        targets: Ground Truth 字典

    Returns:
        指标字典 (skel_iou, skel_precision, skel_recall, skel_f1,
                  kp_topo_recall, kp_geo_recall)
    """
    # ========== 骨架分割指标 ==========
    pred_skel = (outputs["skeleton"] > 0.5).flatten()
    tgt_skel = (targets["skeleton"] > 0.5).flatten()

    # 使用 torchmetrics 计算二分类指标
    # stat_scores 返回: tp, fp, tn, fn, support
    skel_stats = tmF.stat_scores(
        pred_skel,
        tgt_skel.int(),
        task='binary',
    )

    tp, fp, tn, fn = skel_stats[:4]

    # IoU (Jaccard Index) - 手动计算保证兼容性
    # Jaccard = TP / (TP + FP + FN)
    skel_iou = tp / (tp + fp + fn + 1e-6)

    # Precision, Recall, F1
    skel_precision = tmF.precision(
        pred_skel,
        tgt_skel.int(),
        task='binary',
    )
    skel_recall = tmF.recall(
        pred_skel,
        tgt_skel.int(),
        task='binary',
    )
    skel_f1 = tmF.f1_score(
        pred_skel,
        tgt_skel.int(),
        task='binary',
    )

    # ========== 关键点指标 ==========
    # Topo keypoints
    pred_kp_topo = (outputs["keypoints"][:, 0:1] > 0.3).flatten()
    tgt_kp_topo = (targets["keypoints"][:, 0:1] > 0.3).flatten()

    kp_topo_recall = tmF.recall(
        pred_kp_topo,
        tgt_kp_topo.int(),
        task='binary',
    )

    # Geo keypoints
    pred_kp_geo = (outputs["keypoints"][:, 1:2] > 0.3).flatten()
    tgt_kp_geo = (targets["keypoints"][:, 1:2] > 0.3).flatten()

    kp_geo_recall = tmF.recall(
        pred_kp_geo,
        tgt_kp_geo.int(),
        task='binary',
    )

    return {
        "skel_iou": skel_iou.item(),
        "skel_precision": skel_precision.item(),
        "skel_recall": skel_recall.item(),
        "skel_f1": skel_f1.item(),
        "kp_topo_recall": kp_topo_recall.item(),
        "kp_geo_recall": kp_geo_recall.item(),
        # 额外的统计信息
        "skel_tp": tp.item(),
        "skel_fp": fp.item(),
        "skel_fn": fn.item(),
    }


# =============================================================================
# 核心渲染函数
# =============================================================================


def create_grid_image(
    imgs: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    num_samples: int = 4,
    include_overlay: bool = True,
) -> np.ndarray:
    """
    创建批量对比可视化网格

    布局：每行一个样本，每列一种可视化
    列：[Input, GT Skel, Pred Skel, GT Tan, Pred Tan, Overlay]

    Args:
        imgs: [B, 1, H, W] 输入图像
        outputs: 模型输出字典
        targets: Ground Truth 字典
        num_samples: 可视化样本数
        include_overlay: 是否包含叠加图

    Returns:
        [H, W, 3] RGB 图像 (numpy array)
    """
    num_samples = min(num_samples, imgs.shape[0])
    cols = 6 if include_overlay else 5
    rows = num_samples

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        # 提取单个样本数据
        img = to_numpy(imgs[i])  # [H, W]
        pred_skel = to_numpy(outputs["skeleton"][i, 0])  # [H, W]
        tgt_skel = to_numpy(targets["skeleton"][i, 0])
        pred_tan = to_numpy(outputs["tangent"][i])  # [2, H, W]
        tgt_tan = to_numpy(targets["tangent"][i])

        # 1. Input
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("Input" if i == 0 else "")
        axes[i, 0].axis("off")

        # 2. GT Skeleton
        axes[i, 1].imshow(tgt_skel, cmap="gray")
        axes[i, 1].set_title("GT Skel" if i == 0 else "")
        axes[i, 1].axis("off")

        # 3. Pred Skeleton
        axes[i, 2].imshow(pred_skel, cmap="gray")
        axes[i, 2].set_title("Pred Skel" if i == 0 else "")
        axes[i, 2].axis("off")

        # 4. GT Tangent
        axes[i, 3].imshow(visualize_tangent(tgt_tan, tgt_skel))
        axes[i, 3].set_title("GT Tan" if i == 0 else "")
        axes[i, 3].axis("off")

        # 5. Pred Tangent
        axes[i, 4].imshow(visualize_tangent(pred_tan, pred_skel))
        axes[i, 4].set_title("Pred Tan" if i == 0 else "")
        axes[i, 4].axis("off")

        # 6. Overlay (可选)
        if include_overlay:
            overlay = np.stack([img, img, img], axis=-1)
            overlay[pred_skel > 0.5] = [1, 0, 0]  # Red: 预测
            overlay[tgt_skel > 0.5] = [0, 1, 0]  # Green: GT
            axes[i, 5].imshow(overlay)
            axes[i, 5].set_title("Overlay (R=Pred,G=GT)" if i == 0 else "")
            axes[i, 5].axis("off")

    plt.tight_layout()

    # 转换为 numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_arr = np.array(Image.open(buf))[..., :3]  # Remove alpha channel
    plt.close(fig)

    return img_arr


def visualize_sample(
    img: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
) -> np.ndarray:
    """
    单样本详细可视化 (4x5 网格)

    布局：
    - Row 1: Input, GT Skeleton, Pred Skeleton, Overlay GT, Overlay Pred
    - Row 2: Keypoints Topo/GT/Pred, Geo/GT/Pred, Combined
    - Row 3: Tangent GT/Pred, Width GT/Pred/Diff
    - Row 4: Offset X/Y GT/Pred, Magnitude Diff

    Args:
        img: [1, 1, H, W] 或 [1, H, W] 或 [H, W]
        pred: 模型预测字典
        target: Ground Truth 字典

    Returns:
        [H, W, 3] RGB 图像 (numpy array)
    """
    # 处理输入图像
    img_np = to_numpy(img)

    # Skeleton
    pred_skel = to_numpy(pred["skeleton"][0, 0])
    tgt_skel = to_numpy(target["skeleton"][0, 0])

    # Tangent
    pred_tan = to_numpy(pred["tangent"][0])  # [2, H, W]
    tgt_tan = to_numpy(target["tangent"][0])

    # Keypoints: [2, H, W] - ch0=topo, ch1=geo
    pred_kp = to_numpy(pred["keypoints"][0])
    tgt_kp = to_numpy(target["keypoints"][0])

    # Width: [H, W]
    pred_width = to_numpy(pred["width"][0, 0])
    tgt_width = to_numpy(target["width"][0, 0])

    # Offset: [2, H, W]
    pred_offset = to_numpy(pred["offset"][0])
    tgt_offset = to_numpy(target["offset"][0])

    # 创建 4x5 网格
    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))

    # Row 1: Input, Skeleton GT, Skeleton Pred, Overlay GT, Overlay Pred
    axes[0, 0].imshow(img_np, cmap="gray")
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(tgt_skel, cmap="gray")
    axes[0, 1].set_title("GT Skeleton")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_skel, cmap="gray")
    axes[0, 2].set_title("Pred Skeleton")
    axes[0, 2].axis("off")

    # Overlay GT
    overlay_gt = np.stack([img_np, img_np, img_np], axis=-1)
    overlay_gt[tgt_skel > 0.5] = [0, 1, 0]
    axes[0, 3].imshow(overlay_gt)
    axes[0, 3].set_title("Overlay GT")
    axes[0, 3].axis("off")

    # Overlay Pred
    overlay_pred = np.stack([img_np, img_np, img_np], axis=-1)
    overlay_pred[pred_skel > 0.5] = [1, 0, 0]
    axes[0, 4].imshow(overlay_pred)
    axes[0, 4].set_title("Overlay Pred")
    axes[0, 4].axis("off")

    # Row 2: Keypoints
    axes[1, 0].imshow(tgt_kp[0], cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title("GT KP Topo")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_kp[0], cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("Pred KP Topo")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(tgt_kp[1], cmap="magma", vmin=0, vmax=1)
    axes[1, 2].set_title("GT KP Geo")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(pred_kp[1], cmap="magma", vmin=0, vmax=1)
    axes[1, 3].set_title("Pred KP Geo")
    axes[1, 3].axis("off")

    # Combined keypoints overlay
    kp_overlay = np.zeros((img_np.shape[0], img_np.shape[1], 3))
    kp_overlay[..., 0] = np.maximum(pred_kp[0], pred_kp[1])  # Red: pred
    kp_overlay[..., 1] = np.maximum(tgt_kp[0], tgt_kp[1])  # Green: GT
    axes[1, 4].imshow(kp_overlay)
    axes[1, 4].set_title("KP Overlay (R=Pred,G=GT)")
    axes[1, 4].axis("off")

    # Row 3: Tangent & Width
    axes[2, 0].imshow(visualize_tangent(tgt_tan, tgt_skel))
    axes[2, 0].set_title("GT Tangent")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(visualize_tangent(pred_tan, pred_skel))
    axes[2, 1].set_title("Pred Tangent")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(tgt_width * tgt_skel, cmap="viridis")
    axes[2, 2].set_title("GT Width")
    axes[2, 2].axis("off")

    axes[2, 3].imshow(pred_width * pred_skel, cmap="viridis")
    axes[2, 3].set_title("Pred Width")
    axes[2, 3].axis("off")

    # Width difference
    width_diff = np.abs(pred_width - tgt_width) * np.maximum(pred_skel, tgt_skel)
    axes[2, 4].imshow(width_diff, cmap="hot")
    axes[2, 4].set_title("Width Diff")
    axes[2, 4].axis("off")

    # Row 4: Offset
    axes[3, 0].imshow(tgt_offset[0] * tgt_skel, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[3, 0].set_title("GT Offset X")
    axes[3, 0].axis("off")

    axes[3, 1].imshow(pred_offset[0] * pred_skel, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[3, 1].set_title("Pred Offset X")
    axes[3, 1].axis("off")

    axes[3, 2].imshow(tgt_offset[1] * tgt_skel, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[3, 2].set_title("GT Offset Y")
    axes[3, 2].axis("off")

    axes[3, 3].imshow(pred_offset[1] * pred_skel, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[3, 3].set_title("Pred Offset Y")
    axes[3, 3].axis("off")

    # Offset magnitude difference
    tgt_off_mag = np.sqrt(tgt_offset[0] ** 2 + tgt_offset[1] ** 2) * tgt_skel
    pred_off_mag = np.sqrt(pred_offset[0] ** 2 + pred_offset[1] ** 2) * pred_skel
    off_diff = np.abs(pred_off_mag - tgt_off_mag)
    axes[3, 4].imshow(off_diff, cmap="hot")
    axes[3, 4].set_title("Offset Mag Diff")
    axes[3, 4].axis("off")

    plt.tight_layout()

    # 转换为 numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img_arr = np.array(Image.open(buf))[..., :3]
    plt.close(fig)

    return img_arr
