"""
Dense GT Maps 可视化工具 v2

可视化训练数据集（使用 datasets_v2.py）生成的 Dense GT Maps:
- skeleton: 骨架图
- keypoints: 关键点图 (2通道: 拓扑节点 + 几何锚点) - 高斯热力图格式
- tangent: 切向场 (HSV 颜色编码)
- width: 宽度图
- offset: 亚像素偏移

默认启用从高斯热力图提取关键点并标记。

用法:
    python visualize_dense_gt_v2.py                      # 可视化默认阶段
    python visualize_dense_gt_v2.py --stage 5            # 可视化指定阶段
    python visualize_dense_gt_v2.py --kp-threshold 0.2   # 调整关键点阈值
    python visualize_dense_gt_v2.py --no-extract-kps     # 禁用关键点标记
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import colorsys
from datasets_v2 import DenseInkTraceDataset
from inference_tool import extract_keypoints


def tangent_to_rgb(cos2t: np.ndarray, sin2t: np.ndarray, skeleton: np.ndarray = None):
    """
    将双倍角切向场转换为 RGB 颜色

    Args:
        cos2t: cos(2θ) 分量 [H, W]
        sin2t: sin(2θ) 分量 [H, W]
        skeleton: 骨架 mask [H, W]，用于遮罩

    Returns:
        RGB 图像 [H, W, 3]
    """
    # 从双倍角恢复角度 (0, π)
    angle = np.arctan2(sin2t, cos2t) / 2
    angle = np.mod(angle, np.pi)

    # HSV: Hue = angle / π, Saturation = 1, Value = 1
    h = angle / np.pi

    # 创建 RGB
    shape = cos2t.shape
    rgb = np.zeros((*shape, 3))

    for y in range(shape[0]):
        for x in range(shape[1]):
            rgb[y, x] = colorsys.hsv_to_rgb(h[y, x], 1.0, 1.0)

    # 如果提供骨架 mask，只显示骨架上的颜色
    if skeleton is not None:
        mask = skeleton < 0.5
        rgb[mask] = [0.9, 0.9, 0.9]  # 背景色

    return rgb


def visualize_dense_batch(
    img: torch.Tensor,
    targets: dict,
    save_path: str = None,
    extract_keypoints_flag: bool = True,
    kp_threshold: float = 0.1,
    kp_topk: int = 100,
):
    """
    可视化单个样本的 Dense GT Maps

    Args:
        img: 图像 tensor [1, H, W]
        targets: 包含 Dense GT Maps 的字典
        save_path: 保存路径 (None 则显示)
        extract_keypoints_flag: 是否标记提取的关键点
        kp_threshold: 关键点提取的置信度阈值（只保留热力图值>该阈值的点）
        kp_topk: 每个通道最多提取的点数
    """
    # 转换为 numpy 数组
    img = img.squeeze(0).cpu().numpy()
    skeleton = targets["skeleton"].squeeze(0).cpu().numpy()
    keypoints = targets["keypoints"].cpu().numpy()  # [2, H, W]
    tangent = targets["tangent"].cpu().numpy()  # [2, H, W]
    width = targets["width"].squeeze(0).cpu().numpy()
    offset = targets["offset"].cpu().numpy()  # [2, H, W]

    # 分离 keypoints 两个通道
    topo_keypoints = keypoints[0]  # 拓扑节点
    geom_keypoints = keypoints[1]  # 几何锚点

    # 提取关键点（如果启用）
    topo_kps = None
    geom_kps = None
    if extract_keypoints_flag:
        heatmap_batch = targets["keypoints"].unsqueeze(0)  # [1, 2, H, W]
        keypoints_list = extract_keypoints(
            heatmap_batch, kernel_size=3, threshold=kp_threshold, topk=kp_topk
        )
        topo_kps = keypoints_list[0][0].cpu().numpy()  # [N, 3] = (y, x, score)
        geom_kps = keypoints_list[0][1].cpu().numpy()  # [N, 3]

    # 创建图形 (3x3)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Image (Rendered)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(skeleton, cmap="gray")
    axes[0, 1].set_title("Skeleton")
    axes[0, 1].axis("off")

    # 组合 keypoints 显示
    kp_rgb = np.zeros((*skeleton.shape, 3))
    kp_rgb[..., 0] = topo_keypoints  # Red: 拓扑节点
    kp_rgb[..., 1] = geom_keypoints  # Green: 几何锚点
    kp_rgb[..., 2] = 0.2 * skeleton  # Blue: 骨架轮廓
    axes[0, 2].imshow(kp_rgb.clip(0, 1))

    # 标记提取的关键点
    if topo_kps is not None and len(topo_kps) > 0:
        for y, x, score in topo_kps:
            circle = plt.Circle((x, y), radius=2.5, color="red", alpha=0.6, fill=False, linewidth=2)
            axes[0, 2].add_patch(circle)
    if geom_kps is not None and len(geom_kps) > 0:
        for y, x, score in geom_kps:
            circle = plt.Circle((x, y), radius=2.5, color="lime", alpha=0.6, fill=False, linewidth=2)
            axes[0, 2].add_patch(circle)

    axes[0, 2].set_title("Keypoints (R=Topo, G=Geom)")
    axes[0, 2].axis("off")

    # Tangent as HSV
    tangent_rgb = tangent_to_rgb(tangent[0], tangent[1], skeleton)
    axes[0, 3].imshow(tangent_rgb)
    axes[0, 3].set_title("Tangent (HSV)")
    axes[0, 3].axis("off")

    # Row 2
    # 分别显示两个 keypoints 通道
    axes[1, 0].imshow(topo_keypoints, cmap="Reds", vmin=0, vmax=1)
    if topo_kps is not None and len(topo_kps) > 0:
        for y, x, score in topo_kps:
            circle = plt.Circle((x, y), radius=2.5, color="red", alpha=0.6, fill=False, linewidth=2)
            axes[1, 0].add_patch(circle)
    axes[1, 0].set_title(f"Topo Keypoints (sum={topo_keypoints.sum():.1f})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(geom_keypoints, cmap="Greens", vmin=0, vmax=1)
    if geom_kps is not None and len(geom_kps) > 0:
        for y, x, score in geom_kps:
            circle = plt.Circle((x, y), radius=2.5, color="lime", alpha=0.6, fill=False, linewidth=2)
            axes[1, 1].add_patch(circle)
    axes[1, 1].set_title(f"Geom Keypoints (sum={geom_keypoints.sum():.1f})")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(width, cmap="viridis")
    axes[1, 2].set_title(f"Width (max={width.max():.2f})")
    axes[1, 2].axis("off")

    # Offset magnitude
    offset_mag = np.sqrt(offset[0] ** 2 + offset[1] ** 2)
    axes[1, 3].imshow(offset_mag * skeleton, cmap="hot")
    axes[1, 3].set_title(f"Offset Magnitude")
    axes[1, 3].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def verify_tangent_consistency(targets: dict):
    """验证切向场的一致性（双倍角表示）"""
    tangent = targets["tangent"].cpu().numpy()
    skeleton = targets["skeleton"].squeeze(0).cpu().numpy()

    cos2t = tangent[0]
    sin2t = tangent[1]

    # 只检查骨架上的点
    mask = skeleton > 0.5
    cos2t_masked = cos2t[mask]
    sin2t_masked = sin2t[mask]

    # 检查单位圆约束
    magnitude = np.sqrt(cos2t_masked**2 + sin2t_masked**2)

    print(f"Tangent Verification:")
    print(f"  Points on skeleton: {mask.sum()}")
    print(
        f"  Magnitude (should be ~1): min={magnitude.min():.3f}, max={magnitude.max():.3f}, mean={magnitude.mean():.3f}"
    )

    # 恢复角度
    angle = np.arctan2(sin2t_masked, cos2t_masked) / 2
    angle_deg = np.rad2deg(angle)
    print(f"  Angle range: {angle_deg.min():.1f}° to {angle_deg.max():.1f}°")


def main():
    parser = argparse.ArgumentParser(description="Dense GT Maps 可视化工具 v2")
    parser.add_argument("--stage", type=int, default=0, help="训练阶段 (0-9)")
    parser.add_argument("--img-size", type=int, default=64, help="图像尺寸")
    parser.add_argument("--save", type=str, default=None, help="保存目录")
    parser.add_argument("--verify", action="store_true", help="验证切向场一致性")
    parser.add_argument("--keypoint-sigma", type=float, default=1.5, help="高斯热力图标准差")
    parser.add_argument(
        "--no-extract-kps",
        action="store_true",
        help="禁用关键点提取标记（默认启用）",
    )
    parser.add_argument(
        "--kp-threshold",
        type=float,
        default=0.1,
        help="关键点提取的置信度阈值（只保留热力图值>该阈值的点）",
    )
    parser.add_argument("--kp-topk", type=int, default=100, help="每个通道最多提取的点数")
    args = parser.parse_args()

    # 使用 dataset 生成单个阶段的数据
    dataset = DenseInkTraceDataset(
        img_size=args.img_size,
        batch_size=4,
        epoch_length=4,
        curriculum_stage=args.stage,
        keypoint_sigma=args.keypoint_sigma,
    )

    stage_info = dataset.stage_info
    print(f"Stage {args.stage}: {stage_info['name']} ({stage_info['mode']})")

    # 获取第一个样本
    img, targets = next(iter(dataset))

    print(f"Sample shapes:")
    print(f"  image:     {img.shape}")
    print(f"  skeleton:  {targets['skeleton'].shape}")
    print(f"  keypoints: {targets['keypoints'].shape}")
    print(f"  tangent:   {targets['tangent'].shape}")
    print(f"  width:     {targets['width'].shape}")
    print(f"  offset:    {targets['offset'].shape}")

    if args.verify:
        verify_tangent_consistency(targets)

    save_path = None
    if args.save:
        import os

        os.makedirs(args.save, exist_ok=True)
        save_path = os.path.join(args.save, f"stage_{args.stage}.png")

    visualize_dense_batch(
        img,
        targets,
        save_path=save_path,
        extract_keypoints_flag=not args.no_extract_kps,
        kp_threshold=args.kp_threshold,
        kp_topk=args.kp_topk,
    )


if __name__ == "__main__":
    main()
