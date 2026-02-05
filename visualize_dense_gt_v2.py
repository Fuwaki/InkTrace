"""
Dense GT Maps 可视化工具 v2

可视化 Rust 生成的 Dense GT Maps:
- skeleton: 骨架图
- keypoints: 关键点图 (2通道: 拓扑节点 + 几何锚点)
- tangent: 切向场 (HSV 颜色编码)
- width: 宽度图
- offset: 亚像素偏移

用法:
    python visualize_dense_gt_v2.py                 # 可视化单个样本
    python visualize_dense_gt_v2.py --all-stages    # 可视化所有阶段
    python visualize_dense_gt_v2.py --stage 5       # 可视化指定阶段
"""

import matplotlib.pyplot as plt
import numpy as np
import ink_trace_rs
import argparse
import colorsys


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


def visualize_dense_batch(batch: dict, idx: int = 0, save_path: str = None):
    """
    可视化单个样本的 Dense GT Maps

    Args:
        batch: Rust 生成的数据批次
        idx: 批次中的样本索引
        save_path: 保存路径 (None 则显示)
    """
    img = batch["image"][idx]
    skeleton = batch["skeleton"][idx]
    keypoints = batch["keypoints"][idx]  # [2, H, W]
    tangent = batch["tangent"][idx]
    width = batch["width"][idx]
    offset = batch["offset"][idx]

    # 分离 keypoints 两个通道
    topo_keypoints = keypoints[0]  # 拓扑节点
    geom_keypoints = keypoints[1]  # 几何锚点

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
    axes[1, 0].set_title(f"Topo Keypoints (sum={topo_keypoints.sum():.1f})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(geom_keypoints, cmap="Greens", vmin=0, vmax=1)
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


def visualize_all_stages(img_size: int = 64, save_dir: str = None):
    """可视化所有训练阶段的样本"""
    stages = ink_trace_rs.list_stages()

    fig, axes = plt.subplots(5, 10, figsize=(24, 12))

    for stage_info in stages:
        stage = stage_info["stage"]
        batch = ink_trace_rs.generate_dense_batch(1, img_size, stage)

        col = stage
        img = batch["image"][0]
        skeleton = batch["skeleton"][0]
        keypoints = batch["keypoints"][0]
        tangent = batch["tangent"][0]
        width = batch["width"][0]

        topo_kp = keypoints[0]
        geom_kp = keypoints[1]

        # Row 0: Image
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"S{stage}", fontsize=8)
        axes[0, col].axis("off")

        # Row 1: Skeleton
        axes[1, col].imshow(skeleton, cmap="gray")
        axes[1, col].axis("off")

        # Row 2: Keypoints RGB
        kp_rgb = np.zeros((*skeleton.shape, 3))
        kp_rgb[..., 0] = topo_kp
        kp_rgb[..., 1] = geom_kp
        axes[2, col].imshow(kp_rgb.clip(0, 1))
        axes[2, col].axis("off")

        # Row 3: Tangent
        tangent_rgb = tangent_to_rgb(tangent[0], tangent[1], skeleton)
        axes[3, col].imshow(tangent_rgb)
        axes[3, col].axis("off")

        # Row 4: Width
        axes[4, col].imshow(width * skeleton, cmap="viridis")
        axes[4, col].axis("off")

    # Row labels
    row_labels = ["Image", "Skeleton", "Keypoints", "Tangent", "Width"]
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=10)

    plt.suptitle("Curriculum Learning Stages (0-9)", fontsize=14)
    plt.tight_layout()

    if save_dir:
        import os

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "all_stages.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")
        plt.close()
    else:
        plt.show()


def verify_tangent_consistency(batch: dict, idx: int = 0):
    """验证切向场的一致性（双倍角表示）"""
    tangent = batch["tangent"][idx]
    skeleton = batch["skeleton"][idx]

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
    parser.add_argument("--all-stages", action="store_true", help="可视化所有阶段")
    parser.add_argument("--img-size", type=int, default=64, help="图像尺寸")
    parser.add_argument("--save", type=str, default=None, help="保存目录")
    parser.add_argument("--verify", action="store_true", help="验证切向场一致性")
    args = parser.parse_args()

    if args.all_stages:
        visualize_all_stages(args.img_size, args.save)
    else:
        # 生成单个阶段的数据
        stage_info = ink_trace_rs.get_stage_info(args.stage)
        print(f"Stage {args.stage}: {stage_info['name']} ({stage_info['mode']})")

        batch = ink_trace_rs.generate_dense_batch(4, args.img_size, args.stage)
        print(f"Generated batch shapes:")
        print(f"  image:     {batch['image'].shape}")
        print(f"  skeleton:  {batch['skeleton'].shape}")
        print(f"  keypoints: {batch['keypoints'].shape}")
        print(f"  tangent:   {batch['tangent'].shape}")
        print(f"  width:     {batch['width'].shape}")
        print(f"  offset:    {batch['offset'].shape}")

        if args.verify:
            verify_tangent_consistency(batch, 0)

        save_path = None
        if args.save:
            import os

            os.makedirs(args.save, exist_ok=True)
            save_path = os.path.join(args.save, f"stage_{args.stage}.png")

        visualize_dense_batch(batch, 0, save_path)


if __name__ == "__main__":
    main()
