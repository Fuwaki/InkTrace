import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import from datasets (new API)
from datasets import DenseInkTraceDataset, collate_dense_batch
from torch.utils.data import DataLoader


def test_visualization(curriculum_stage=0, num_samples=4, save_path="vis_dense_gt.png"):
    """
    使用 DenseInkTraceDataset 可视化 Dense GT Maps。

    Args:
        curriculum_stage: 渐进训练阶段 (0-5)
        num_samples: 显示样本数
        save_path: 输出文件路径
    """
    # 1. 创建 Dataset
    dataset = DenseInkTraceDataset(
        mode="independent",
        batch_size=num_samples,
        epoch_length=num_samples,
        curriculum_stage=curriculum_stage,
        return_vector_labels=True,  # 返回原始矢量标签用于调试
    )

    print(
        f"[Dataset] Stage {curriculum_stage}: "
        f"img_size={dataset.img_size}, max_strokes={dataset.max_strokes}"
    )

    # 2. 获取一批数据
    dataloader = DataLoader(
        dataset,
        batch_size=num_samples,
        num_workers=0,
        collate_fn=collate_dense_batch,
    )

    imgs, targets, labels = next(iter(dataloader))

    # 转换为 numpy
    imgs = imgs.numpy()
    B = imgs.shape[0]

    print(f"[Data] imgs: {imgs.shape}, skeleton: {targets['skeleton'].shape}")

    # 3. Visualize
    # Keys: skeleton, junction, tangent, width, offset

    rows = B
    cols = 7  # Input, Skeleton, Junction, Tangent, Width, Offset, Overlay

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes[None, :]

    for i in range(B):
        # Input (squeeze channel dim)
        axes[i, 0].imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        # Skeleton
        skel = targets["skeleton"][i, 0].numpy()
        axes[i, 1].imshow(skel, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Skeleton")
        axes[i, 1].axis("off")

        # Junction
        junc = targets["junction"][i, 0].numpy()
        junc_vis = cv2.dilate(junc, np.ones((3, 3)))
        axes[i, 2].imshow(junc_vis, cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title("Junction")
        axes[i, 2].axis("off")

        # Tangent: Show Vector Field using HSV
        # Note: Targets are now Double Angle Representation (cos2t, sin2t)
        # This eliminates 0 vs 180 degree ambiguity.
        tan_x = targets["tangent"][i, 0].numpy()
        tan_y = targets["tangent"][i, 1].numpy()

        # Calculate angle in degrees [0, 360]
        # This angle is 2*theta.
        angle = np.degrees(np.arctan2(tan_y, tan_x))
        angle[angle < 0] += 360

        # Mask out non-skeleton areas
        mask = skel > 0

        # HSV: H=angle, S=1, V=mask
        # If angle=0 (Red) -> theta=0
        # If angle=180 (Cyan) -> theta=90
        # If angle=360 (Red) -> theta=180.
        # So Red means horizontal, Cyan means vertical.
        from matplotlib.colors import hsv_to_rgb

        hsv_img_norm = np.stack(
            [angle / 360.0, np.ones_like(angle), np.where(mask, 1.0, 0.0)], axis=-1
        )
        tan_vis = hsv_to_rgb(hsv_img_norm)

        axes[i, 3].imshow(tan_vis)
        axes[i, 3].set_title("Tangent (2Theta)")
        axes[i, 3].axis("off")

        # Width
        w_map = targets["width"][i, 0].numpy()
        axes[i, 4].imshow(w_map, cmap="viridis")
        axes[i, 4].set_title("Width")
        axes[i, 4].axis("off")

        # Offset Norm (show magnitude)
        off = targets["offset"][i].numpy()  # [2, H, W]
        off_norm = np.sqrt(off[0] ** 2 + off[1] ** 2)
        axes[i, 5].imshow(off_norm * skel, cmap="plasma")
        axes[i, 5].set_title("Offset")
        axes[i, 5].axis("off")

        # Overlay (Skeleton on Input)
        base = np.stack([imgs[i, 0]] * 3, axis=-1)
        base[skel > 0] = [1, 0, 0]
        axes[i, 6].imshow(base)
        axes[i, 6].set_title("Overlay")
        axes[i, 6].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=2, help="Curriculum stage (0-5)")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples")
    parser.add_argument(
        "--output", type=str, default="vis_dense_gt.png", help="Output path"
    )
    args = parser.parse_args()

    test_visualization(
        curriculum_stage=args.stage, num_samples=args.samples, save_path=args.output
    )
