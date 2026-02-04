#!/usr/bin/env python3
"""
测试Graph Reconstruction算法

加载训练好的模型，生成测试数据，运行后处理，可视化结果
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import torch
from torch.utils.data import DataLoader
import argparse
from PIL import Image
import torchvision.transforms.functional as TF

# Import model and dataset
from encoder import StrokeEncoder
from models import DenseVectorModel
from datasets import DenseInkTraceDataset, collate_dense_batch
from graph_reconstruction import SimpleGraphReconstructor, fit_bezier_curves


def load_model(checkpoint_path, device="cpu"):
    """加载训练好的模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Determine embed_dim from checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        for key in state_dict.keys():
            if "encoder.token_embed.weight" in key:
                embed_dim = state_dict[key].shape[0]
                break
        else:
            embed_dim = 128
    else:
        embed_dim = 128

    print(f"  Detected embed_dim: {embed_dim}")

    # Create model
    encoder = StrokeEncoder(in_channels=1, embed_dim=embed_dim)
    model = DenseVectorModel(encoder).to(device)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "loss" in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.6f}")

    print("  Model loaded successfully")
    return model, checkpoint


def bezier_to_points(control_points, num_samples=50):
    """将贝塞尔曲线控制点转换为采样点"""
    p0, p1, p2, p3 = [np.array(p) for p in control_points]
    t = np.linspace(0, 1, num_samples)

    points = (
        np.outer((1 - t) ** 3, p0)
        + np.outer(3 * (1 - t) ** 2 * t, p1)
        + np.outer(3 * (1 - t) * t**2, p2)
        + np.outer(t**3, p3)
    )

    return points


def visualize_reconstruction(
    img, pred_maps, vector_paths, save_path, strokes=None, gt_targets=None
):
    """
    可视化重建结果

    Args:
        img: [H, W] input image
        pred_maps: dict of prediction maps
        vector_paths: list of bezier curves
        save_path: output path
        strokes: list of traced strokes (optional, for debugging)
        gt_targets: dict of GT maps (optional, for comparison)
    """
    n_rows = 4 if strokes is not None else 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))

    # Row 1: Input and prediction maps
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_maps["skeleton"], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Predicted Skeleton (max={pred_maps['skeleton'].max():.2f})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_maps["junction"], cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Predicted Junction (max={pred_maps['junction'].max():.2f})")
    axes[0, 2].axis("off")

    # Row 2: GT comparison (if available) or more predictions
    if gt_targets is not None and "skeleton" in gt_targets:
        gt_skel = gt_targets["skeleton"]
        if gt_skel.ndim == 3:
            gt_skel = gt_skel.squeeze(0)
        axes[1, 0].imshow(gt_skel, cmap="gray", vmin=0, vmax=1)
        axes[1, 0].set_title("GT Skeleton")
        axes[1, 0].axis("off")

        # Difference between pred and GT
        diff = np.abs(pred_maps["skeleton"] - gt_skel)
        axes[1, 1].imshow(diff, cmap="hot", vmin=0, vmax=1)
        axes[1, 1].set_title(f"Skeleton Diff (MAE={diff.mean():.3f})")
        axes[1, 1].axis("off")
    else:
        # Visualize tangent field
        angle = np.degrees(np.arctan2(pred_maps["tangent"][1], pred_maps["tangent"][0]))
        angle[angle < 0] += 360
        from matplotlib.colors import hsv_to_rgb

        hsv = np.stack(
            [
                angle / 360.0,
                np.ones_like(angle),
                np.where(pred_maps["skeleton"] > 0.2, 1.0, 0.0),
            ],
            axis=-1,
        )
        axes[1, 0].imshow(hsv_to_rgb(hsv))
        axes[1, 0].set_title("Tangent Field")
        axes[1, 0].axis("off")

        # Width map
        axes[1, 1].imshow(pred_maps["width"], cmap="viridis")
        axes[1, 1].set_title(f"Width Map (max={pred_maps['width'].max():.2f})")
        axes[1, 1].axis("off")

    # Overlay skeleton on input
    overlay = np.stack([img, img, img], axis=-1)
    skeleton_mask = pred_maps["skeleton"] > 0.2
    overlay[skeleton_mask] = [1, 0, 0]
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Skeleton Overlay (threshold=0.2)")
    axes[1, 2].axis("off")

    # Row 3: Reconstructed vector paths
    axes[2, 0].imshow(img, cmap="gray", alpha=0.3)
    axes[2, 0].set_title(f"Bezier Curves ({len(vector_paths)} curves)")

    # Draw each bezier curve with different colors
    curve_colors = plt.cm.Set1(np.linspace(0, 1, max(len(vector_paths), 1)))
    for i, curve in enumerate(vector_paths):
        points = bezier_to_points(curve["points"])
        axes[2, 0].plot(
            points[:, 0], points[:, 1], "-", color=curve_colors[i], linewidth=2
        )
        # Mark control points
        cp = np.array(curve["points"])
        axes[2, 0].plot(
            cp[:, 0], cp[:, 1], "o", color=curve_colors[i], markersize=3, alpha=0.5
        )

    axes[2, 0].axis("off")
    axes[2, 0].set_xlim(0, img.shape[1])
    axes[2, 0].set_ylim(img.shape[0], 0)  # Flip y-axis

    # Compare with input
    axes[2, 1].imshow(img, cmap="gray")
    for i, curve in enumerate(vector_paths):
        points = bezier_to_points(curve["points"])
        axes[2, 1].plot(
            points[:, 0], points[:, 1], "-", color="red", linewidth=1, alpha=0.8
        )
    axes[2, 1].set_title("Curves on Input")
    axes[2, 1].axis("off")

    # Statistics
    if len(vector_paths) > 0:
        lengths = [
            np.linalg.norm(np.array(c["points"][3]) - np.array(c["points"][0]))
            for c in vector_paths
        ]
        widths = [c["width"] for c in vector_paths]
        axes[2, 2].bar(
            ["Curves", "Avg Len", "Avg Width"],
            [len(vector_paths), np.mean(lengths), np.mean(widths)],
        )
        axes[2, 2].set_title("Statistics")
    else:
        axes[2, 2].text(0.5, 0.5, "No curves", ha="center", va="center")
        axes[2, 2].axis("off")

    # Row 4: Traced strokes (debugging)
    if strokes is not None and n_rows >= 4:
        axes[3, 0].imshow(img, cmap="gray", alpha=0.3)
        axes[3, 0].set_title(f"Traced Strokes ({len(strokes)} strokes)")

        # Draw each stroke with different colors
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(strokes), 1)))
        for i, stroke in enumerate(strokes):
            pts = np.array(stroke["points"])
            axes[3, 0].plot(pts[:, 0], pts[:, 1], "-", color=colors[i], linewidth=1)
            # Mark start and end
            axes[3, 0].plot(pts[0, 0], pts[0, 1], "go", markersize=3)
            axes[3, 0].plot(pts[-1, 0], pts[-1, 1], "ro", markersize=3)

        axes[3, 0].axis("off")
        axes[3, 0].set_xlim(0, img.shape[1])
        axes[3, 0].set_ylim(img.shape[0], 0)

        # Stroke length histogram
        if len(strokes) > 0:
            axes[3, 1].hist(
                [len(s["points"]) for s in strokes], bins=min(20, len(strokes))
            )
            axes[3, 1].set_title("Stroke Length Distribution")
            axes[3, 1].set_xlabel("Pixels")
            axes[3, 1].set_ylabel("Count")

            # Width histogram
            axes[3, 2].hist([s["width"] for s in strokes], bins=min(20, len(strokes)))
            axes[3, 2].set_title("Stroke Width Distribution")
            axes[3, 2].set_xlabel("Width")
            axes[3, 2].set_ylabel("Count")
        else:
            axes[3, 1].text(0.5, 0.5, "No strokes", ha="center", va="center")
            axes[3, 2].text(0.5, 0.5, "No strokes", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualization: {save_path}")


def test_single_sample(model, dataloader, device, output_dir, idx=0):
    """测试单个样本"""
    model.eval()

    # Get a single sample from dataloader
    imgs, targets = next(iter(dataloader))
    img = imgs[0]  # Take first from batch
    img_tensor = img.unsqueeze(0).to(device)

    print(f"\n=== Testing sample {idx} ===")
    print(f"  Image shape: {img.shape}")

    # Model inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Convert predictions to numpy
    pred_maps = {}
    for key in ["skeleton", "junction", "tangent", "width", "offset"]:
        if key in outputs:
            data = outputs[key][0].cpu().numpy()
            # Handle different shapes: [1, H, W] or [2, H, W]
            if data.ndim == 3:
                if data.shape[0] == 1:  # [1, H, W] -> [H, W]
                    data = data.squeeze(0)
                # else: [2, H, W] keep as is
            pred_maps[key] = data

    print(f"  Prediction shapes:")
    for key, val in pred_maps.items():
        print(f"    {key}: {val.shape}")

    # Run graph reconstruction
    print(f"\n  Running graph reconstruction...")

    # 尝试不同的 skeleton 阈值
    skeleton_threshold = 0.2  # 降低阈值以获取更多骨架点

    reconstructor = SimpleGraphReconstructor(
        config={
            "skeleton_threshold": skeleton_threshold,
            "junction_threshold": 0.5,
            "min_stroke_length": 5,
            "use_thinning": True,
            "cross_junctions": True,  # 穿越交叉点，保持笔画完整性
        }
    )

    # First, trace strokes
    strokes = reconstructor.process(pred_maps)

    # Then fit bezier curves (使用较大的容差，减少过度分割)
    bezier_curves = fit_bezier_curves(strokes, tolerance=3.0)
    print(f"  Generated {len(bezier_curves)} Bezier curves")

    # Print some curve info
    for i, curve in enumerate(bezier_curves[:5]):
        p0, p1, p2, p3 = curve["points"]
        length = np.linalg.norm(np.array(p3) - np.array(p0))
        print(
            f"    Curve {i}: width={curve['width']:.2f}, length={length:.1f}, "
            f"endpoints={p0} -> {p3}"
        )

    # Also visualize strokes before fitting
    print(f"  Stroke lengths: {[len(s['points']) for s in strokes]}")

    # Convert GT targets to numpy for visualization
    gt_targets = {}
    for key in ["skeleton", "junction", "tangent", "width", "offset"]:
        if key in targets:
            data = targets[key][0]
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            gt_targets[key] = data

    # Visualize
    img_np = img.cpu().numpy().squeeze()
    save_path = os.path.join(output_dir, f"reconstruction_{idx}.png")
    visualize_reconstruction(
        img_np, pred_maps, bezier_curves, save_path, strokes, gt_targets
    )

    return bezier_curves, pred_maps


def test_multiple_samples(model, dataset, device, output_dir, num_samples=4):
    """测试多个样本"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Testing {num_samples} samples")
    print(f"{'=' * 60}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_dense_batch,
    )

    all_results = []
    for i in range(num_samples):
        try:
            bezier_curves, pred_maps = test_single_sample(
                model, dataloader, device, output_dir, idx=i
            )
            all_results.append(
                {
                    "idx": i,
                    "num_curves": len(bezier_curves),
                    "paths": bezier_curves,
                    "pred_maps": pred_maps,
                }
            )
        except Exception as e:
            print(f"  ERROR in sample {i}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    if len(all_results) > 0:
        total_curves = sum(r["num_curves"] for r in all_results)
        print(f"  Total samples processed: {len(all_results)}")
        print(f"  Total curves generated: {total_curves}")
        print(f"  Average curves per sample: {total_curves / len(all_results):.1f}")


def test_custom_image(
    model, image_path, device, output_dir, img_size=64, invert=False, threshold=0.2
):
    """测试单张自定义图片"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    print(f"\n=== Testing custom image: {filename} ===")

    # Load image
    try:
        pil_img = Image.open(image_path).convert("L")  # Gray scale
        print(f"  Loaded image size: {pil_img.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Resize
    if pil_img.size != (img_size, img_size):
        print(f"  Resizing to {img_size}x{img_size}")
        pil_img = pil_img.resize((img_size, img_size), Image.Resampling.BILINEAR)

    # To Tensor and Normalize
    img_tensor = TF.to_tensor(pil_img)  # [1, H, W], range [0, 1]

    if invert:
        print("  Inverting image colors")
        img_tensor = 1.0 - img_tensor

    img_tensor_dev = img_tensor.to(device)
    input_tensor = img_tensor_dev.unsqueeze(0)  # [1, 1, H, W]

    # Model inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)

    # Process outputs
    pred_maps = {}
    for key in ["skeleton", "junction", "tangent", "width", "offset"]:
        if key in outputs:
            data = outputs[key][0].cpu().numpy()
            if data.ndim == 3 and data.shape[0] == 1:
                data = data.squeeze(0)
            pred_maps[key] = data

    # Reconstruction
    print("  Running reconstruction...")
    reconstructor = SimpleGraphReconstructor(
        config={
            "skeleton_threshold": threshold,
            "junction_threshold": 0.5,
            "min_stroke_length": 5,
            "use_thinning": True,
            "cross_junctions": True,
        }
    )

    strokes = reconstructor.process(pred_maps)
    bezier_curves = fit_bezier_curves(strokes, tolerance=3.0)
    print(f"  Generated {len(bezier_curves)} curves")

    # Visualize
    img_np = (
        img_tensor.squeeze(0).squeeze(0).numpy()
    )  # Use CPU tensor for visualization
    save_path = os.path.join(output_dir, f"reconstruction_{name}.png")

    visualize_reconstruction(img_np, pred_maps, bezier_curves, save_path, strokes, None)

    return bezier_curves


def main():
    parser = argparse.ArgumentParser(
        description="Test Graph Reconstruction on InkTrace model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_dense_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to custom image file (optional)"
    )
    parser.add_argument(
        "--output", type=str, default="results_reconstruction", help="Output directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of random samples if no image provided",
    )
    parser.add_argument(
        "--size", type=int, default=64, help="Input image size for resizing"
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert image colors (for white background)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2, help="Skeleton threshold"
    )
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()

    # Configuration
    checkpoint_path = args.checkpoint
    output_dir = args.output

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    print(f"Using device: {device}")

    # Load model
    try:
        model, ckpt = load_model(checkpoint_path, device)
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    if args.image:
        # Test custom image
        test_custom_image(
            model,
            args.image,
            device,
            output_dir,
            img_size=args.size,
            invert=args.invert,
            threshold=args.threshold,
        )
    else:
        # Test random samples from dataset
        num_samples = args.num_samples
        print("\nCreating dataset...")
        dataset = DenseInkTraceDataset(
            mode="independent",
            img_size=args.size,
            batch_size=1,
            epoch_length=num_samples,
            curriculum_stage=2,  # Medium difficulty
        )

        print(f"  Dataset size: {len(dataset)}")
        print(f"  Max strokes: {dataset.max_strokes}")

        # Test
        test_multiple_samples(model, dataset, device, output_dir, num_samples)

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
