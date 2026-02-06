#!/usr/bin/env python3
"""
Dense Prediction 模型可视化与评估工具 (V2)

功能：
1. 从 checkpoint 加载模型 (通过 ModelFactory)
2. 在测试数据上可视化预测结果 (Skeleton, Keypoints, Tangent, Width, Offset)
3. 计算评估指标 (IoU, Precision, Recall, F1)
4. 提供 DenseVisualizer 类用于 TensorBoard 集成

使用方法:
    # 可视化单个 checkpoint
    python visualize_dense.py --checkpoint checkpoints_dense/best_dense_model.pth --num-samples 8

    # 指定 curriculum stage 生成对应难度的测试数据
    python visualize_dense.py --checkpoint best.pth --stage 3 --num-samples 4

    # 计算统计指标
    python visualize_dense.py --checkpoint best.pth --stats-samples 100
"""

import argparse
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image

import torch
from torch.utils.data import DataLoader

from models import ModelFactory
from datasets_v2 import DenseInkTraceDataset, collate_dense_batch


# =============================================================================
# 可视化函数
# =============================================================================


def visualize_prediction(img, pred, target, save_path, idx=0):
    """
    Visualize prediction vs target for one sample.
    Shows ALL output heads: Skeleton, Keypoints (Topo/Geo), Tangent, Width, Offset

    Args:
        img: [1, H, W] input image tensor
        pred: dict of prediction tensors (each [1, C, H, W])
        target: dict of target tensors (each [1, C, H, W])
        save_path: output file path
        idx: sample index (for logging)
    """
    # Detach and CPU
    # Handle various input shapes: [1, 1, H, W], [1, H, W], [H, W]
    img_cpu = img.cpu()
    while img_cpu.dim() > 2:
        img_cpu = img_cpu.squeeze(0)
    img_np = img_cpu.numpy()  # [H, W]

    # Skeleton
    pred_skel = pred["skeleton"][0, 0].cpu().numpy()
    tgt_skel = target["skeleton"][0, 0].cpu().numpy()

    # Tangent (RGB Vis)
    pred_tan = pred["tangent"][0].cpu().numpy()  # [2, H, W]
    tgt_tan = target["tangent"][0].cpu().numpy()

    vis_pred_tan = _vis_tangent(pred_tan, pred_skel)
    vis_tgt_tan = _vis_tangent(tgt_tan, tgt_skel)

    # Keypoints: [2, H, W] - ch0=topo, ch1=geo
    tgt_kp = target["keypoints"][0].cpu().numpy()
    pred_kp = pred["keypoints"][0].cpu().numpy()

    # Width: [1, H, W]
    pred_width = pred["width"][0, 0].cpu().numpy()
    tgt_width = target["width"][0, 0].cpu().numpy()

    # Offset: [2, H, W]
    pred_offset = pred["offset"][0].cpu().numpy()
    tgt_offset = target["offset"][0].cpu().numpy()

    # Create 4x5 grid for comprehensive visualization
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

    # Row 2: Keypoints - Topo GT, Topo Pred, Geo GT, Geo Pred, Combined
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

    # Row 3: Tangent GT, Tangent Pred, Width GT, Width Pred, Width Diff
    axes[2, 0].imshow(vis_tgt_tan)
    axes[2, 0].set_title("GT Tangent")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(vis_pred_tan)
    axes[2, 1].set_title("Pred Tangent")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(tgt_width * tgt_skel, cmap="viridis")
    axes[2, 2].set_title("GT Width")
    axes[2, 2].axis("off")

    axes[2, 3].imshow(pred_width * pred_skel, cmap="viridis")
    axes[2, 3].set_title("Pred Width")
    axes[2, 3].axis("off")

    # Width difference (masked)
    width_diff = np.abs(pred_width - tgt_width) * np.maximum(pred_skel, tgt_skel)
    axes[2, 4].imshow(width_diff, cmap="hot")
    axes[2, 4].set_title("Width Diff")
    axes[2, 4].axis("off")

    # Row 4: Offset X GT, Offset X Pred, Offset Y GT, Offset Y Pred, Offset Magnitude
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

    # Offset magnitude comparison
    tgt_off_mag = np.sqrt(tgt_offset[0] ** 2 + tgt_offset[1] ** 2) * tgt_skel
    pred_off_mag = np.sqrt(pred_offset[0] ** 2 + pred_offset[1] ** 2) * pred_skel
    off_diff = np.abs(pred_off_mag - tgt_off_mag)
    axes[3, 4].imshow(off_diff, cmap="hot")
    axes[3, 4].set_title("Offset Mag Diff")
    axes[3, 4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_visualization_grid(imgs, outputs, targets, num_samples=4):
    """
    Create a visualization grid for multiple samples.
    Returns a numpy array (H, W, 3) suitable for TensorBoard add_image.

    Args:
        imgs: [B, 1, H, W] input images
        outputs: dict of prediction tensors
        targets: dict of target tensors
        num_samples: number of samples to visualize

    Returns:
        img_arr: numpy array [H, W, 3] RGB image
    """
    num_samples = min(num_samples, imgs.shape[0])
    rows = num_samples
    cols = 6  # Input, GT Skel, Pred Skel, GT Tan, Pred Tan, Overlay

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        img = imgs[i, 0].cpu().numpy()
        pred_skel = outputs["skeleton"][i, 0].cpu().numpy()
        tgt_skel = targets["skeleton"][i, 0].cpu().numpy()
        pred_tan = outputs["tangent"][i].cpu().numpy()
        tgt_tan = targets["tangent"][i].cpu().numpy()

        # Input
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("Input" if i == 0 else "")
        axes[i, 0].axis("off")

        # GT Skeleton
        axes[i, 1].imshow(tgt_skel, cmap="gray")
        axes[i, 1].set_title("GT Skel" if i == 0 else "")
        axes[i, 1].axis("off")

        # Pred Skeleton
        axes[i, 2].imshow(pred_skel, cmap="gray")
        axes[i, 2].set_title("Pred Skel" if i == 0 else "")
        axes[i, 2].axis("off")

        # GT Tangent
        axes[i, 3].imshow(_vis_tangent(tgt_tan, tgt_skel))
        axes[i, 3].set_title("GT Tan" if i == 0 else "")
        axes[i, 3].axis("off")

        # Pred Tangent
        axes[i, 4].imshow(_vis_tangent(pred_tan, pred_skel))
        axes[i, 4].set_title("Pred Tan" if i == 0 else "")
        axes[i, 4].axis("off")

        # Overlay
        overlay = np.stack([img, img, img], axis=-1)
        overlay[pred_skel > 0.5] = [1, 0, 0]  # Red for prediction
        overlay[tgt_skel > 0.5] = [0, 1, 0]  # Green for GT (overlaps show yellow)
        axes[i, 5].imshow(overlay)
        axes[i, 5].set_title("Overlay (R=Pred,G=GT)" if i == 0 else "")
        axes[i, 5].axis("off")

    plt.tight_layout()

    # Convert to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_arr = np.array(Image.open(buf))[..., :3]  # Remove alpha channel
    plt.close(fig)

    return img_arr


def _vis_tangent(tan_map, mask):
    """
    Visualize tangent field as HSV color image.

    Args:
        tan_map: [2, H, W] (cos2θ, sin2θ)
        mask: [H, W] skeleton mask

    Returns:
        rgb: [H, W, 3] RGB image
    """
    angle = np.degrees(np.arctan2(tan_map[1], tan_map[0]))
    angle[angle < 0] += 360
    hsv = np.stack(
        [angle / 360.0, np.ones_like(angle), np.where(mask > 0.5, 1.0, 0.0)],
        axis=-1,
    )
    return hsv_to_rgb(hsv)


# =============================================================================
# 评估指标
# =============================================================================


def compute_metrics(outputs, targets):
    """
    Compute evaluation metrics for skeleton and keypoints prediction.

    Args:
        outputs: dict of prediction tensors
        targets: dict of target tensors

    Returns:
        dict of metric values
    """
    pred_skel = (outputs["skeleton"] > 0.5).float()
    tgt_skel = (targets["skeleton"] > 0.5).float()

    # Pixel-wise metrics for skeleton
    intersection = (pred_skel * tgt_skel).sum()
    union = pred_skel.sum() + tgt_skel.sum() - intersection

    # IoU
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Precision & Recall
    precision = (intersection + 1e-6) / (pred_skel.sum() + 1e-6)
    recall = (intersection + 1e-6) / (tgt_skel.sum() + 1e-6)

    # F1
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # Keypoints metrics (2 channels: topo + geo)
    # Use threshold for heatmap detection
    pred_kp_topo = (outputs["keypoints"][:, 0:1] > 0.3).float()
    tgt_kp_topo = (targets["keypoints"][:, 0:1] > 0.3).float()
    pred_kp_geo = (outputs["keypoints"][:, 1:2] > 0.3).float()
    tgt_kp_geo = (targets["keypoints"][:, 1:2] > 0.3).float()

    # Topo keypoints recall
    kp_topo_inter = (pred_kp_topo * tgt_kp_topo).sum()
    kp_topo_recall = (kp_topo_inter + 1e-6) / (tgt_kp_topo.sum() + 1e-6)

    # Geo keypoints recall
    kp_geo_inter = (pred_kp_geo * tgt_kp_geo).sum()
    kp_geo_recall = (kp_geo_inter + 1e-6) / (tgt_kp_geo.sum() + 1e-6)

    return {
        "skel_iou": iou.item(),
        "skel_precision": precision.item(),
        "skel_recall": recall.item(),
        "skel_f1": f1.item(),
        "kp_topo_recall": kp_topo_recall.item(),
        "kp_geo_recall": kp_geo_recall.item(),
    }


def compute_batch_metrics(model, dataloader, device, num_batches=10):
    """
    Compute metrics over multiple batches.

    Args:
        model: model instance
        dataloader: data loader
        device: torch device
        num_batches: number of batches to evaluate

    Returns:
        dict of averaged metrics
    """
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break

            imgs = imgs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(imgs)
            metrics = compute_metrics(outputs, targets)
            all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)

    return avg_metrics


# =============================================================================
# DenseVisualizer for TensorBoard Integration
# =============================================================================


class DenseVisualizer:
    """
    用于在训练过程中生成可视化并写入 TensorBoard 的类。

    Usage:
        visualizer = DenseVisualizer(writer, device)
        visualizer.visualize(model, dataloader, global_step)
    """

    def __init__(self, writer, device, num_samples=4):
        """
        Args:
            writer: TensorBoard SummaryWriter
            device: torch device
            num_samples: number of samples to visualize
        """
        self.writer = writer
        self.device = device
        self.num_samples = num_samples

    def visualize(self, model, dataloader, global_step, prefix="Dense"):
        """
        Generate visualization and write to TensorBoard.

        Args:
            model: model instance (in eval mode)
            dataloader: data loader
            global_step: current training step
            prefix: tag prefix for TensorBoard
        """
        model.eval()

        # Get a batch
        try:
            imgs, targets = next(iter(dataloader))
        except StopIteration:
            print("Warning: dataloader is empty, skipping visualization")
            return

        imgs = imgs[: self.num_samples].to(self.device)
        targets = {k: v[: self.num_samples].to(self.device) for k, v in targets.items()}

        with torch.no_grad():
            outputs = model(imgs)

        # Create visualization grid
        grid = create_visualization_grid(
            imgs,
            {k: v.detach() for k, v in outputs.items()},
            {k: v.detach() for k, v in targets.items()},
            num_samples=self.num_samples,
        )

        # Write to TensorBoard (HWC -> CHW)
        grid_tensor = torch.from_numpy(grid).permute(2, 0, 1).float() / 255.0
        self.writer.add_image(f"{prefix}/Visualization", grid_tensor, global_step)

        # Compute and log metrics
        metrics = compute_metrics(outputs, targets)
        for name, value in metrics.items():
            self.writer.add_scalar(f"Metrics/{name}", value, global_step)

        model.train()
        return metrics


# =============================================================================
# 模型加载
# =============================================================================


def load_dense_model(checkpoint_path, device="cpu", config=None):
    """
    Load model from checkpoint using ModelFactory.

    Args:
        checkpoint_path: path to .pth file
        device: torch device
        config: optional config dict

    Returns:
        model: model in eval mode
        checkpoint: raw checkpoint dict (for metadata)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Determine model config from checkpoint or use defaults
    model_config = {
        "encoder_type": "repvit",
        "img_size": 64,
        "embed_dim": 128,
    }

    if config and "model" in config:
        model_config.update(config["model"])

    if "config" in checkpoint:
        ckpt_config = checkpoint["config"]
        if "model" in ckpt_config:
            model_config.update(ckpt_config["model"])

    print(f"  Model config: {model_config}")

    # Create model using factory
    model = ModelFactory.create_unified_model(
        embed_dim=model_config.get("embed_dim", 128),
        num_layers=model_config.get("num_layers", 4),
        full_heads=True,  # Dense model needs all heads
        device=device,
    )

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise KeyError("Checkpoint missing 'model_state_dict'")

    model.eval()

    # Print metadata
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "loss" in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.6f}")

    print("  ✓ Model loaded successfully")
    return model, checkpoint


# =============================================================================
# 主程序
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Dense Prediction 可视化与评估 (V2)")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="模型 checkpoint 路径"
    )
    parser.add_argument("--stage", type=int, default=2, help="Curriculum stage (0-9)")
    parser.add_argument("--num-samples", type=int, default=8, help="可视化样本数")
    parser.add_argument(
        "--stats-samples", type=int, default=0, help="统计指标样本数 (0=跳过)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results_dense", help="输出目录"
    )
    parser.add_argument("--device", type=str, default=None, help="设备 (auto/cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=64, help="Image size")

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, ckpt = load_dense_model(args.checkpoint, device)

    # Dataset (V2)
    dataset = DenseInkTraceDataset(
        img_size=args.img_size,
        batch_size=args.batch_size,
        epoch_length=max(args.num_samples, args.stats_samples, 100),
        curriculum_stage=args.stage,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate_dense_batch,
    )

    print(f"Dataset: stage={args.stage}, img_size={dataset.img_size}")

    # 1. Visualization
    if args.num_samples > 0:
        print(f"\nGenerating visualization ({args.num_samples} samples)...")

        imgs, targets = next(iter(dataloader))
        imgs = imgs[: args.num_samples].to(device)
        targets = {k: v[: args.num_samples].to(device) for k, v in targets.items()}

        with torch.no_grad():
            outputs = model(imgs)

        # Grid visualization
        grid = create_visualization_grid(
            imgs,
            {k: v.detach() for k, v in outputs.items()},
            {k: v.detach() for k, v in targets.items()},
            num_samples=min(args.num_samples, 4),
        )

        grid_path = os.path.join(args.output_dir, "vis_grid.png")
        Image.fromarray(grid).save(grid_path)
        print(f"  Saved: {grid_path}")

        # Individual samples
        for i in range(min(args.num_samples, 4)):
            sample_path = os.path.join(args.output_dir, f"vis_sample_{i}.png")
            visualize_prediction(
                imgs[i : i + 1],
                {k: v[i : i + 1].detach() for k, v in outputs.items()},
                {k: v[i : i + 1].detach() for k, v in targets.items()},
                sample_path,
                idx=i,
            )
        print(f"  Saved individual samples to {args.output_dir}/")

    # 2. Statistics
    if args.stats_samples > 0:
        print(f"\nComputing statistics ({args.stats_samples} samples)...")

        num_batches = (args.stats_samples + args.batch_size - 1) // args.batch_size
        metrics = compute_batch_metrics(model, dataloader, device, num_batches)

        print("\n=== Evaluation Metrics ===")
        print(
            f"  Skeleton IoU:       {metrics['skel_iou']:.4f} ± {metrics['skel_iou_std']:.4f}"
        )
        print(
            f"  Skeleton Precision: {metrics['skel_precision']:.4f} ± {metrics['skel_precision_std']:.4f}"
        )
        print(
            f"  Skeleton Recall:    {metrics['skel_recall']:.4f} ± {metrics['skel_recall_std']:.4f}"
        )
        print(
            f"  Skeleton F1:        {metrics['skel_f1']:.4f} ± {metrics['skel_f1_std']:.4f}"
        )
        print(
            f"  KP Topo Recall:     {metrics['kp_topo_recall']:.4f} ± {metrics['kp_topo_recall_std']:.4f}"
        )
        print(
            f"  KP Geo Recall:      {metrics['kp_geo_recall']:.4f} ± {metrics['kp_geo_recall_std']:.4f}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
