#!/usr/bin/env python3
"""
结构预训练脚本 (Structural Pretraining)

核心理念：
  - 使用遮挡输入 + 无跳连解码
  - 强迫 Encoder 从残缺图像推断完整的骨架结构和笔画方向
  - Encoder 必须在 bottleneck (8×8×128) 中编码结构化信息

训练流程：
  1. 输入图像随机遮挡 (mask_ratio=0.5~0.7)
  2. Encoder 提取特征，Decoder 关闭跳连 (use_skips=False)
  3. 预测完整的 skeleton + tangent
  4. Loss = L_skeleton + λ * L_tangent

使用方法:
  # 从零开始预训练
  python train_structural.py --from-scratch --epochs 50

  # 从检查点继续
  python train_structural.py --resume checkpoints/structural_epoch10.pth

  # 预训练完成后，进行正式训练
  python train_dense.py --init_from checkpoints/best_structural.pth --stage 1
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from models import ModelFactory, MaskingGenerator, StructuralPretrainLoss
from datasets import DenseInkTraceDataset, collate_dense_batch


def parse_args():
    parser = argparse.ArgumentParser(description="结构预训练脚本")

    # Checkpoint
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--from-scratch", action="store_true", help="从零开始训练")
    ckpt_group.add_argument("--resume", type=str, help="从检查点恢复")

    # Training params
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument(
        "--epoch-length", type=int, default=10000, help="每 epoch 样本数"
    )

    # Model params
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding 维度")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer 层数")

    # Masking params
    parser.add_argument("--mask-ratio", type=float, default=0.6, help="遮挡比例")
    parser.add_argument(
        "--mask-strategy",
        type=str,
        default="block",
        choices=["block", "random"],
        help="遮挡策略",
    )

    # Other
    parser.add_argument("--device", type=str, default=None, help="设备")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="保存目录")
    parser.add_argument(
        "--log-dir", type=str, default="runs", help="TensorBoard 日志目录"
    )

    return parser.parse_args()


def visualize_predictions(
    imgs, masked_imgs, mask, pred_skel, pred_tan, gt_skel, gt_tan, step
):
    """可视化预测结果"""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    idx = 0  # 只显示第一个样本

    # Row 1: Input & Skeleton
    axes[0, 0].imshow(imgs[idx, 0].cpu(), cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(masked_imgs[idx, 0].cpu(), cmap="gray")
    axes[0, 1].set_title(f"Masked Input")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask[idx, 0].cpu(), cmap="gray")
    axes[0, 2].set_title("Mask")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(gt_skel[idx, 0].cpu(), cmap="gray")
    axes[0, 3].set_title("GT Skeleton")
    axes[0, 3].axis("off")

    axes[0, 4].imshow(pred_skel[idx, 0].detach().cpu(), cmap="gray")
    axes[0, 4].set_title("Pred Skeleton")
    axes[0, 4].axis("off")

    # Skeleton diff
    diff = torch.abs(pred_skel[idx, 0].detach().cpu() - gt_skel[idx, 0].cpu())
    axes[0, 5].imshow(diff, cmap="hot")
    axes[0, 5].set_title("Skeleton Diff")
    axes[0, 5].axis("off")

    # Row 2: Tangent Field
    # 将 cos2θ, sin2θ 转换为角度用于可视化
    gt_angle = torch.atan2(gt_tan[idx, 1], gt_tan[idx, 0]).cpu() / 2  # 恢复原始角度
    pred_angle = torch.atan2(pred_tan[idx, 1], pred_tan[idx, 0]).detach().cpu() / 2

    axes[1, 0].imshow(gt_tan[idx, 0].cpu(), cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 0].set_title("GT cos2θ")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gt_tan[idx, 1].cpu(), cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 1].set_title("GT sin2θ")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(pred_tan[idx, 0].detach().cpu(), cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 2].set_title("Pred cos2θ")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(pred_tan[idx, 1].detach().cpu(), cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 3].set_title("Pred sin2θ")
    axes[1, 3].axis("off")

    # Angle visualization (masked by skeleton)
    gt_angle_vis = gt_angle * (gt_skel[idx, 0].cpu() > 0.5).float()
    pred_angle_vis = pred_angle * (pred_skel[idx, 0].detach().cpu() > 0.5).float()

    axes[1, 4].imshow(gt_angle_vis, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
    axes[1, 4].set_title("GT Angle")
    axes[1, 4].axis("off")

    axes[1, 5].imshow(pred_angle_vis, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
    axes[1, 5].set_title("Pred Angle")
    axes[1, 5].axis("off")

    plt.tight_layout()
    return fig


def train(args):
    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"structural_{timestamp}"
    writer = SummaryWriter(log_dir)

    # Dataset
    dataset = DenseInkTraceDataset(
        mode="independent",
        img_size=64,
        batch_size=args.batch_size,
        epoch_length=args.epoch_length,
        curriculum_stage=1,  # 从简单开始
        rust_threads=4,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_dense_batch,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Model
    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model = ModelFactory.create_unified_model(
            embed_dim=checkpoint.get("config", {}).get("embed_dim", args.embed_dim),
            num_layers=checkpoint.get("config", {}).get("num_layers", args.num_layers),
            full_heads=False,  # 预训练只需要 skeleton + tangent
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("loss", float("inf"))
        print(f"Resuming from epoch {start_epoch}, best_loss={best_loss:.4f}")
    else:
        model = ModelFactory.create_unified_model(
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            full_heads=False,
            device=device,
        )

    # Masking & Loss
    mask_gen = MaskingGenerator(mask_ratio=args.mask_ratio, strategy=args.mask_strategy)
    criterion = StructuralPretrainLoss()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    if args.resume and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if args.resume and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"\n{'=' * 60}")
    print(f"Structural Pretraining")
    print(f"  Mask ratio: {args.mask_ratio}, Strategy: {args.mask_strategy}")
    print(f"  Model: embed_dim={args.embed_dim}, num_layers={args.num_layers}")
    print(
        f"  Training: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}"
    )
    print(f"{'=' * 60}\n")

    global_step = start_epoch * len(dataloader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {"skeleton": 0, "tangent": 0, "total": 0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            imgs = batch["image"].to(device)
            gt_skel = batch["skeleton"].to(device)
            gt_tan = batch["tangent"].to(device)

            # 1. Generate masked input
            masked_imgs, mask = mask_gen(imgs)
            mask = mask.to(device)

            # 2. Forward (NO SKIPS!)
            outputs = model.pretrain_forward(masked_imgs)
            pred_skel = outputs["skeleton"]
            pred_tan = outputs["tangent"]

            # 3. Loss
            losses = criterion(pred_skel, pred_tan, gt_skel, gt_tan, mask)
            loss = losses["total"]

            # 4. Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Logging
            epoch_losses["skeleton"] += losses["loss_skeleton"].item()
            epoch_losses["tangent"] += losses["loss_tangent"].item()
            epoch_losses["total"] += loss.item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "skel": f"{losses['loss_skeleton'].item():.4f}",
                    "tan": f"{losses['loss_tangent'].item():.4f}",
                }
            )

            # TensorBoard
            if global_step % 100 == 0:
                writer.add_scalar("Loss/total", loss.item(), global_step)
                writer.add_scalar(
                    "Loss/skeleton", losses["loss_skeleton"].item(), global_step
                )
                writer.add_scalar(
                    "Loss/tangent", losses["loss_tangent"].item(), global_step
                )
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

            # Visualize
            if global_step % 500 == 0:
                model.eval()
                with torch.no_grad():
                    fig = visualize_predictions(
                        imgs,
                        masked_imgs,
                        mask,
                        pred_skel,
                        pred_tan,
                        gt_skel,
                        gt_tan,
                        global_step,
                    )
                    writer.add_figure("Predictions", fig, global_step)
                    plt.close(fig)
                model.train()

            global_step += 1

        # Epoch end
        scheduler.step()

        avg_loss = epoch_losses["total"] / num_batches
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Skeleton: {epoch_losses['skeleton'] / num_batches:.4f}")
        print(f"  Tangent: {epoch_losses['tangent'] / num_batches:.4f}")

        # Save checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "config": {
                "embed_dim": args.embed_dim,
                "num_layers": args.num_layers,
                "mask_ratio": args.mask_ratio,
            },
        }

        # Regular checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint_data, save_dir / f"structural_epoch{epoch + 1}.pth")

        # Best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint_data, save_dir / "best_structural.pth")
            print(f"  New best! Saved to best_structural.pth")

    writer.close()
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(
        f"\nNext step: python train_dense.py --init_from {save_dir}/best_structural.pth"
    )


if __name__ == "__main__":
    args = parse_args()
    train(args)
