import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from models import ModelFactory
from losses import DenseLoss
from datasets import DenseInkTraceDataset, collate_dense_batch
from visualize_dense import (
    visualize_prediction,
    create_visualization_grid,
    compute_metrics,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset & Dataloader
    dataset = DenseInkTraceDataset(
        mode="independent",
        img_size=64,
        batch_size=args.batch_size,
        epoch_length=args.epoch_length,
        curriculum_stage=args.stage,
        rust_threads=args.rust_threads,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_dense_batch,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Dataset initialized. Stage: {args.stage}, Workers: {args.num_workers}")

    # 2. Model
    start_epoch = 0
    best_loss = float("inf")

    # 区分两种加载方式:
    # --resume: 完全恢复中断的训练 (epoch + optimizer + scheduler)
    # --init_from: 加载模型权重，从头开始新 stage 训练

    if args.resume:
        # 完全恢复: 用于训练中断后继续
        print(f"Resuming interrupted training from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        # Use config from checkpoint if available, else args
        config = checkpoint.get("config", {})
        embed_dim = config.get("embed_dim", args.embed_dim)
        num_layers = config.get("num_layers", args.num_layers)

        model = ModelFactory.create_unified_model(
            embed_dim=embed_dim, num_layers=num_layers, full_heads=True, device=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"  Continuing from epoch {start_epoch}")
        if "loss" in checkpoint:
            best_loss = checkpoint["loss"]
            print(f"  Previous best loss: {best_loss:.4f}")

    elif args.init_from:
        # 跨 Stage 继续: 加载上一阶段模型 (可能是 structural pretrain)，从 epoch 0 开始新训练
        print(f"Initializing from previous stage: {args.init_from}")
        checkpoint = torch.load(args.init_from, map_location=device)

        config = checkpoint.get("config", {})
        embed_dim = config.get("embed_dim", args.embed_dim)
        num_layers = config.get("num_layers", args.num_layers)

        # Create fresh model with same config
        model = ModelFactory.create_unified_model(
            embed_dim=embed_dim,
            num_layers=num_layers,
            full_heads=True,  # Ensure full heads for dense training
            device=device,
        )

        # Load state dict
        # 如果主要从 structural pretrain 加载，可能只有 encoder 和部分 decoder
        # load_state_dict(..., strict=False) 这是一个好主意，因为 pretrain 没有 width/offset head
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"  Loaded model weights (strict=False)")
        except RuntimeError as e:
            print(f"  Warning: strict load failed, trying looser load: {e}")
            # Try loading encoder only if full load fails massively (fallback)
            model.encoder.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )  # this is risky if keys don't match

        print(f"  Loaded weights from {args.init_from}")
        # start_epoch = 0, best_loss = inf (重置)

    else:
        # 从头开始，可选加载 encoder 预训练权重
        model = ModelFactory.create_unified_model(
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            full_heads=True,
            device=device,
            encoder_ckpt=args.encoder_ckpt,
        )

    # Freeze Encoder if requested
    if args.freeze_encoder:
        print("Freezing Encoder...")
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()

    # 3. Loss & Optimizer
    criterion = DenseLoss().to(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # IterableDataset doesn't have accurate len, compute steps manually
    # Use ceil division to account for the last partial batch
    steps_per_epoch = (args.epoch_length + args.batch_size - 1) // args.batch_size
    # Calculate remaining steps if resuming
    remaining_epochs = args.epochs - start_epoch
    # Add buffer to avoid "Tried to step X times" error if dataloader yields extra batches
    total_steps = int(steps_per_epoch * remaining_epochs * 1.05) + 500

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps
    )

    # Restore optimizer/scheduler state if fully resuming
    if args.resume and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("  Restored optimizer state")
        except:
            print("  Warning: Could not restore optimizer state")

    # 4. Training Loop
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "vis"), exist_ok=True)

    # TensorBoard setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"dense_{timestamp}_stage{args.stage}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    # Log hyperparameters
    writer.add_text(
        "hyperparameters",
        f"""
    - lr: {args.lr}
    - batch_size: {args.batch_size}
    - epochs: {args.epochs}
    - epoch_length: {args.epoch_length}
    - stage: {args.stage}
    - freeze_encoder: {args.freeze_encoder}
    - encoder_ckpt: {args.encoder_ckpt}
    - resume: {args.resume}
    """,
    )

    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()  # Keep encoder in eval mode

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        epoch_losses = {
            "total": 0,
            "skel": 0,
            "junc": 0,
            "tan": 0,
            "width": 0,
            "off": 0,
        }

        step_cnt = 0
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            # Targets to device
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()

            # Use device-agnostic autocast
            amp_enabled = device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(imgs)
                losses = criterion(outputs, targets)
                loss = losses["total"]

            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ NaN/Inf loss detected, skipping batch")
                continue

            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Logging
            loss_val = loss.item()
            epoch_losses["total"] += loss_val
            epoch_losses["skel"] += losses["loss_skel"].item()
            epoch_losses["junc"] += losses["loss_junc"].item()
            epoch_losses["tan"] += losses["loss_tan"].item()
            epoch_losses["width"] += losses["loss_width"].item()
            epoch_losses["off"] += losses["loss_off"].item()
            step_cnt += 1
            global_step += 1

            # TensorBoard: Log step losses (every 10 steps to reduce overhead)
            if global_step % 10 == 0:
                writer.add_scalar("Loss/step_total", loss_val, global_step)
                writer.add_scalar(
                    "Loss/step_skel", losses["loss_skel"].item(), global_step
                )
                writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix(loss=loss_val, skel=losses["loss_skel"].item())

        # Average losses
        avg_losses = {k: v / step_cnt for k, v in epoch_losses.items()}
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_losses['total']:.4f}")

        # TensorBoard: Log epoch losses
        writer.add_scalar("Loss/epoch_total", avg_losses["total"], epoch)
        writer.add_scalar("Loss/epoch_skel", avg_losses["skel"], epoch)
        writer.add_scalar("Loss/epoch_junc", avg_losses["junc"], epoch)
        writer.add_scalar("Loss/epoch_tan", avg_losses["tan"], epoch)
        writer.add_scalar("Loss/epoch_width", avg_losses["width"], epoch)
        writer.add_scalar("Loss/epoch_off", avg_losses["off"], epoch)

        # Save Checkpoint
        if avg_losses["total"] < best_loss:
            best_loss = avg_losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                os.path.join(args.save_dir, "best_dense_model.pth"),
            )
            print("Saved Best Model.")

        # Evaluation & Visualization (Every N epochs)
        if (epoch + 1) % args.vis_interval == 0 or epoch == 0:
            print(f"Running evaluation at epoch {epoch + 1}...")
            model.eval()

            # Generate fresh test batch for evaluation
            test_imgs, test_targets = next(iter(dataloader))
            test_imgs = test_imgs.to(device)
            test_targets = {k: v.to(device) for k, v in test_targets.items()}

            with torch.no_grad():
                test_outputs = model(test_imgs)

                # Compute metrics
                metrics = compute_metrics(test_outputs, test_targets)

                # TensorBoard: Log metrics
                for name, value in metrics.items():
                    writer.add_scalar(f"Metrics/{name}", value, epoch)

                print(
                    f"  Metrics: IoU={metrics['skel_iou']:.4f}, F1={metrics['skel_f1']:.4f}, "
                    f"Precision={metrics['skel_precision']:.4f}, Recall={metrics['skel_recall']:.4f}"
                )

                # TensorBoard: Log visualization grid
                vis_grid = create_visualization_grid(
                    test_imgs,
                    {k: v.detach() for k, v in test_outputs.items()},
                    {k: v.detach() for k, v in test_targets.items()},
                    num_samples=min(4, args.batch_size),
                )
                writer.add_image("Predictions", vis_grid, epoch, dataformats="HWC")

                # Also save to file
                visualize_prediction(
                    test_imgs[:1],
                    {k: v[:1].detach() for k, v in test_outputs.items()},
                    {k: v[:1].detach() for k, v in test_targets.items()},
                    os.path.join(args.save_dir, "vis", f"epoch_{epoch + 1}.png"),
                )

    # Close TensorBoard writer
    writer.close()
    print(f"Training finished. TensorBoard logs saved to: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epoch_length", type=int, default=10000)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="checkpoints_dense")
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help="Path to pretrained encoder checkpoint",
    )
    parser.add_argument(
        "--freeze_encoder", action="store_true", help="Freeze encoder weights"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume interrupted training (restores epoch, optimizer)",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Initialize from previous stage checkpoint (fresh training)",
    )
    parser.add_argument(
        "--vis_interval", type=int, default=2, help="Visualization interval (epochs)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader num_workers"
    )
    parser.add_argument(
        "--rust_threads", type=int, default=4, help="Rust data generation threads"
    )
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)

    args = parser.parse_args()
    train(args)
