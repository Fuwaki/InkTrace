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
from encoder import StrokeEncoder
from dense_models import DenseVectorNet, DenseLoss
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
        rust_threads=4,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_dense_batch,
        pin_memory=True,
    )

    print(f"Dataset initialized. Stage: {args.stage}")

    # 2. Model
    encoder = StrokeEncoder(in_channels=1, embed_dim=128)

    # Load Pretrained Encoder if needed
    if args.encoder_ckpt:
        print(f"Loading pretrained encoder from {args.encoder_ckpt}")
        ckpt = torch.load(args.encoder_ckpt, map_location="cpu")

        # Handle state dict keys
        if "encoder_state_dict" in ckpt:
            state_dict = ckpt["encoder_state_dict"]
        elif "model_state_dict" in ckpt:
            # Try to find encoder part if it's a full model
            state_dict = {
                k.replace("encoder.", ""): v
                for k, v in ckpt["model_state_dict"].items()
                if k.startswith("encoder.")
            }
            if not state_dict:
                state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # 直接加载新架构权重 (需要先用 convert_checkpoint.py 转换旧权重)
        msg = encoder.load_state_dict(state_dict, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            print(f"  Warning: Missing keys: {msg.missing_keys}")
            print(f"  Warning: Unexpected keys: {msg.unexpected_keys}")
            print("  如果看到 stem.* 相关的 unexpected keys，请先运行:")
            print(
                f"    python convert_checkpoint.py --input {args.encoder_ckpt} --output <new_path>"
            )
        else:
            print("  Encoder weights loaded successfully.")

    model = DenseVectorNet(encoder).to(device)

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
    steps_per_epoch = args.epoch_length // args.batch_size
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs
    )

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
    """,
    )

    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
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

            loss.backward()
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
    parser.add_argument("--epoch_length", type=int, default=1000)
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
        "--vis_interval", type=int, default=2, help="Visualization interval (epochs)"
    )

    args = parser.parse_args()
    train(args)
