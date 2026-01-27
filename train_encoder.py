#!/usr/bin/env python3
"""
统一的 Encoder 训练脚本

支持多阶段渐进式训练：
  - phase1:   单一贝塞尔曲线重建 (基础笔画理解)
  - phase1.5: 连续多段贝塞尔曲线重建 (连笔理解)
  - phase1.6: 独立多笔画重建 (复杂布局理解)
  - phase1.7: 多路径连接曲线重建 (文档级理解)

使用方法:
  python train_encoder.py --phase 1 --from-scratch
  python train_encoder.py --phase 1.5 --resume best_reconstruction.pth
  python train_encoder.py --phase 1.6 --resume best_reconstruction_multi.pth
  python train_encoder.py --phase 1.7 --resume best_reconstruction_independent.pth

特性:
  - 自动设备选择: CUDA > XPU > CPU
  - 支持从零开始或加载检查点继续训练
  - 使用 Rust 高性能数据生成后端
  - TensorBoard 日志记录
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

from models import ReconstructionModel
from datasets import InkTraceDataset


# ==================== 配置定义 ====================

PHASE_CONFIGS = {
    "1": {
        "name": "Phase 1: 单曲线重建",
        "mode": "single",
        "checkpoint_name": "best_reconstruction.pth",
        "default_epochs": 50,
        "default_lr": 1e-3,
        "default_batch_size": 32,
        "dataset_params": {},
    },
    "1.5": {
        "name": "Phase 1.5: 连续多段曲线重建",
        "mode": "continuous",
        "checkpoint_name": "best_reconstruction_multi.pth",
        "default_epochs": 30,
        "default_lr": 5e-4,
        "default_batch_size": 32,
        "dataset_params": {
            "max_segments": 8,
        },
    },
    "1.6": {
        "name": "Phase 1.6: 独立多笔画重建",
        "mode": "independent",
        "checkpoint_name": "best_reconstruction_independent.pth",
        "default_epochs": 30,
        "default_lr": 5e-4,
        "default_batch_size": 32,
        "dataset_params": {
            "max_strokes": 8,
        },
    },
    "1.7": {
        "name": "Phase 1.7: 多路径连接曲线重建",
        "mode": "multi_connected",
        "checkpoint_name": "best_reconstruction_multipath.pth",
        "default_epochs": 30,
        "default_lr": 5e-4,
        "default_batch_size": 32,
        "dataset_params": {
            "max_paths": 4,
            "max_segments": 6,
        },
    },
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="统一的 Encoder 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从零开始训练 Phase 1
  python train_encoder.py --phase 1 --from-scratch

  # 从 Phase 1 检查点继续训练 Phase 1.5
  python train_encoder.py --phase 1.5 --resume best_reconstruction.pth

  # 自定义参数训练
  python train_encoder.py --phase 1 --epochs 100 --lr 1e-4 --batch-size 64

  # 指定设备
  python train_encoder.py --phase 1 --device cuda:0
        """,
    )

    # 必须参数
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["1", "1.5", "1.6", "1.7"],
        help="训练阶段: 1, 1.5, 1.6, 1.7",
    )

    # 检查点参数 (互斥组)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--from-scratch",
        action="store_true",
        help="从零开始训练",
    )
    checkpoint_group.add_argument(
        "--resume",
        type=str,
        metavar="PATH",
        help="从指定检查点恢复训练",
    )

    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数 (默认根据阶段自动设置)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="学习率 (默认根据阶段自动设置)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批次大小 (默认根据阶段自动设置)",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=20000,
        help="每个 Epoch 的数据量 (默认: 20000)",
    )

    # 模型参数
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Embedding 维度 (默认: 128)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Transformer 注意力头数 (默认: 4)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Transformer 层数 (默认: 6)",
    )

    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备 (默认: 自动选择 cuda > xpu > cpu)",
    )

    # DataLoader 参数
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers 数量 (默认: 4)",
    )

    # 日志参数
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="TensorBoard 日志目录 (默认: runs)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="日志打印间隔 (batch 数, 默认: 100)",
    )

    # 其他参数
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="检查点保存间隔 (epoch 数, 默认: 5)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="禁用训练结束后的可视化",
    )

    return parser.parse_args()


def get_device(device_arg=None):
    """
    获取训练设备

    优先级: 用户指定 > CUDA > XPU > CPU
    """
    if device_arg is not None:
        device = torch.device(device_arg)
        print(f"使用指定设备: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"使用 XPU 设备")
    else:
        device = torch.device("cpu")
        print(f"使用 CPU 设备")

    return device


def create_model(embed_dim, num_heads, num_layers, device):
    """创建模型"""
    from encoder import StrokeEncoder
    from pixel_decoder import PixelDecoder

    encoder = StrokeEncoder(
        in_channels=1,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
    ).to(device)

    pixel_decoder = PixelDecoder(embed_dim=embed_dim).to(device)

    model = ReconstructionModel(encoder, pixel_decoder)
    return model


def load_checkpoint(model, checkpoint_path, device, load_optimizer=False, optimizer=None):
    """
    加载检查点

    Returns:
        start_epoch: 起始 epoch
        best_loss: 最佳损失
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

    print(f"\n加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载 encoder 权重
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    print(f"  ✓ 加载 Encoder 权重")

    # 加载 decoder 权重 (兼容不同的 key 名)
    if "decoder_state_dict" in checkpoint:
        model.pixel_decoder.load_state_dict(checkpoint["decoder_state_dict"])
        print(f"  ✓ 加载 Pixel Decoder 权重")
    elif "pixel_decoder_state_dict" in checkpoint:
        model.pixel_decoder.load_state_dict(checkpoint["pixel_decoder_state_dict"])
        print(f"  ✓ 加载 Pixel Decoder 权重")

    # 加载优化器状态 (可选)
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"  ✓ 加载优化器状态")

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    print(f"  Epoch: {epoch}")
    print(f"  Loss: {loss:.6f}")

    return epoch, loss


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": model.encoder.state_dict(),
        "decoder_state_dict": model.pixel_decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, save_path)


def create_dataset(phase_config, dataset_size, batch_size, num_workers):
    """创建数据集和 DataLoader"""
    mode = phase_config["mode"]
    dataset_params = phase_config["dataset_params"].copy()

    print(f"\n创建数据集...")
    print(f"  模式: {mode}")
    print(f"  Epoch 大小: {dataset_size}")
    for k, v in dataset_params.items():
        print(f"  {k}: {v}")

    dataset = InkTraceDataset(
        mode=mode,
        img_size=64,
        batch_size=batch_size,  # Rust 内部 batch
        epoch_length=dataset_size,
        **dataset_params,
    )

    # 使用 IterableDataset 时，shuffle 在内部处理
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, dataloader


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, log_interval, writer=None):
    """训练一个 epoch"""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)

        optimizer.zero_grad()

        # 前向传播
        reconstructed, embeddings = model(imgs)

        # 计算损失
        loss = criterion(reconstructed, imgs)

        # 反向传播
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        # 日志
        if (batch_idx + 1) % log_interval == 0:
            print(f"  Batch [{batch_idx + 1}], Loss: {loss.item():.6f}")

        # TensorBoard
        if writer is not None:
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar("Loss/batch", loss.item(), global_step)

    avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
    return avg_loss


def visualize_reconstruction(model, dataset, device, save_path, num_samples=6):
    """可视化重建效果"""
    model.eval()

    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    fig.suptitle("Reconstruction Results", fontsize=14)

    # 从数据集获取样本
    samples = []
    sample_iter = iter(dataset)
    for _ in range(num_samples):
        try:
            img, _ = next(sample_iter)
            samples.append(img)
        except StopIteration:
            break

    if len(samples) == 0:
        print("  ✗ 无法获取样本进行可视化")
        return

    with torch.no_grad():
        for i, img_tensor in enumerate(samples):
            if i >= num_samples:
                break

            # 原图
            img = img_tensor.squeeze().numpy() * 255

            # 重建
            img_batch = img_tensor.unsqueeze(0).to(device)
            reconstructed, _ = model(img_batch)
            recon_img = reconstructed.cpu().squeeze().numpy() * 255

            # 计算差异
            diff = np.abs(img - recon_img).mean()

            # 显示原图
            axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            # 显示重建图
            axes[1, i].imshow(recon_img, cmap="gray", vmin=0, vmax=255)
            axes[1, i].set_title(f"Recon (diff={diff:.1f})")
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ 保存可视化图像: {save_path}")
    plt.close()


def train(args):
    """主训练函数"""
    # 获取阶段配置
    phase_config = PHASE_CONFIGS[args.phase]
    print("\n" + "=" * 60)
    print(f"  {phase_config['name']}")
    print("=" * 60)

    # 设置参数 (命令行 > 默认配置)
    epochs = args.epochs or phase_config["default_epochs"]
    lr = args.lr or phase_config["default_lr"]
    batch_size = args.batch_size or phase_config["default_batch_size"]

    print(f"\n训练配置:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Dataset Size: {args.dataset_size}")

    # 获取设备
    device = get_device(args.device)

    # 创建模型
    print(f"\n创建模型...")
    model = create_model(args.embed_dim, args.num_heads, args.num_layers, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 加载检查点 (如果指定)
    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        load_checkpoint(model, args.resume, device)
        # 注意：继续训练时，从 epoch 0 重新开始计数
        # 但权重是从检查点加载的

    # 创建数据集
    dataset, dataloader = create_dataset(
        phase_config, args.dataset_size, batch_size, args.num_workers
    )

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"phase{args.phase}_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard 日志: {log_dir}")

    # 检查点保存路径
    checkpoint_path = phase_config["checkpoint_name"]
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # 训练循环
    print("\n" + "-" * 60)
    print("开始训练...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # 训练一个 epoch
        avg_loss = train_epoch(
            model, dataloader, criterion, optimizer, device,
            epoch, args.log_interval, writer
        )

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # TensorBoard 记录
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_path)
            print(f"  ✓ 保存最佳模型 -> {checkpoint_path}")

        # 定期保存检查点
        if epoch % args.save_interval == 0:
            periodic_path = checkpoint_dir / f"phase{args.phase}_epoch{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, periodic_path)

        # 打印 epoch 统计
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  最佳损失: {best_loss:.6f}")
        print(f"  学习率: {current_lr:.6f}")
        print("-" * 60)

    writer.close()

    # 训练完成
    print("\n" + "=" * 60)
    print(f"{phase_config['name']} 训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型已保存: {checkpoint_path}")
    print("=" * 60)

    # 可视化
    if not args.no_visualize:
        print("\n生成可视化...")
        vis_path = f"reconstruction_phase{args.phase}.png"
        # 创建新的数据集用于可视化 (避免迭代器耗尽问题)
        vis_dataset = InkTraceDataset(
            mode=phase_config["mode"],
            img_size=64,
            batch_size=16,
            epoch_length=100,
            **phase_config["dataset_params"],
        )
        visualize_reconstruction(model, vis_dataset, device, vis_path)

    return model


def main():
    """入口函数"""
    args = parse_args()

    print("\n" + "=" * 60)
    print("  InkTrace Encoder Training")
    print("  手写文字高保真矢量化 - Encoder 训练")
    print("=" * 60)

    try:
        train(args)
        print("\n训练成功完成!")
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练出错: {e}")
        raise


if __name__ == "__main__":
    main()
