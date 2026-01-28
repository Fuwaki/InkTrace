#!/usr/bin/env python3
"""
统一的 DETR 矢量化训练脚本 (V3 训练优化版)

支持多阶段渐进式训练：
  - phase2:   独立笔画矢量化 (简单)
  - phase2.5: 连续笔画矢量化 (引入拓扑关系)
  - phase2.6: 复杂布局矢量化 (端到端完全体)

改进特性:
- 辅助损失 (Aux Loss)
- 笔画状态监督 (New/Continue/Null)
- 课程学习 (简单 -> 困难)
- 梯度裁剪
- 统一的训练接口

使用方法:
  python train_detr.py --phase 2 --from-scratch
  python train_detr.py --phase 2.5 --resume best_detr_independent.pth
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

from models import ModelFactory, VectorizationModel
from datasets import InkTraceDataset
from losses import DETRLoss


# ==================== 配置定义 ====================

PHASE_CONFIGS = {
    "2": {
        "name": "Phase 2: 独立多笔画矢量化",
        "mode": "independent",
        "checkpoint_name": "best_detr_independent.pth",
        "default_epochs": 100,
        "default_lr": 1e-4,
        "default_batch_size": 256,
        "dataset_params": {
            "max_strokes": 8,
        },
    },
    "2.5": {
        "name": "Phase 2.5: 连续笔画矢量化 (拓扑关系)",
        "mode": "continuous",
        "checkpoint_name": "best_detr_continuous.pth",
        "default_epochs": 50,
        "default_lr": 5e-5,
        "default_batch_size": 128,
        "dataset_params": {
            "max_segments": 8,
        },
    },
    "2.6": {
        "name": "Phase 2.6: 复杂文档矢量化 (Mix)",
        "mode": "mixed",
        "checkpoint_name": "best_detr_final.pth",
        "default_epochs": 50,
        "default_lr": 1e-5,
        "default_batch_size": 128,
        "dataset_params": {
            "configs": [
                ({"mode": "independent", "max_strokes": 8}, 0.5),
                ({"mode": "continuous", "max_segments": 8}, 0.5),
            ]
        },
    },
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="统一的 DETR 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必须参数
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["2", "2.5", "2.6"],
        help="训练阶段: 2, 2.5, 2.6",
    )

    # 预设配置
    parser.add_argument(
        "--profile",
        type=str,
        choices=["default", "rtx3060", "rtx5090", "tpu"],
        default="default",
        help="硬件配置预设",
    )

    # 检查点参数
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--from-scratch",
        action="store_true",
        help="从零开始训练 (Encoder 仍会尝试加载 Phase 1.x)",
    )
    checkpoint_group.add_argument(
        "--resume",
        type=str,
        metavar="PATH",
        help="从指定检查点恢复训练",
    )

    parser.add_argument(
        "--pretrained-encoder",
        type=str,
        default="best_reconstruction_independent.pth",
        help="Phase 1 预训练 Encoder 路径",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataset-size", type=int, default=50000)

    # 模型参数
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)  # Decoder layers

    # 训练技巧参数
    parser.add_argument("--grad-clip", type=float, default=0.1, help="梯度裁剪阈值")
    parser.add_argument("--warmup-epochs", type=int, default=5)

    # Loss 权重
    parser.add_argument("--coord-weight", type=float, default=5.0)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=1.0)

    # 系统参数
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rust-threads", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="runs_detr")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--no-visualize", action="store_true")

    return parser.parse_args()


def get_device(device_arg=None):
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")


def create_model(args, device):
    """创建矢量化模型"""
    print(f"\n创建模型...")
    # 使用 Factory 创建，它会自动处理 Encoder/Decoder 的初始化
    # 注意：这里我们使用 V3 的参数
    model = ModelFactory.create_vectorization_model(
        embed_dim=args.embed_dim,
        num_slots=8,  # 固定为 8 (最大支持)
        device=device,
        include_pixel_decoder=True,
    )

    # 尝试加载预训练 Encoder
    # 只有在从零开始训练 DETR (Phase 2) 时才需要加载预训练 Encoder
    # 如果是 Resume (Phase 2.5/2.6)，则会加载整个 checkpoint
    if args.from_scratch:
        # 1. 确定要加载的文件路径
        encoder_path = args.pretrained_encoder

        # 如果指定的文件不存在，尝试自动搜索其他可能的候选
        if not os.path.exists(encoder_path):
            candidates = [
                "best_reconstruction_final.pth",
                "best_reconstruction_multipath.pth",
                "best_reconstruction_independent.pth",
                "best_reconstruction.pth",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    print(
                        f"  提示: 指定的 Encoder {encoder_path} 不存在，自动切换为: {cand}"
                    )
                    encoder_path = cand
                    break

        if os.path.exists(encoder_path):
            print(f"  正在加载预训练 Encoder: {encoder_path}")
            try:
                checkpoint = torch.load(encoder_path, map_location=device)

                # 检查 Keys
                if "encoder_state_dict" in checkpoint:
                    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
                    print("  ✓ Encoder 权重加载成功")
                else:
                    print("  ⚠ Checkpoint 中未找到 'encoder_state_dict'")

                # 可选：加载 Pixel Decoder (虽然 DETR 训练主要不用它，但保持参数一致有好处)
                if "decoder_state_dict" in checkpoint:
                    model.pixel_decoder.load_state_dict(
                        checkpoint["decoder_state_dict"]
                    )
                    print("  ✓ Pixel Decoder 权重加载成功")

                print("  (说明: Encoder 使用较小的学习率进行微调)")
            except Exception as e:
                print(f"  ⚠ Encoder 加载出错: {e}")
                print("  ⚠ 警告: 将使用随机初始化的 Encoder 进行训练 (极难收敛!)")
        else:
            print(
                f"  ⚠ 警告: 未找到任何预训练 Encoder 权重 ({args.pretrained_encoder})"
            )
            print("  ⚠ 警告: 将使用随机初始化的 Encoder 进行训练 (极难收敛!)")
            print("  强烈建议先运行 train_encoder.py 获取预训练模型")

    return model


def create_dataset(args, phase_config):
    """创建数据集"""
    from datasets import MixedInkTraceDataset

    mode = phase_config["mode"]
    dataset_params = phase_config["dataset_params"].copy()

    print(f"\n创建数据集 (Mode: {mode})...")

    if mode == "mixed":
        configs = dataset_params.pop("configs")
        dataset = MixedInkTraceDataset(
            configs=configs,
            epoch_length=args.dataset_size,
            batch_size=args.batch_size,
            rust_threads=args.rust_threads,
            for_detr=True,  # 启用 DETR 标签格式 (New/Continue)
        )
    else:
        dataset = InkTraceDataset(
            mode=mode,
            img_size=64,
            batch_size=args.batch_size,
            epoch_length=args.dataset_size,
            rust_threads=args.rust_threads,
            for_detr=True,  # 启用 DETR 标签格式 (New/Continue)
            **dataset_params,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return dataset, dataloader


def train_epoch(
    model, dataloader, criterion, optimizer, device, epoch, args, writer=None
):
    model.train()
    epoch_loss = 0.0
    epoch_loss_dict = {}
    num_batches = 0

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        # 使用 non_blocking=True 加速数据传输
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 混合数据集可能导致 targets 形状不一致
        # 需要确保 targets 是 [B, 8, 11]
        
        optimizer.zero_grad()

        # 前向传播
        outputs = model(imgs, mode="vectorize")

        # 计算 Loss (Hungarian Matcher 在 CPU 上运行，这是主要瓶颈)
        loss, loss_dict = criterion(outputs, targets)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # 累积统计 (注意: loss.item() 会触发同步，但对于打印日志是必须的)
        current_loss = loss.item()
        epoch_loss += current_loss
        num_batches += 1

        for k, v in loss_dict.items():
            epoch_loss_dict[k] = epoch_loss_dict.get(k, 0.0) + v

        # Logging & TensorBoard
        # 将 TensorBoard 写入与打印日志同步，或降低频率，避免频繁 I/O 阻塞高速训练
        # train_encoder 虽然每步都写，但因为计算量极小所以不明显。DETR 建议适当降低频率。
        if (batch_idx + 1) % args.log_interval == 0:
            # Print
            print(
                f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {current_loss:.4f} "
                f"Class: {loss_dict.get('class', 0):.4f} "
                f"Coord: {loss_dict.get('coord', 0):.4f}"
            )
            
            # TensorBoard (移到这里或每N步写一次)
            if writer:
                global_step = (epoch - 1) * len(dataloader) + batch_idx
                writer.add_scalar("Loss/batch", current_loss, global_step)
                for k, v in loss_dict.items():
                    writer.add_scalar(f"LossComponent/{k}", v, global_step)
                # 强制刷新，确保用户能立即看到
                writer.flush()
        
        # 为了更平滑的曲线，也可以选择每 10 步写一次 TB，但不打印
        elif writer and (batch_idx + 1) % 10 == 0:
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar("Loss/batch", current_loss, global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"LossComponent/{k}", v, global_step)

    # Average
    avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
    for k in epoch_loss_dict:
        epoch_loss_dict[k] /= num_batches

    return avg_loss, epoch_loss_dict


def train(args):
    # 应用 Profile
    if args.profile == "rtx5090":
        if args.batch_size is None:
            args.batch_size = 512
        if args.dataset_size == 50000:
            args.dataset_size = 200000
    elif args.profile == "rtx3060":
        if args.batch_size is None:
            args.batch_size = 128

    phase_config = PHASE_CONFIGS[args.phase]
    print(f"\n" + "=" * 60)
    print(f"  {phase_config['name']}")
    print(f"  Profile: {args.profile}")
    print("=" * 60)

    # 参数设置
    epochs = args.epochs or phase_config["default_epochs"]
    batch_size = args.batch_size or phase_config["default_batch_size"]
    lr = args.lr or phase_config["default_lr"]

    device = get_device(args.device)

    # 模型
    model = create_model(args, device)

    # 尝试 Resume
    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        if os.path.exists(args.resume):
            print(f"  Resume from: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            # 加载权重
            # 注意 Key 可能不匹配，需要灵活处理
            if "detr_decoder_state_dict" in ckpt:
                model.detr_decoder.load_state_dict(ckpt["detr_decoder_state_dict"])
            if "encoder_state_dict" in ckpt:
                model.encoder.load_state_dict(ckpt["encoder_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss = ckpt.get("loss", float("inf"))
        else:
            print(f"  ⚠ 检查点不存在: {args.resume}, 从头开始")

    # 数据集
    dataset, dataloader = create_dataset(args, phase_config)

    # Loss
    criterion = DETRLoss(
        coord_weight=args.coord_weight,
        class_weight=args.class_weight,
        aux_weight=args.aux_weight,
        p0_match_weight=2.0,  # 增强起点匹配
    ).to(device)

    # 优化器 (分层 LR)
    # 策略：
    # 1. 收集 Encoder 参数 (LR * 0.1)
    # 2. 收集其余所有参数 (Decoder, PixelDecoder, Adapter 等) (LR)

    encoder_params = []
    encoder_param_ids = set()

    for n, p in model.encoder.named_parameters():
        if p.requires_grad:
            encoder_params.append(p)
            encoder_param_ids.add(id(p))

    rest_params = []
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in encoder_param_ids:
            rest_params.append(p)

    param_dicts = [
        {"params": encoder_params, "lr": lr * 0.1},
        {"params": rest_params, "lr": lr},
    ]

    optimizer = optim.AdamW(param_dicts, weight_decay=1e-4)

    # Scheduler
    # Warmup + Cosine
    def lr_lambda(current_step):
        # Epoch based
        if current_step < args.warmup_epochs:
            return float(current_step + 1) / float(max(1, args.warmup_epochs))
        progress = float(current_step - args.warmup_epochs) / float(
            max(1, epochs - args.warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"phase{args.phase}_{timestamp}"
    writer = SummaryWriter(log_dir)

    # 训练循环
    print("\nStarting Training...")
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    save_path = phase_config["checkpoint_name"]

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs} | LR: {optimizer.param_groups[1]['lr']:.2e}")

        avg_loss, loss_dict = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch, args, writer
        )

        scheduler.step()

        # Log
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        print(f"  Epoch Loss: {avg_loss:.6f}")
        for k, v in loss_dict.items():
            print(f"    {k}: {v:.6f}")

        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ModelFactory.save_model(model, save_path, epoch, best_loss, optimizer)
            print(f"  ★ New Best Model Saved: {save_path}")

        # Periodic Save
        if epoch % args.save_interval == 0:
            p_path = checkpoint_dir / f"detr_phase{args.phase}_epoch{epoch}.pth"
            ModelFactory.save_model(model, p_path, epoch, avg_loss, optimizer)

    writer.close()
    print("\nTraining Finished.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
