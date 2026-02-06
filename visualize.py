#!/usr/bin/env python3
"""
离线可视化 CLI 工具 (Offline Adapter)

职责：
- 命令行工具，用于离线加载训练好的模型
- 生成高清对比图用于分析
- 计算评估指标

使用方法：
    # 基本用法
    python visualize.py --ckpt checkpoints/dense/last.ckpt --output results/

    # 指定样本数和 curriculum stage
    python visualize.py --ckpt best.ckpt --stage 5 --samples 8 --output analysis/

    # 计算统计指标
    python visualize.py --ckpt best.ckpt --stats-samples 100

    # 生成单样本详细可视化
    python visualize.py --ckpt best.ckpt --samples 4 --detailed
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# 导入 Lightning 组件
from lightning_model import UnifiedTask
from lightning_data import InkTraceDataModule

# 导入核心渲染层
from vis_core import create_grid_image, visualize_sample, compute_metrics


def load_model_from_checkpoint(
    checkpoint_path: str,
    stage: str = "dense",
    device: str = "auto",
) -> UnifiedTask:
    """
    从 checkpoint 加载模型

    Args:
        checkpoint_path: checkpoint 文件路径
        stage: 训练阶段 ("structural" 或 "dense")
        device: 设备 ("auto", "cuda", "cpu")

    Returns:
        加载好的模型 (eval 模式)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    # 确定设备
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "xpu") and torch.backends.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # 使用 Lightning 的 load_from_checkpoint
    # 这会自动处理超参数和模型架构
    try:
        model = UnifiedTask.load_from_checkpoint_with_stage(
            checkpoint_path=checkpoint_path,
            stage=stage,
            strict=False,  # 允许部分加载 (用于迁移学习)
        )
    except Exception as e:
        print(f"Warning: Failed to load with stage, trying generic load: {e}")
        model = UnifiedTask.load_from_checkpoint(
            checkpoint_path,
            strict=False,
        )

    model.eval()
    model.to(device)

    print("✓ Model loaded successfully")
    return model


def compute_statistics(
    model: UnifiedTask,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int,
) -> dict:
    """
    计算批量统计指标

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_batches: 批次数

    Returns:
        平均指标字典
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

    # 计算均值和标准差
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(
        description="InkTrace 离线可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必需参数
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="模型 checkpoint 路径",
    )

    # 输出配置
    parser.add_argument(
        "--output",
        type=str,
        default="results_vis",
        help="输出目录",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="可视化样本数",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="生成单样本详细可视化 (4x5 网格)",
    )

    # 数据配置
    parser.add_argument(
        "--stage",
        type=int,
        default=2,
        help="Curriculum stage (0-9)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=64,
        help="图像尺寸",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )

    # 统计指标
    parser.add_argument(
        "--stats-samples",
        type=int,
        default=0,
        help="统计指标样本数 (0=跳过)",
    )

    # 设备
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "xpu"],
        help="设备",
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 加载模型
    model = load_model_from_checkpoint(args.ckpt, device=args.device)
    device = next(model.parameters()).device

    # 创建 DataModule
    print(f"\nCreating datamodule (curriculum stage={args.stage})...")
    datamodule = InkTraceDataModule(
        img_size=args.img_size,
        batch_size=args.batch_size,
        epoch_length=max(args.samples, args.stats_samples, 100),
        curriculum_stage=args.stage,
        num_workers=0,  # 离线评估通常不需要多进程
    )
    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()

    # 1. 生成网格可视化
    if args.samples > 0:
        print(f"\nGenerating visualization ({args.samples} samples)...")

        # 获取一个 batch
        imgs, targets = next(iter(train_loader))
        imgs = imgs[: args.samples].to(device)
        targets = {k: v[: args.samples].to(device) for k, v in targets.items()}

        # 推理
        with torch.no_grad():
            outputs = model(imgs)

        # 生成网格图
        grid = create_grid_image(
            imgs,
            outputs,
            targets,
            num_samples=min(args.samples, 4),
        )

        # 保存
        grid_path = output_dir / "vis_grid.png"
        Image.fromarray(grid).save(grid_path)
        print(f"  ✓ Saved: {grid_path}")

        # 生成单样本详细图
        if args.detailed:
            for i in range(min(args.samples, 4)):
                sample_img = imgs[i : i + 1]
                sample_pred = {k: v[i : i + 1] for k, v in outputs.items()}
                sample_tgt = {k: v[i : i + 1] for k, v in targets.items()}

                detailed = visualize_sample(sample_img, sample_pred, sample_tgt)

                sample_path = output_dir / f"vis_sample_{i:02d}_detailed.png"
                Image.fromarray(detailed).save(sample_path)

            print(f"  ✓ Saved {min(args.samples, 4)} detailed samples")

    # 2. 计算统计指标
    if args.stats_samples > 0:
        print(f"\nComputing statistics ({args.stats_samples} samples)...")

        num_batches = (args.stats_samples + args.batch_size - 1) // args.batch_size
        metrics = compute_statistics(model, train_loader, device, num_batches)

        print("\n" + "=" * 50)
        print("EVALUATION METRICS")
        print("=" * 50)
        print(f"  Skeleton IoU:       {metrics['skel_iou']:.4f} ± {metrics['skel_iou_std']:.4f}")
        print(f"  Skeleton Precision: {metrics['skel_precision']:.4f} ± {metrics['skel_precision_std']:.4f}")
        print(f"  Skeleton Recall:    {metrics['skel_recall']:.4f} ± {metrics['skel_recall_std']:.4f}")
        print(f"  Skeleton F1:        {metrics['skel_f1']:.4f} ± {metrics['skel_f1_std']:.4f}")
        print(f"  KP Topo Recall:     {metrics['kp_topo_recall']:.4f} ± {metrics['kp_topo_recall_std']:.4f}")
        print(f"  KP Geo Recall:      {metrics['kp_geo_recall']:.4f} ± {metrics['kp_geo_recall_std']:.4f}")

        # 额外的统计信息（如果存在）
        if 'skel_tp' in metrics:
            print(f"\n  Confusion Stats (avg):")
            print(f"    True Positives:  {metrics['skel_tp']:.0f}")
            print(f"    False Positives: {metrics['skel_fp']:.0f}")
            print(f"    False Negatives: {metrics['skel_fn']:.0f}")

        print("=" * 50)

        # 保存到文件
        metrics_path = output_dir / "metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("EVALUATION METRICS\n")
            f.write("=" * 50 + "\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.6f}\n")
        print(f"  ✓ Saved metrics to: {metrics_path}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
