#!/usr/bin/env python3
"""
统一的重建可视化脚本

用于评估和可视化各个阶段的训练模型重建效果。
支持对 Phase 1, 1.5, 1.6, 1.7 的模型进行可视化和定量分析。

使用方法:
  python visualize_reconstruction.py --phase 1 --model best_reconstruction.pth
  python visualize_reconstruction.py --phase 1.5 --model best_reconstruction_multi.pth
  python visualize_reconstruction.py --phase 1.6 --model best_reconstruction_independent.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import ModelFactory, ReconstructionModel
from datasets import InkTraceDataset

# ==================== 配置定义 ====================

PHASE_CONFIGS = {
    "1": {
        "name": "Phase 1: 单曲线重建",
        "mode": "single",
        "dataset_params": {},
    },
    "1.5": {
        "name": "Phase 1.5: 连续多段曲线重建",
        "mode": "continuous",
        "dataset_params": {
            "max_segments": 8,
        },
    },
    "1.6": {
        "name": "Phase 1.6: 独立多笔画重建",
        "mode": "independent",
        "dataset_params": {
            "max_strokes": 8,
        },
    },
    "1.7": {
        "name": "Phase 1.7: 多路径连接曲线重建",
        "mode": "multi_connected",
        "dataset_params": {
            "max_paths": 4,
            "max_segments": 6,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一的重建可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["1", "1.5", "1.6", "1.7"],
        help="评估阶段",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型检查点路径 (.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备 (默认: 自动选择)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="可视化样本数量 (默认: 8)",
    )
    parser.add_argument(
        "--stats-samples",
        type=int,
        default=100,
        help="用于计算统计指标的样本数量 (默认: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="输出目录 (默认: results)",
    )

    return parser.parse_args()


def get_device(device_arg=None):
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")


def load_model(checkpoint_path, device):
    """加载模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    print(f"正在加载模型: {checkpoint_path}")

    # 使用 Factory 加载，它会自动处理结构创建
    try:
        model = ModelFactory.load_reconstruction_model(checkpoint_path, device=device)
        print(f"  ✓ 模型加载成功 (ReconstructionModel)")

        # 打印一下 loss 信息（如果存在）
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "loss" in checkpoint:
            print(f"  Checkpoint Loss: {checkpoint['loss']:.6f}")
        if "epoch" in checkpoint:
            print(f"  Checkpoint Epoch: {checkpoint['epoch']}")

        return model
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        raise


def compute_statistics(model, phase_config, device, num_samples=1000):
    """计算在大批量数据上的重建指标"""
    print(f"\n正在计算统计指标 (N={num_samples})...")

    # 创建临时数据集用于统计
    # 设为 num_workers=0 以避免小样本下的进程启动开销
    # 对于在线生成，单进程通常已足够快，且调试更友好
    dataset = InkTraceDataset(
        mode=phase_config["mode"],
        img_size=64,
        batch_size=32,
        epoch_length=num_samples,
        **phase_config["dataset_params"],
    )

    dataloader = DataLoader(dataset, batch_size=32, num_workers=0)

    mse_loss_fn = nn.MSELoss(reduction="none")

    all_mse = []
    all_max_diff = []

    model.eval()
    print(f"  进度: ", end="", flush=True)
    with torch.no_grad():
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)

            reconstructed, _ = model(imgs)

            # 还原到 0-255 用于计算差异
            imgs_np = imgs.cpu().numpy() * 255.0
            recon_np = reconstructed.cpu().numpy() * 255.0

            # 计算 Batch 内每个样本的 MSE
            # [B, 1, 64, 64] -> [B, 64*64] -> mean(dim=1)
            diff = imgs_np - recon_np
            mse = np.mean(diff**2, axis=(1, 2, 3))
            max_diff = np.max(np.abs(diff), axis=(1, 2, 3))

            all_mse.extend(mse)
            all_max_diff.extend(max_diff)

            # 简单进度条
            print(f".", end="", flush=True)
            if (i + 1) % 10 == 0:
                print(f" {len(all_mse)}/{num_samples}", end="", flush=True)

            if len(all_mse) >= num_samples:
                break
    print(" 完成")

    all_mse = np.array(all_mse[:num_samples])
    all_max_diff = np.array(all_max_diff[:num_samples])

    print(f"  MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"  Max Diff: {np.mean(all_max_diff):.2f} ± {np.std(all_max_diff):.2f}")

    return np.mean(all_mse), np.mean(all_max_diff)


def visualize_samples(model, phase_config, device, num_samples, save_path):
    """可视化样本对比图"""
    print(f"\n正在生成可视化图像: {save_path}")

    dataset = InkTraceDataset(
        mode=phase_config["mode"],
        img_size=64,
        batch_size=min(num_samples, 16),
        epoch_length=num_samples * 2,  # 多一点防止不够
        **phase_config["dataset_params"],
    )

    # 获取样本
    dataloader = DataLoader(dataset, batch_size=num_samples)
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:num_samples].to(device)

    model.eval()
    with torch.no_grad():
        reconstructed, _ = model(imgs)

    imgs_np = imgs.cpu().numpy().squeeze(1) * 255.0
    recon_np = reconstructed.cpu().numpy().squeeze(1) * 255.0

    # 绘图: 3 行 x N 列 (原图, 重建, 差异)
    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6.5))

    # 标题
    phase_name = phase_config["name"].split(":")[0]
    fig.suptitle(f"Reconstruction Results - {phase_name}", fontsize=16)

    for i in range(num_samples):
        # 1. 原图
        axes[0, i].imshow(imgs_np[i], cmap="gray", vmin=0, vmax=255)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Input", loc="left", fontsize=12)

        # 2. 重建图
        axes[1, i].imshow(recon_np[i], cmap="gray", vmin=0, vmax=255)
        axes[1, i].axis("off")

        # 计算单张 MSE
        img_mse = np.mean((imgs_np[i] - recon_np[i]) ** 2)
        axes[1, i].text(
            0.5,
            -0.15,
            f"MSE:{img_mse:.1f}",
            transform=axes[1, i].transAxes,
            ha="center",
            fontsize=9,
        )

        if i == 0:
            axes[1, i].set_title("Reconstructed", loc="left", fontsize=12)

        # 3. 差异图 (Heatmap)
        diff = np.abs(imgs_np[i] - recon_np[i])
        im = axes[2, i].imshow(
            diff, cmap="inferno", vmin=0, vmax=50
        )  # vmax=50为了突显差异
        axes[2, i].axis("off")

        # 显示最大差异
        axes[2, i].text(
            0.5,
            -0.15,
            f"MaxDiff:{np.max(diff):.1f}",
            transform=axes[2, i].transAxes,
            ha="center",
            fontsize=9,
        )

        if i == 0:
            axes[2, i].set_title("Difference", loc="left", fontsize=12)

    # 增加色条说明
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.2])
    fig.colorbar(im, cax=cbar_ax, label="Abs Diff")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 图像已保存")


def visualize_complexity_analysis(model, phase_config, device, save_path):
    """分析不同复杂度（笔画数/段数）下的重建质量"""
    mode = phase_config["mode"]

    # 只对支持动态数量的模式进行分析
    param_key = ""
    if mode == "independent":
        param_key = "max_strokes"
        nums = [1, 2, 4, 6, 8]
        label = "Strokes"
    elif mode == "continuous":
        param_key = "max_segments"
        nums = [2, 4, 6, 8, 12]
        label = "Segments"
    elif mode == "multi_connected":
        # 简化分析，只变路径数
        param_key = "max_paths"
        nums = [1, 2, 3, 4]
        label = "Paths"
    else:
        # Single mode 可以在这里跳过或者做其他的分析
        return

    print(f"\n正在进行复杂度分析 ({label})...")

    mses = []
    stds = []

    # 准备一张图展示不同复杂度的重建样本
    fig, axes = plt.subplots(2, len(nums), figsize=(2.5 * len(nums), 5))
    fig.suptitle(f"Performance vs Complexity ({label})", fontsize=16)

    model.eval()

    for idx, n in enumerate(nums):
        # 针对每个 N 创建特定的 dataset
        # 注意: datasets.py 中有些参数是 max_, 但对于分析我们需要固定或接近固定
        # InkTraceDataset 目前主要通过 Rust 随机生成，我们这里只能设置 max_ 上限
        # 并假设平均值会随之增加，或者通过固定种子来观察趋势

        # 技巧：如果是 independent 模式，dataset 支持 fixed_count (如果 Rust 接口支持)
        # 查看 datasets.py, independent 模式支持 fixed_count

        current_params = phase_config["dataset_params"].copy()

        if mode == "independent":
            dataset = InkTraceDataset(
                mode=mode,
                img_size=64,
                batch_size=1,
                epoch_length=50,
                max_strokes=n,
                fixed_count=n,  # 假设我们修改 datasets.py 支持了 fixed_count 或 Rust 端支持
            )
        else:
            # 对于其他模式，我们只能调整 max 值
            current_params[param_key] = n
            dataset = InkTraceDataset(
                mode=mode, img_size=64, batch_size=1, epoch_length=50, **current_params
            )

        # 计算统计
        dataloader = DataLoader(dataset, batch_size=16)
        batch_mse = []

        # 取第一个 batch 做可视化
        vis_img = None
        vis_recon = None

        with torch.no_grad():
            for i, (imgs, _) in enumerate(dataloader):
                imgs = imgs.to(device)
                reconstructed, _ = model(imgs)

                imgs_np = imgs.cpu().numpy() * 255.0
                recon_np = reconstructed.cpu().numpy() * 255.0

                diff = imgs_np - recon_np
                mse = np.mean(diff**2, axis=(1, 2, 3))
                batch_mse.extend(mse)

                if i == 0:
                    vis_img = imgs_np[0].squeeze()
                    vis_recon = recon_np[0].squeeze()

        avg_mse = np.mean(batch_mse)
        std_mse = np.std(batch_mse)
        mses.append(avg_mse)
        stds.append(std_mse)

        print(f"  {label}={n}: MSE={avg_mse:.2f}")

        # 绘图
        axes[0, idx].imshow(vis_img, cmap="gray", vmin=0, vmax=255)
        axes[0, idx].set_title(f"{label}={n}", fontsize=11)
        axes[0, idx].axis("off")

        axes[1, idx].imshow(vis_recon, cmap="gray", vmin=0, vmax=255)
        axes[1, idx].set_title(f"MSE={avg_mse:.1f}", fontsize=11)
        axes[1, idx].axis("off")

    plt.tight_layout()
    plot_path = save_path.replace(".png", "_complexity_vis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 绘制折线图
    plt.figure(figsize=(8, 5))
    plt.errorbar(nums, mses, yerr=stds, fmt="-o", capsize=5)
    plt.xlabel(f"Complexity ({label})")
    plt.ylabel("MSE Loss")
    plt.title(f"Reconstruction Error vs {label}")
    plt.grid(True, alpha=0.3)

    chart_path = save_path.replace(".png", "_complexity_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"  ✓ 复杂度分析图表已保存")


import os


def main():
    args = parse_args()

    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 获取配置
    if args.phase not in PHASE_CONFIGS:
        print(f"错误: 未知的 Phase {args.phase}")
        sys.exit(1)

    config = PHASE_CONFIGS[args.phase]
    device = get_device(args.device)

    print("=" * 60)
    print(f"可视化配置: {config['name']}")
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 加载模型
    model = load_model(args.model, device)

    # 1. 基础样本可视化
    save_path = output_dir / f"vis_phase{args.phase}_samples.png"
    visualize_samples(model, config, device, args.num_samples, str(save_path))

    # 2. 统计指标计算
    compute_statistics(model, config, device, args.stats_samples)

    # 3. 复杂度分析 (Phase 1 除外)
    if args.phase != "1":
        analysis_path = output_dir / f"vis_phase{args.phase}_analysis.png"
        visualize_complexity_analysis(model, config, device, str(analysis_path))

    print("\n可视化任务全部完成!")


if __name__ == "__main__":
    main()
