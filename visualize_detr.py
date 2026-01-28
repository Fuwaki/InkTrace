#!/usr/bin/env python3
"""
统一的 DETR 矢量化可视化工具 (V3)

功能：
1. 可视化模型预测的矢量线条 (贝塞尔曲线)
2. 展示 Refinement 过程 (每一层的中间输出)
3. 可视化笔画拓扑状态 (New/Continue)
4. 计算评估指标 (坐标误差, 数量误差)

使用方法：
  python visualize_detr.py --phase 2 --model best_detr_independent.pth
  python visualize_detr.py --phase 2.5 --model best_detr_continuous.pth --show-refine
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path as MplPath
from torch.utils.data import DataLoader

from models import ModelFactory
from datasets import InkTraceDataset

# ==================== 配置 ====================

PHASE_CONFIGS = {
    "2": {
        "name": "Phase 2: Independent Strokes",
        "mode": "independent",
        "dataset_params": {"max_strokes": 8},
    },
    "2.5": {
        "name": "Phase 2.5: Continuous Strokes",
        "mode": "continuous",
        "dataset_params": {"max_segments": 8},
    },
    "2.6": {
        "name": "Phase 2.6: Mix",
        "mode": "mixed",
        "dataset_params": {
            "configs": [
                ({"mode": "independent", "max_strokes": 8}, 0.5),
                ({"mode": "continuous", "max_segments": 8}, 0.5),
            ]
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="DETR 矢量化可视化工具")
    parser.add_argument("--phase", type=str, required=True, choices=["2", "2.5", "2.6"])
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--show-refine", action="store_true", help="显示迭代优化过程")
    parser.add_argument("--output-dir", type=str, default="results_vis")
    return parser.parse_args()


def get_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


# ==================== 绘图工具 ====================


def draw_bezier(ax, p0, p1, p2, p3, width, color="red", alpha=0.8, linestyle="-"):
    """绘制单条贝塞尔曲线"""
    verts = [
        (p0[0] * 64, p0[1] * 64),
        (p1[0] * 64, p1[1] * 64),
        (p2[0] * 64, p2[1] * 64),
        (p3[0] * 64, p3[1] * 64),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    path = MplPath(verts, codes)
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        lw=width,
        alpha=alpha,
        linestyle=linestyle,
    )
    ax.add_patch(patch)

    # 绘制控制点 (可选)
    # ax.plot([v[0] for v in verts], [v[1] for v in verts], 'o--', color=color, alpha=0.1, markersize=2)


def visualize_prediction(ax, img, strokes, pen_logits, title="Prediction"):
    """
    在给定的 Axes 上绘制预测结果

    Args:
        strokes: [N, 10]
        pen_logits: [N, 3] (Null, New, Continue)
    """
    # 背景图
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, alpha=0.6)

    # 解析预测
    pen_probs = torch.softmax(pen_logits, dim=-1)
    pen_classes = torch.argmax(pen_probs, dim=-1)  # [N]
    scores = torch.max(pen_probs, dim=-1).values

    num_strokes = 0

    # 按顺序绘制
    for i in range(len(strokes)):
        cls = pen_classes[i].item()
        score = scores[i].item()

        # 过滤无效笔画 (Null=0) 或 低置信度
        if cls == 0 or score < 0.3:
            continue

        s = strokes[i]  # [10]
        p0 = s[0:2].cpu().numpy()
        p1 = s[2:4].cpu().numpy()
        p2 = s[4:6].cpu().numpy()
        p3 = s[6:8].cpu().numpy()
        w_start = s[8].item()
        w_end = s[9].item()
        avg_w = (w_start + w_end) / 2.0

        # 颜色编码
        # New = Green (Start), Continue = Blue (Join)
        if cls == 1:  # New
            color = "#00FF00"  # Lime Green
            # 绘制起点标记
            ax.add_patch(
                Circle((p0[0] * 64, p0[1] * 64), radius=1.0, color=color, alpha=0.8)
            )
        else:  # Continue
            color = "#00FFFF"  # Cyan
            # 连笔不需要强调起点，因为它应该连接上一笔

        draw_bezier(ax, p0, p1, p2, p3, width=max(1, avg_w / 2), color=color)
        num_strokes += 1

    ax.set_title(f"{title}\n({num_strokes} strokes)", fontsize=10)
    ax.axis("off")


def visualize_gt(ax, img, target):
    """可视化 Ground Truth"""
    # Target shape: [N, 11] (last dim is class: 0=Pad, 1=New, 2=Cont)
    strokes = target[:, :10]
    classes = target[:, 10]

    ax.imshow(img, cmap="gray", vmin=0, vmax=255, alpha=0.6)

    count = 0
    for i in range(len(strokes)):
        cls = classes[i].item()
        if cls < 0.5:
            continue  # Padding/Null

        s = strokes[i]
        p0, p1, p2, p3 = s[0:2], s[2:4], s[4:6], s[6:8]
        avg_w = (s[8] + s[9]) / 2.0

        color = "orange" if cls == 1 else "yellow"
        draw_bezier(
            ax, p0, p1, p2, p3, width=max(1, avg_w / 2), color=color, linestyle="--"
        )
        count += 1

    ax.set_title(f"Ground Truth\n({count} strokes)", fontsize=10)
    ax.axis("off")


def main():
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading Phase {args.phase} config...")
    config = PHASE_CONFIGS[args.phase]

    # 1. Load Model
    print(f"Loading model: {args.model}")
    model = ModelFactory.load_vectorization_model(
        args.model, device=device, include_pixel_decoder=False
    )
    model.eval()

    # 2. Dataset
    print("Creating dataset...")
    if config["mode"] == "mixed":
        from datasets import MixedInkTraceDataset

        dataset = MixedInkTraceDataset(
            configs=config["dataset_params"]["configs"],
            epoch_length=100,
            batch_size=1,
            for_detr=True,  # 启用 DETR 标签格式
        )
    else:
        dataset = InkTraceDataset(
            mode=config["mode"],
            img_size=64,
            batch_size=1,
            epoch_length=100,
            for_detr=True,  # 启用 DETR 标签格式
            **config["dataset_params"],
        )
    loader = DataLoader(dataset, batch_size=args.num_samples)

    # 3. Predict
    imgs, targets = next(iter(loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        # Mode 'vectorize' returns (strokes, pen_logits, aux_outputs)
        outputs = model(imgs, mode="vectorize")

        # 兼容旧接口 (如果 load 的是 V2 模型)
        if len(outputs) == 2:
            final_strokes, final_logits = outputs
            aux_outputs = []
        else:
            final_strokes, final_logits, aux_outputs = outputs

    # 4. Visualize
    print("Generating visualizations...")

    # 如果开启 --show-refine，绘制详细过程
    # Grid: Rows = Samples, Cols = GT | Step 0 | Step 1 | ... | Final

    num_steps = len(aux_outputs)
    cols = 2 + (num_steps if args.show_refine else 0)
    if not args.show_refine:
        cols = 2  # Only GT and Final

    fig_width = 3 * cols
    fig_height = 3 * args.num_samples

    fig, axes = plt.subplots(args.num_samples, cols, figsize=(fig_width, fig_height))
    if args.num_samples == 1:
        axes = axes[None, :]  # Ensure 2D array

    for i in range(args.num_samples):
        img_np = imgs[i].cpu().squeeze().numpy() * 255.0

        # 1. GT
        visualize_gt(axes[i, 0], img_np, targets[i].cpu())

        col_idx = 1

        # 2. Refinement Steps
        if args.show_refine:
            for step, aux in enumerate(aux_outputs):
                strokes = aux["strokes"][i]  # [N, 10]
                logits = aux["pen_state_logits"][i]
                visualize_prediction(
                    axes[i, col_idx], img_np, strokes, logits, title=f"Step {step + 1}"
                )
                col_idx += 1

        # 3. Final Prediction
        visualize_prediction(
            axes[i, col_idx],
            img_np,
            final_strokes[i],
            final_logits[i],
            title="Final Output",
        )

    plt.tight_layout()
    save_path = output_dir / f"vis_detr_phase{args.phase}_refine{args.show_refine}.png"
    plt.savefig(save_path, dpi=100)
    print(f"Saved to {save_path}")

    # Optional: Complexity/Accuracy Metrics could be added here
    print("Done.")


if __name__ == "__main__":
    main()
