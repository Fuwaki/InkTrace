"""
DETR 矢量化结果可视化（使用统一接口）

使用 best_detr_vectorization.pth 模型可视化矢量化结果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from models import ModelFactory
from datasets import IndependentStrokesDataset


def draw_bezier_strokes(ax, strokes, validity, threshold=0.5, color='red'):
    """绘制贝塞尔笔画"""
    valid_mask = validity.squeeze(-1) > threshold

    if not valid_mask.any():
        return

    valid_strokes = strokes[valid_mask]

    for stroke in valid_strokes:
        x0, y0, x1, y1, x2, y2, x3, y3, w_start, w_end = stroke

        points = np.array([
            [x0 * 64, y0 * 64],
            [x1 * 64, y1 * 64],
            [x2 * 64, y2 * 64],
            [x3 * 64, y3 * 64]
        ])

        verts = points.tolist()
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

        path = Path(verts, codes)

        avg_width = (w_start + w_end) / 2
        patch = PathPatch(
            path,
            facecolor='none',
            edgecolor=color,
            linewidth=max(1, avg_width / 2),
            alpha=0.8
        )
        ax.add_patch(patch)


def visualize_detr_predictions(model, dataset, device, num_samples=6):
    """可视化 DETR 预测"""
    model.eval()

    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    fig.suptitle('DETR Vectorization Predictions', fontsize=16)

    with torch.no_grad():
        for i in range(num_samples):
            img_tensor, target = dataset[i * 100]
            img = img_tensor.numpy().squeeze() * 255

            img_batch = img_tensor.unsqueeze(0).to(device)

            # 使用统一的接口预测
            strokes, validity, _ = model(img_batch, mode='vectorize')

            # 提取 GT
            gt_strokes = target[..., :10].numpy()
            gt_validity = target[..., 10:11].numpy()

            # 显示原图 + GT
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
            draw_bezier_strokes(axes[0, i], gt_strokes, gt_validity,
                              threshold=0.5, color='lime')
            num_gt = (gt_validity > 0.5).sum()
            axes[0, i].set_title(f'GT ({num_gt} strokes)', fontsize=10)
            axes[0, i].axis('off')

            # 显示原图 + 预测
            axes[1, i].imshow(img, cmap='gray', vmin=0, vmax=255, alpha=0.5)
            pred_strokes_np = strokes.cpu().squeeze(0).numpy()
            pred_validity_np = validity.cpu().squeeze(0).unsqueeze(-1).numpy()
            draw_bezier_strokes(axes[1, i], pred_strokes_np, pred_validity_np,
                              threshold=0.5, color='red')
            num_pred = (pred_validity_np > 0.5).sum()
            axes[1, i].set_title(f'Prediction ({num_pred} strokes)', fontsize=10)
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('detr_predictions.png', dpi=150, bbox_inches='tight')
    print("✓ 保存图像: detr_predictions.png")
    plt.show()


def compute_metrics(model, dataset, device, num_samples=100):
    """计算评估指标"""
    model.eval()

    coord_errors = []
    width_errors = []
    count_errors = []

    print("\n计算评估指标...")

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            img_tensor, target = dataset[i]
            img_batch = img_tensor.unsqueeze(0).to(device)

            # 使用统一的接口预测
            strokes, validity, _ = model(img_batch, mode='vectorize')

            pred_strokes = strokes.cpu().squeeze(0)
            pred_validity = validity.cpu().squeeze(0)

            gt_strokes = target[..., :10]
            gt_validity = target[..., 10:11]

            valid_gt_mask = gt_validity.squeeze(-1) > 0.5
            valid_pred_mask = pred_validity.squeeze(-1) > 0.5

            num_gt = valid_gt_mask.sum().item()
            num_pred = valid_pred_mask.sum().item()

            count_errors.append(abs(num_gt - num_pred))

            if num_gt > 0 and num_pred > 0:
                gt_curves = gt_strokes[valid_gt_mask]
                pred_curves = pred_strokes[valid_pred_mask]

                min_curves = min(gt_curves.shape[0], pred_curves.shape[0])

                coord_error = torch.abs(
                    gt_curves[:min_curves, :8] - pred_curves[:min_curves, :8]
                ).mean().item()
                coord_errors.append(coord_error)

                width_error = torch.abs(
                    gt_curves[:min_curves, 8:10] - pred_curves[:min_curves, 8:10]
                ).mean().item()
                width_errors.append(width_error)

    # 打印统计
    print("\n" + "=" * 60)
    print("评估指标")
    print("=" * 60)

    if coord_errors:
        print(f"坐标误差 (L1): {np.mean(coord_errors):.6f} ± {np.std(coord_errors):.6f}")
        print(f"宽度误差 (L1): {np.mean(width_errors):.6f} ± {np.std(width_errors):.6f}")

    print(f"笔画数量误差: {np.mean(count_errors):.2f} ± {np.std(count_errors):.2f}")

    correct_counts = sum(1 for e in count_errors if e == 0)
    print(f"\n笔画数量准确率: {correct_counts}/{len(count_errors)} "
          f"({100*correct_counts/len(count_errors):.1f}%)")

    print("=" * 60)


def test_single_sample(model, dataset, device, index=0):
    """测试单个样本"""
    model.eval()

    img_tensor, target = dataset[index]
    img = img_tensor.numpy().squeeze() * 255

    print(f"\n样本 #{index}")
    print(f"  图像形状: {img.shape}")

    img_batch = img_tensor.unsqueeze(0).to(device)

    # 使用统一的接口预测
    strokes, validity, _ = model(img_batch, mode='vectorize')

    gt_strokes = target[..., :10].numpy()
    gt_validity = target[..., 10:11].numpy()

    num_gt = (gt_validity > 0.5).sum()
    num_pred = (validity > 0.5).sum().item()

    print(f"  GT 笔画数: {num_gt}")
    print(f"  预测笔画数: {num_pred}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray', vmin=0, vmax=255, alpha=0.5)
    draw_bezier_strokes(axes[1], gt_strokes, gt_validity,
                      threshold=0.5, color='lime')
    axes[1].set_title(f'GT ({num_gt} strokes)')
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray', vmin=0, vmax=255, alpha=0.5)
    pred_strokes_np = strokes.cpu().squeeze(0).numpy()
    pred_validity_np = validity.cpu().squeeze(0).unsqueeze(-1).numpy()
    draw_bezier_strokes(axes[2], pred_strokes_np, pred_validity_np,
                      threshold=0.5, color='red')
    axes[2].set_title(f'Prediction ({num_pred} strokes)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'detr_sample_{index}.png', dpi=150, bbox_inches='tight')
    print(f"\n  ✓ 保存图像: detr_sample_{index}.png")
    plt.show()


def main():
    device = 'xpu'
    print(f"使用设备: {device}")

    # 使用统一的工厂类加载模型
    print("\n加载矢量化模型...")
    model = ModelFactory.load_vectorization_model(
        'best_detr_vectorization.pth',
        device=device,
        include_pixel_decoder=False  # 可视化不需要 Pixel Decoder
    )
    print("  模型加载完成")

    # 创建数据集
    print("\n创建数据集...")
    dataset = IndependentStrokesDataset(size=64, length=500, max_strokes=8)

    # 测试单个样本
    print("\n" + "=" * 60)
    print("测试单个样本")
    print("=" * 60)
    test_single_sample(model, dataset, device, index=0)

    # 可视化多个样本
    print("\n" + "=" * 60)
    print("可视化多个样本")
    print("=" * 60)
    visualize_detr_predictions(model, dataset, device, num_samples=6)

    # 计算指标
    compute_metrics(model, dataset, device, num_samples=100)

    print("\n完成!")


if __name__ == '__main__':
    main()
