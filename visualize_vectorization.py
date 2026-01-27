"""
可视化和推理脚本：测试矢量化模型的效果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from model import StrokeEncoder
from pixel_decoder import PixelDecoder
from vector_decoder import VectorDecoder, VectorizationModel
from data import StrokeDataset


def load_model(checkpoint_path, device):
    """加载完整的矢量化模型"""
    print(f"加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 创建模型
    encoder = StrokeEncoder(
        in_channels=1,
        embed_dim=128,
        num_heads=4,
        num_layers=6,
        dropout=0.1
    ).to(device)

    pixel_decoder = PixelDecoder(embed_dim=128).to(device)

    vector_decoder = VectorDecoder(
        embed_dim=128,
        max_curves=8,
        num_layers=3,
        num_heads=4,
        dropout=0.1
    ).to(device)

    # 加载权重
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    vector_decoder.load_state_dict(checkpoint['vector_decoder_state_dict'])

    if 'pixel_decoder_state_dict' in checkpoint:
        pixel_decoder.load_state_dict(checkpoint['pixel_decoder_state_dict'])

    # 创建完整模型
    model = VectorizationModel(
        encoder=encoder,
        vector_decoder=vector_decoder,
        pixel_decoder=pixel_decoder
    ).to(device)

    model.eval()

    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")

    return model


@torch.no_grad()
def infer_vectorization(model, image, device, validity_threshold=0.5):
    """
    推理：从图像生成矢量路径

    Args:
        model: VectorizationModel
        image: [1, 1, 64, 64] 输入图像
        device: 设备
        validity_threshold: 有效性阈值

    Returns:
        List of curves: 每个曲线是 dict with keys:
            - 'points': [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
            - 'w_start': float
            - 'w_end': float
            - 'validity': float
    """
    model.eval()

    image = image.to(device)

    # 前向传播
    curves, validity = model(image)

    # 移到 CPU
    curves = curves.cpu().squeeze(0)  # [max_curves, 8]
    validity = validity.cpu().squeeze(0)  # [max_curves, 1]

    # 提取有效曲线
    valid_mask = validity.squeeze(-1) > validity_threshold

    result = []
    prev_point = None

    for i, is_valid in enumerate(valid_mask):
        if not is_valid:
            continue

        # 提取参数
        x1, y1, x2, y2, x3, y3, w_start, w_end = curves[i]

        # 确定起点 P0
        if prev_point is None:
            # 第一个点：从中心开始（或者可以从第一个控制点推导）
            p0 = torch.tensor([0.5, 0.5])
        else:
            p0 = prev_point

        p1 = torch.tensor([x1, y1])
        p2 = torch.tensor([x2, y2])
        p3 = torch.tensor([x3, y3])

        result.append({
            'points': [p0, p1, p2, p3],
            'w_start': w_start * 10.0,  # 反归一化
            'w_end': w_end * 10.0,
            'validity': validity[i].item()
        })

        # 更新下一段的起点
        prev_point = p3

    return result


def visualize_comparison(model, dataset, device, num_samples=4):
    """
    可视化：原始图像 vs 矢量重建

    Args:
        model: VectorizationModel
        dataset: StrokeDataset
        device: 设备
        num_samples: 可视化样本数
    """
    model.eval()

    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    fig.suptitle('Original vs Vectorized', fontsize=16)

    with torch.no_grad():
        for i in range(num_samples):
            # 获取样本
            img_tensor, target = dataset[i]
            img = img_tensor.numpy().squeeze() * 255  # [64, 64]

            # 推理
            img_batch = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 64, 64]
            curves = infer_vectorization(model, img_batch, device)

            # 显示原图
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title(f'Original #{i+1}')
            axes[0, i].axis('off')

            # 显示矢量重建
            axes[1, i].imshow(img, cmap='gray', vmin=0, vmax=255, alpha=0.3)
            axes[1, i].set_title(f'Vectorized ({len(curves)} curves)')
            axes[1, i].axis('off')

            # 绘制贝塞尔曲线
            for curve in curves:
                draw_bezier_curve(axes[1, i], curve)

    plt.tight_layout()
    plt.savefig('vectorization_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存可视化结果: vectorization_comparison.png")
    plt.show()


def draw_bezier_curve(ax, curve, color='red', linewidth=2):
    """
    在 matplotlib axes 上绘制贝塞尔曲线

    Args:
        ax: matplotlib axes
        curve: dict with keys: 'points', 'w_start', 'w_end'
    """
    points = curve['points']
    verts = [points[0].tolist()] + points[1:].tolist()

    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, linewidth=linewidth)
    ax.add_patch(patch)

    # 绘制控制点（可选）
    # for i, point in enumerate(points):
    #     ax.plot(point[0]*64, point[1]*64, 'go' if i == 0 or i == 3 else 'ro', markersize=3)


def test_single_image(model, dataset, device, index=0):
    """测试单个图像"""
    model.eval()

    # 获取样本
    img_tensor, target = dataset[index]
    img = img_tensor.numpy().squeeze() * 255

    print(f"\n测试图像 #{index}")
    print(f"  图像形状: {img.shape}")
    print(f"  灰度范围: [{img.min():.1f}, {img.max():.1f}]")

    # 推理
    img_batch = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    curves = infer_vectorization(model, img_batch, device)

    print(f"  预测曲线数: {len(curves)}")

    for i, curve in enumerate(curves):
        print(f"    曲线 #{i+1}:")
        print(f"      P0: ({curve['points'][0][0]:.3f}, {curve['points'][0][1]:.3f})")
        print(f"      P1: ({curve['points'][1][0]:.3f}, {curve['points'][1][1]:.3f})")
        print(f"      P2: ({curve['points'][2][0]:.3f}, {curve['points'][2][1]:.3f})")
        print(f"      P3: ({curve['points'][3][0]:.3f}, {curve['points'][3][1]:.3f})")
        print(f"      宽度: {curve['w_start']:.2f} -> {curve['w_end']:.2f}")
        print(f"      有效性: {curve['validity']:.3f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 原图
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # 矢量重建
    axes[1].imshow(img, cmap='gray', vmin=0, vmax=255, alpha=0.3)
    axes[1].set_title(f'Vectorized ({len(curves)} curves)')
    axes[1].axis('off')

    for curve in curves:
        draw_bezier_curve(axes[1], curve)

    plt.tight_layout()
    plt.savefig(f'vectorization_test_{index}.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存图像: vectorization_test_{index}.png")
    plt.show()


def evaluate_metrics(model, dataset, device, num_samples=100):
    """
    评估模型性能

    Metrics:
    - 坐标误差（L1）
    - 宽度误差（L1）
    - 曲线数量准确率
    """
    model.eval()

    coord_errors = []
    width_errors = []
    count_errors = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            img_tensor, target = dataset[i]
            img_batch = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

            # 推理
            curves, validity = model(img_batch)

            # 提取有效曲线
            valid_mask = target[..., 8] > 0.5
            num_gt_curves = valid_mask.sum().item()

            pred_valid_mask = validity.squeeze(0).squeeze(-1).cpu() > 0.5
            num_pred_curves = pred_valid_mask.sum().item()

            # 曲线数量误差
            count_errors.append(abs(num_gt_curves - num_pred_curves))

            # 如果都有曲线，计算坐标和宽度误差
            if num_gt_curves > 0 and num_pred_curves > 0:
                gt_curves = target[valid_mask][:, :8].cpu()  # [N, 8]
                pred_curves = curves.squeeze(0)[pred_valid_mask][:, :8].cpu()  # [M, 8]

                # 取最小数量
                min_curves = min(gt_curves.shape[0], pred_curves.shape[0])

                # 坐标误差
                coord_error = torch.abs(gt_curves[:min_curves, :6] - pred_curves[:min_curves, :6]).mean().item()
                coord_errors.append(coord_error)

                # 宽度误差
                width_error = torch.abs(gt_curves[:min_curves, 6:8] - pred_curves[:min_curves, 6:8]).mean().item()
                width_errors.append(width_error)

    # 打印统计
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    if coord_errors:
        print(f"坐标误差 (L1): {np.mean(coord_errors):.6f} ± {np.std(coord_errors):.6f}")
        print(f"宽度误差 (L1): {np.mean(width_errors):.6f} ± {np.std(width_errors):.6f}")

    print(f"曲线数量误差: {np.mean(count_errors):.2f} ± {np.std(count_errors):.2f}")
    print("=" * 60)


if __name__ == '__main__':
    # 设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = 'xpu'

    print(f"使用设备: {device}")

    # 加载模型
    model = load_model('best_vectorization.pth', device)

    # 创建数据集
    from data import MultiStrokeDataset
    dataset = MultiStrokeDataset(size=64, length=100, max_strokes=8)

    # 测试单个图像
    test_single_image(model, dataset, device, index=0)

    # 可视化多个样本
    visualize_comparison(model, dataset, device, num_samples=4)

    # 评估性能
    evaluate_metrics(model, dataset, device, num_samples=100)
