"""
独立笔画重建可视化脚本

使用 best_reconstruction_independent.pth 模型可视化独立笔画重建效果
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from model import StrokeEncoder
from pixel_decoder import PixelDecoder


class IndependentStrokesDataset:
    """独立多笔画数据集（用于可视化）"""

    def __init__(self, size=64, length=1000, max_strokes=8):
        self.size = size
        self.length = length
        self.max_strokes = max_strokes

    def __len__(self):
        return self.length

    def get_bezier_point(self, t, points):
        p0, p1, p2, p3 = points
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * p3
        )

    def generate_independent_strokes(self, num_strokes=None):
        """生成指定数量的独立笔画"""
        if num_strokes is None:
            num_strokes = np.random.randint(1, self.max_strokes + 1)

        scale = 2
        canvas_size = self.size * scale
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        strokes = []

        for _ in range(num_strokes):
            # 每个笔画完全独立，随机起点
            p0 = np.random.rand(2) * self.size * 0.8 + self.size * 0.1  # 留边距

            # 随机方向和长度
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            length = np.random.uniform(10, 25)

            # P3 = P0 + direction * length
            p3 = p0 + direction * length

            # P1, P2 控制点
            p1 = p0 + direction * length * np.random.uniform(0.2, 0.4) + np.random.randn(2) * 2
            p2 = p0 + direction * length * np.random.uniform(0.6, 0.8) + np.random.randn(2) * 2

            points = np.stack([p0, p1, p2, p3])

            # 随机宽度
            w_start = np.random.uniform(2.0, 5.0)
            w_end = np.random.uniform(2.0, 5.0)

            # 渲染
            num_steps = 200
            for i in range(num_steps):
                t = i / (num_steps - 1)
                pt = self.get_bezier_point(t, points) * scale
                current_width = w_start + (w_end - w_start) * t

                import cv2
                cv2.circle(
                    canvas,
                    (int(pt[0]), int(pt[1])),
                    int(current_width * scale / 2),
                    255,
                    -1
                )

            strokes.append({
                'points': points,
                'w_start': w_start,
                'w_end': w_end
            })

        # 下采样
        import cv2
        canvas = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return canvas, strokes, num_strokes

    def __getitem__(self, idx):
        canvas, strokes, num_strokes = self.generate_independent_strokes()

        img_tensor = torch.from_numpy(canvas).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        return img_tensor, strokes, num_strokes


def load_model(checkpoint_path, device):
    """加载 Phase 1.6 模型"""
    print(f"加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = StrokeEncoder(
        in_channels=1, embed_dim=128, num_heads=4, num_layers=6, dropout=0.1
    ).to(device)
    pixel_decoder = PixelDecoder(embed_dim=128).to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    pixel_decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    pixel_decoder.eval()

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")

    return encoder, pixel_decoder


def visualize_reconstruction_independent(encoder, decoder, dataset, device, num_samples=8):
    """可视化独立笔画重建效果"""
    encoder.eval()
    decoder.eval()

    # 创建 3 行：原图、重建图、差异图
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    fig.suptitle('Independent Strokes Reconstruction (Phase 1.6)', fontsize=16)

    with torch.no_grad():
        for i in range(num_samples):
            # 获取样本
            img_tensor, strokes, num_strokes = dataset[i * 50]
            img = img_tensor.numpy().squeeze() * 255

            # 重建
            img_batch = img_tensor.unsqueeze(0).to(device)
            embeddings = encoder(img_batch)
            reconstructed = decoder(embeddings)

            recon_img = reconstructed.cpu().squeeze().numpy() * 255

            # 计算差异
            diff = np.abs(img - recon_img)

            # 显示原图
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title(f'Original ({num_strokes} strokes)', fontsize=10)
            axes[0, i].axis('off')

            # 显示重建图
            axes[1, i].imshow(recon_img, cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title('Reconstructed', fontsize=10)
            axes[1, i].axis('off')

            # 显示差异图
            axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=50)
            max_diff = diff.max()
            axes[2, i].set_title(f'Diff (max={max_diff:.1f})', fontsize=10)
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('independent_strokes_reconstruction.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存图像: independent_strokes_reconstruction.png")
    plt.show()


def test_different_num_strokes(encoder, decoder, device):
    """测试不同笔画数量的重建效果"""
    encoder.eval()
    decoder.eval()

    # 测试 1, 2, 4, 8 个独立笔画
    num_strokes_list = [1, 2, 4, 8]

    fig, axes = plt.subplots(2, len(num_strokes_list), figsize=(4*len(num_strokes_list), 8))
    fig.suptitle('Independent Strokes Quality vs Number of Strokes', fontsize=16)

    with torch.no_grad():
        for idx, num_strokes in enumerate(num_strokes_list):
            # 生成指定数量的笔画
            dataset = IndependentStrokesDataset(size=64, length=100, max_strokes=8)
            canvas, strokes, _ = dataset.generate_independent_strokes(num_strokes)

            img = canvas  # 已经是 numpy 数组

            # 重建
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_batch = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
            embeddings = encoder(img_batch)
            reconstructed = decoder(embeddings)

            recon_img = reconstructed.cpu().squeeze().numpy() * 255

            # 计算误差
            mse = np.mean((img - recon_img) ** 2)
            max_diff = np.max(np.abs(img - recon_img))

            # 显示原图
            axes[0, idx].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0, idx].set_title(f'{num_strokes} Stroke(s)', fontsize=12)
            axes[0, idx].axis('off')

            # 显示重建图
            axes[1, idx].imshow(recon_img, cmap='gray', vmin=0, vmax=255)
            axes[1, idx].set_title(f'Reconstructed\nMSE={mse:.4f}', fontsize=10)
            axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig('independent_strokes_quality_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存图像: independent_strokes_quality_comparison.png")
    plt.show()


def visualize_stroke_separation(encoder, decoder, device):
    """可视化笔画分离效果（显示独立笔画的特点）"""
    encoder.eval()
    decoder.eval()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Independent Stroke Examples - Separated Strokes', fontsize=16)

    with torch.no_grad():
        for i in range(4):
            # 生成 2-3 个分离的笔画
            dataset = IndependentStrokesDataset(size=64, length=100, max_strokes=8)
            canvas, strokes, num_strokes = dataset.generate_independent_strokes(np.random.randint(2, 4))

            img = canvas

            # 重建
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_batch = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
            embeddings = encoder(img_batch)
            reconstructed = decoder(embeddings)

            recon_img = reconstructed.cpu().squeeze().numpy() * 255

            # 计算差异
            diff = np.abs(img - recon_img)

            # 显示原图
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title(f'Original ({num_strokes} independent strokes)', fontsize=11)
            axes[0, i].axis('off')

            # 显示重建图 + 差异信息
            axes[1, i].imshow(recon_img, cmap='gray', vmin=0, vmax=255)
            mse = np.mean((img - recon_img) ** 2)
            axes[1, i].set_title(f'Reconstructed (MSE={mse:.4f})', fontsize=11)
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('independent_strokes_separated.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存图像: independent_strokes_separated.png")
    plt.show()


def compute_statistics(encoder, decoder, dataset, device, num_samples=100):
    """计算重建统计信息"""
    encoder.eval()
    decoder.eval()

    mse_list = []
    max_diff_list = []
    stroke_count_mse = []

    print("\n计算重建统计信息...")

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            img_tensor, _, num_strokes = dataset[i]

            img = img_tensor.numpy().squeeze() * 255

            img_batch = img_tensor.unsqueeze(0).to(device)
            embeddings = encoder(img_batch)
            reconstructed = decoder(embeddings)

            recon_img = reconstructed.cpu().squeeze().numpy() * 255

            # 计算误差
            mse = np.mean((img - recon_img) ** 2)
            max_diff = np.max(np.abs(img - recon_img))

            mse_list.append(mse)
            max_diff_list.append(max_diff)
            stroke_count_mse.append((num_strokes, mse))

    # 打印统计
    print("\n" + "=" * 60)
    print("重建统计信息")
    print("=" * 60)
    print(f"总体 MSE: {np.mean(mse_list):.6f} ± {np.std(mse_list):.6f}")
    print(f"最大差异: {np.mean(max_diff_list):.2f} ± {np.std(max_diff_list):.2f}")
    print(f"最小 MSE: {np.min(mse_list):.6f}")
    print(f"最大 MSE: {np.max(mse_list):.6f}")

    # 按笔画数量分析
    print("\n按笔画数量分析:")
    from collections import defaultdict
    count_dict = defaultdict(list)
    for num_strokes, mse in stroke_count_mse:
        count_dict[num_strokes].append(mse)

    for num_strokes in sorted(count_dict.keys()):
        mses = count_dict[num_strokes]
        print(f"  {num_strokes} 笔画: MSE = {np.mean(mses):.6f} ± {np.std(mses):.6f} (n={len(mses)})")

    print("=" * 60)


def test_single_sample_detailed(encoder, decoder, dataset, device, index=0):
    """详细测试单个样本"""
    encoder.eval()
    decoder.eval()

    img_tensor, strokes, num_strokes = dataset[index]
    img = img_tensor.numpy().squeeze() * 255

    print(f"\n样本 #{index}")
    print(f"  图像形状: {img.shape}")
    print(f"  笔画数量: {num_strokes}")

    # 打印每个笔画的信息
    print(f"\n  笔画信息:")
    for i, stroke in enumerate(strokes):
        points = stroke['points']
        p0 = points[0]
        p3 = points[3]
        print(f"    笔画 {i+1}: P0=({p0[0]:.1f}, {p0[1]:.1f}), "
              f"P3=({p3[0]:.1f}, {p3[1]:.1f}), "
              f"width=({stroke['w_start']:.2f}, {stroke['w_end']:.2f})")

    # 重建
    with torch.no_grad():
        img_batch = img_tensor.unsqueeze(0).to(device)
        embeddings = encoder(img_batch)
        reconstructed = decoder(embeddings)

        recon_img = reconstructed.cpu().squeeze().numpy() * 255

    # 计算误差
    mse = np.mean((img - recon_img) ** 2)
    max_diff = np.max(np.abs(img - recon_img))

    print(f"\n  重建误差:")
    print(f"    MSE: {mse:.6f}")
    print(f"    最大差异: {max_diff:.2f}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f'Original ({num_strokes} independent strokes)', fontsize=12)
    axes[0].axis('off')

    # 重建图
    axes[1].imshow(recon_img, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Reconstructed', fontsize=12)
    axes[1].axis('off')

    # 差异图
    diff = np.abs(img - recon_img)
    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=50)
    axes[2].set_title(f'Difference (max={max_diff:.1f})', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(f'independent_stroke_sample_{index}.png', dpi=150, bbox_inches='tight')
    print(f"\n  ✓ 保存图像: independent_stroke_sample_{index}.png")
    plt.show()


def main():
    # 设备检测
    device = 'xpu'
    print(f"使用设备: {device}")

    # 加载模型
    encoder, pixel_decoder = load_model('best_reconstruction_independent.pth', device)

    # 创建数据集
    print("\n创建独立笔画数据集...")
    dataset = IndependentStrokesDataset(
        size=64,
        length=500,
        max_strokes=8
    )

    # 可视化 1：多样本重建
    print("\n可视化 1: 多样本重建")
    visualize_reconstruction_independent(encoder, pixel_decoder, dataset, device, num_samples=8)

    # 可视化 2：不同笔画数量对比
    print("\n可视化 2: 不同笔画数量对比")
    test_different_num_strokes(encoder, pixel_decoder, device)

    # 可视化 3：笔画分离效果
    print("\n可视化 3: 笔画分离效果展示")
    visualize_stroke_separation(encoder, pixel_decoder, device)

    # 详细测试单个样本
    print("\n详细测试单个样本")
    test_single_sample_detailed(encoder, pixel_decoder, dataset, device, index=0)

    # 计算统计信息
    print("\n计算统计信息...")
    compute_statistics(encoder, pixel_decoder, dataset, device, num_samples=100)

    print("\n完成!")


if __name__ == '__main__':
    main()
