"""
多笔画重建可视化脚本

使用 best_reconstruction_multi.pth 模型可视化多笔画重建效果
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from model import StrokeEncoder
from pixel_decoder import PixelDecoder


class MultiStrokeReconstructionDataset:
    """多笔画数据集（用于可视化）"""

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

    def generate_multi_strokes(self, num_strokes=None):
        """生成指定数量的笔画"""
        if num_strokes is None:
            num_strokes = np.random.randint(1, self.max_strokes + 1)

        scale = 2
        canvas_size = self.size * scale
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        curves = []
        current_point = np.random.rand(2) * self.size

        for i in range(num_strokes):
            # 当前点作为 P0
            p0 = current_point

            # 生成方向和长度
            direction = np.random.randn(2)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            length = np.random.uniform(8, 20)

            # P3 = P0 + direction * length
            p3 = p0 + direction * length

            # P1, P2 在中间，加随机扰动
            p1 = (
                p0
                + direction * length * np.random.uniform(0.2, 0.4)
                + np.random.randn(2) * 3
            )
            p2 = (
                p0
                + direction * length * np.random.uniform(0.6, 0.8)
                + np.random.randn(2) * 3
            )

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
                    -1,
                )

            curves.append({"points": points, "w_start": w_start, "w_end": w_end})

            current_point = p3

        # 下采样
        import cv2

        canvas = cv2.resize(
            canvas, (self.size, self.size), interpolation=cv2.INTER_AREA
        )

        return canvas, curves, num_strokes

    def __getitem__(self, idx):
        canvas, curves, num_strokes = self.generate_multi_strokes()

        img_tensor = torch.from_numpy(canvas).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        return img_tensor, curves, num_strokes


def load_model(checkpoint_path, device):
    """加载 Phase 1.5 模型"""
    print(f"加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = StrokeEncoder(
        in_channels=1, embed_dim=128, num_heads=4, num_layers=6, dropout=0.1
    ).to(device)
    pixel_decoder = PixelDecoder(embed_dim=128).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    pixel_decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    pixel_decoder.eval()

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")

    return encoder, pixel_decoder


def visualize_reconstruction_multi_stroke(
    encoder, decoder, dataset, device, num_samples=8
):
    """可视化多笔画重建效果"""
    encoder.eval()
    decoder.eval()

    # 创建 3 行：原图、重建图、差异图
    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9))
    fig.suptitle("Multi-Stroke Reconstruction (Phase 1.5)", fontsize=16)

    with torch.no_grad():
        for i in range(num_samples):
            # 获取样本
            img_tensor, curves, num_strokes = dataset[i * 50]  # 取不同的样本
            img = img_tensor.numpy().squeeze() * 255

            # 重建
            img_batch = img_tensor.unsqueeze(0).to(device)
            embeddings = encoder(img_batch)
            reconstructed = decoder(embeddings)

            recon_img = reconstructed.cpu().squeeze().numpy() * 255

            # 计算差异
            diff = np.abs(img - recon_img)

            # 显示原图
            axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[0, i].set_title(f"Original ({num_strokes} strokes)", fontsize=10)
            axes[0, i].axis("off")

            # 显示重建图
            axes[1, i].imshow(recon_img, cmap="gray", vmin=0, vmax=255)
            axes[1, i].set_title(f"Reconstructed", fontsize=10)
            axes[1, i].axis("off")

            # 显示差异图
            axes[2, i].imshow(diff, cmap="hot", vmin=0, vmax=50)
            axes[2, i].set_title(f"Diff (max={diff.max():.1f})", fontsize=10)
            axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig("multi_stroke_reconstruction.png", dpi=150, bbox_inches="tight")
    print(f"✓ 保存图像: multi_stroke_reconstruction.png")
    plt.show()


def test_different_num_strokes(encoder, decoder, device):
    """测试不同笔画数量的重建效果"""
    encoder.eval()
    decoder.eval()

    # 测试 1, 2, 4, 8 条笔画
    num_strokes_list = [1, 2, 4, 8]

    fig, axes = plt.subplots(
        2, len(num_strokes_list), figsize=(4 * len(num_strokes_list), 8)
    )
    fig.suptitle("Reconstruction Quality vs Number of Strokes", fontsize=16)

    with torch.no_grad():
        for idx, num_strokes in enumerate(num_strokes_list):
            # 生成指定数量的笔画
            dataset = MultiStrokeReconstructionDataset(
                size=64, length=100, max_strokes=8
            )
            canvas, curves, _ = dataset.generate_multi_strokes(num_strokes)

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
            axes[0, idx].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[0, idx].set_title(f"{num_strokes} Stroke(s)", fontsize=12)
            axes[0, idx].axis("off")

            # 显示重建图
            axes[1, idx].imshow(recon_img, cmap="gray", vmin=0, vmax=255)
            axes[1, idx].set_title(f"Reconstructed\nMSE={mse:.4f}", fontsize=10)
            axes[1, idx].axis("off")

    plt.tight_layout()
    plt.savefig("multi_stroke_quality_comparison.png", dpi=150, bbox_inches="tight")
    print(f"✓ 保存图像: multi_stroke_quality_comparison.png")
    plt.show()


def compute_statistics(encoder, decoder, dataset, device, num_samples=100):
    """计算重建统计信息"""
    encoder.eval()
    decoder.eval()

    mse_list = []
    max_diff_list = []

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

    # 打印统计
    print("\n" + "=" * 60)
    print("重建统计信息")
    print("=" * 60)
    print(f"MSE: {np.mean(mse_list):.6f} ± {np.std(mse_list):.6f}")
    print(f"最大差异: {np.mean(max_diff_list):.2f} ± {np.std(max_diff_list):.2f}")
    print(f"最小 MSE: {np.min(mse_list):.6f}")
    print(f"最大 MSE: {np.max(mse_list):.6f}")
    print("=" * 60)


def main():
    # 设备检测
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"

    print(f"使用设备: {device}")

    # 加载模型
    encoder, pixel_decoder = load_model("best_reconstruction_multi.pth", device)

    # 创建数据集
    print("\n创建多笔画数据集...")
    dataset = MultiStrokeReconstructionDataset(size=64, length=500, max_strokes=8)

    # 可视化 1：多样本重建
    print("\n可视化 1: 多样本重建")
    visualize_reconstruction_multi_stroke(
        encoder, pixel_decoder, dataset, device, num_samples=8
    )

    # 可视化 2：不同笔画数量对比
    print("\n可视化 2: 不同笔画数量对比")
    test_different_num_strokes(encoder, pixel_decoder, device)

    # 计算统计信息
    print("\n计算统计信息...")
    compute_statistics(encoder, pixel_decoder, dataset, device, num_samples=100)

    print("\n完成!")


if __name__ == "__main__":
    main()
