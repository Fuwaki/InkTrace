"""
Phase 1.6: 独立多笔画重建预训练

目标：让 Encoder 适应**独立的、不连续的**多个笔画
- 与 Phase 1.5 的区别：笔画之间不连接
- 每个笔画有独立的起点 P0
- 为 Phase 2 的 DETR 风格矢量化做准备
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import StrokeEncoder
from pixel_decoder import PixelDecoder


class IndependentStrokesDataset:
    """
    独立多笔画数据集

    关键：每个笔画是独立的，不连接
    """

    def __init__(self, size=64, length=10000, max_strokes=8):
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

    def __getitem__(self, idx):
        # 渐进式增加笔画数量
        progress = idx / self.length
        if progress < 0.3:
            num_strokes = np.random.randint(1, 3)
        elif progress < 0.6:
            num_strokes = np.random.randint(1, 5)
        else:
            num_strokes = np.random.randint(1, self.max_strokes + 1)

        # 生成独立的多笔画
        canvas = self.generate_independent_strokes(num_strokes)

        # 转为 tensor
        img_tensor = torch.from_numpy(canvas).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        return img_tensor, img_tensor.clone()

    def generate_independent_strokes(self, num_strokes):
        """生成多个独立的笔画"""
        scale = 2
        canvas_size = self.size * scale
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

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

        # 下采样
        import cv2
        canvas = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return canvas


def train_phase1_6():
    """Phase 1.6 训练"""

    config = {
        'dataset_size': 20000,
        'max_strokes': 8,
        'batch_size': 32,

        'num_epochs': 30,
        'lr': 5e-4,
    }

    device = 'xpu'
    print(f"使用设备: {device}")

    # 加载 Phase 1.5 模型
    print("\n加载 Phase 1.5 模型...")
    try:
        checkpoint = torch.load('best_reconstruction_multi.pth', map_location=device)
        print("  使用 Phase 1.5 模型（连续多曲线）")
    except FileNotFoundError:
        checkpoint = torch.load('best_reconstruction.pth', map_location=device)
        print("  使用 Phase 1 模型（单曲线）")

    encoder = StrokeEncoder(
        in_channels=1, embed_dim=128, num_heads=4, num_layers=6, dropout=0.1
    ).to(device)
    pixel_decoder = PixelDecoder(embed_dim=128).to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    pixel_decoder.load_state_dict(checkpoint['decoder_state_dict'])

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in pixel_decoder.parameters())
    print(f"\n模型参数: {total_params:,}")

    # 创建独立多笔画数据集
    print("\n创建独立多笔画数据集...")
    dataset = IndependentStrokesDataset(
        size=64,
        length=config['dataset_size'],
        max_strokes=config['max_strokes']
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    print(f"  数据集大小: {len(dataset)}")
    print(f"  批次数: {len(train_loader)}")

    # 测试
    imgs, targets = next(iter(train_loader))
    print(f"\nBatch 形状:")
    print(f"  Images: {imgs.shape}")

    # 损失和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(pixel_decoder.parameters()),
        lr=config['lr']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )

    # 训练
    print("\n" + "=" * 60)
    print("Phase 1.6: 独立多笔画重建训练")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(1, config['num_epochs'] + 1):
        encoder.train()
        pixel_decoder.train()

        epoch_loss = 0.0

        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)

            optimizer.zero_grad()

            embeddings = encoder(imgs)
            reconstructed = pixel_decoder(embeddings)

            loss = criterion(reconstructed, imgs)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': pixel_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_reconstruction_independent.pth')
            print(f"  ✓ 保存最佳模型")

        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  最佳损失: {best_loss:.6f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Phase 1.6 训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型已保存: best_reconstruction_independent.pth")
    print("=" * 60)

    # 可视化
    print("\n生成可视化...")
    visualize_independent_strokes(encoder, pixel_decoder, dataset, device)

    return encoder, pixel_decoder


def visualize_independent_strokes(encoder, decoder, dataset, device, num_samples=6):
    """可视化独立笔画重建"""
    import matplotlib.pyplot as plt

    encoder.eval()
    decoder.eval()

    fig, axes = plt.subplots(2, num_samples, figsize=(12, 4))
    fig.suptitle('Independent Strokes Reconstruction (Phase 1.6)', fontsize=16)

    with torch.no_grad():
        for i in range(num_samples):
            img_tensor, _ = dataset[i * 100]
            img = img_tensor.numpy().squeeze() * 255

            img_batch = img_tensor.unsqueeze(0).to(device)
            embeddings = encoder(img_batch)
            reconstructed = decoder(embeddings)

            recon_img = reconstructed.cpu().squeeze().numpy() * 255

            # 计算差异
            diff = np.abs(img - recon_img).mean()

            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')

            axes[1, i].imshow(recon_img, cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title(f'Reconstructed (diff={diff:.1f})')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction_independent.png', dpi=150, bbox_inches='tight')
    print("  ✓ 保存图像: reconstruction_independent.png")


if __name__ == '__main__':
    train_phase1_6()
