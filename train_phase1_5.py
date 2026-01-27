"""
Phase 1.5: 多曲线重建预训练

目标：让 Encoder 适应多曲线输入
- 加载 Phase 1 的 Encoder 和 Pixel Decoder
- 用多曲线数据继续训练重建任务
- 让 Encoder 学会表示复杂的多曲线场景

之后再训练 Vector Decoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import StrokeEncoder
from pixel_decoder import PixelDecoder


class MultiStrokeReconstructionDataset:
    """
    多笔画重建数据集

    用于 Phase 1.5：让 Encoder 学会处理多曲线
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
        # 渐进式增加曲线数量
        progress = idx / self.length
        if progress < 0.3:
            num_strokes = 1
        elif progress < 0.6:
            num_strokes = np.random.randint(1, 4)
        else:
            num_strokes = np.random.randint(1, self.max_strokes + 1)

        # 生成多笔画
        scale = 2
        canvas_size = self.size * scale
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        current_point = np.random.rand(2) * self.size

        for _ in range(num_strokes):
            # 当前点作为 P0
            p0 = current_point

            # 生成方向和长度
            direction = np.random.randn(2)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            length = np.random.uniform(8, 20)

            # P3 = P0 + direction * length
            p3 = p0 + direction * length

            # P1, P2 在中间，加随机扰动
            p1 = p0 + direction * length * np.random.uniform(0.2, 0.4) + np.random.randn(2) * 3
            p2 = p0 + direction * length * np.random.uniform(0.6, 0.8) + np.random.randn(2) * 3

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

            current_point = p3

        # 下采样
        import cv2
        canvas = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)

        # 转为 tensor
        img_tensor = torch.from_numpy(canvas).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        # 重建目标就是图像本身
        return img_tensor, img_tensor.clone()


def train_phase1_5():
    """Phase 1.5 训练"""

    config = {
        'dataset_size': 20000,
        'max_strokes': 8,
        'batch_size': 32,

        'num_epochs': 30,
        'lr': 5e-4,  # 较小的学习率，微调
    }

    device = 'xpu'
    print(f"使用设备: {device}")

    # 加载 Phase 1 模型
    print("\n加载 Phase 1 模型...")
    checkpoint = torch.load('best_reconstruction.pth', map_location=device)

    encoder = StrokeEncoder(
        in_channels=1, embed_dim=128, num_heads=4, num_layers=6, dropout=0.1
    ).to(device)
    pixel_decoder = PixelDecoder(embed_dim=128).to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    pixel_decoder.load_state_dict(checkpoint['decoder_state_dict'])

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")

    # 统计参数
    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in pixel_decoder.parameters())
    print(f"\n模型参数: {total_params:,}")

    # 创建多曲线数据集
    print("\n创建多曲线数据集...")
    dataset = MultiStrokeReconstructionDataset(
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

    # 测试加载
    imgs, targets = next(iter(train_loader))
    print(f"\nBatch 形状:")
    print(f"  Images: {imgs.shape}")
    print(f"  Targets: {targets.shape}")

    # 损失函数和优化器
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
    print("Phase 1.5: 多曲线重建训练")
    print("=" * 60)

    best_loss = float('inf')
    history = {'train_loss': []}

    for epoch in range(1, config['num_epochs'] + 1):
        encoder.train()
        pixel_decoder.train()

        epoch_loss = 0.0

        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)

            optimizer.zero_grad()

            # 前向传播
            embeddings = encoder(imgs)
            reconstructed = pixel_decoder(embeddings)

            # 计算损失
            loss = criterion(reconstructed, imgs)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 每 100 个 batch 打印一次
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': pixel_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_reconstruction_multi.pth')
            print(f"  ✓ 保存最佳模型")

        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  最佳损失: {best_loss:.6f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Phase 1.5 训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型已保存: best_reconstruction_multi.pth")
    print("=" * 60)

    # 可视化一些重建结果
    print("\n生成重建对比图...")
    visualize_reconstruction(encoder, pixel_decoder, dataset, device)

    return encoder, pixel_decoder


def visualize_reconstruction(encoder, decoder, dataset, device, num_samples=4):
    """可视化重建效果"""
    import matplotlib.pyplot as plt

    encoder.eval()
    decoder.eval()

    fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))
    fig.suptitle('Multi-Stroke Reconstruction (Phase 1.5)', fontsize=16)

    with torch.no_grad():
        for i in range(num_samples):
            img_tensor, _ = dataset[i * 100]  # 取不同的样本
            img = img_tensor.numpy().squeeze() * 255

            img_batch = img_tensor.unsqueeze(0).to(device)
            embeddings = encoder(img_batch)
            reconstructed = decoder(embeddings)

            recon_img = reconstructed.cpu().squeeze().numpy() * 255

            # 原图
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title(f'Original #{i+1}')
            axes[0, i].axis('off')

            # 重建图
            axes[1, i].imshow(recon_img, cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title(f'Reconstructed')
            axes[1, i].axis('off')

    plt.tight_layout()
    
    plt.savefig('reconstruction_multi_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ 保存图像: reconstruction_multi_comparison.png")
    # plt.show()


if __name__ == '__main__':
    train_phase1_5()
