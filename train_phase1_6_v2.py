"""
Phase 1.6: 独立多笔画重建预训练（使用统一接口）

目标：让 Encoder 适应**独立的、不连续的**多个笔画
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import ModelFactory, ReconstructionModel
from datasets import IndependentStrokesDataset


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

    # 使用工厂类创建模型
    model = ModelFactory.create_reconstruction_model(device=device)

    # 加载权重
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.pixel_decoder.load_state_dict(checkpoint['decoder_state_dict'])

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")

    total_params = sum(p.numel() for p in model.parameters())
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
    print(f"  Targets: {targets.shape}")

    # 损失和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    # 训练
    print("\n" + "=" * 60)
    print("Phase 1.6: 独立多笔画重建训练")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(1, config['num_epochs'] + 1):
        model.train()

        epoch_loss = 0.0

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)

            optimizer.zero_grad()

            # 使用统一的接口
            reconstructed, _ = model(imgs)

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
            # 使用工厂类保存
            ModelFactory.save_model(model, 'best_reconstruction_independent.pth',
                                  epoch, best_loss, optimizer)
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
    visualize_independent_strokes(model, dataset, device)

    return model


def visualize_independent_strokes(model, dataset, device, num_samples=6):
    """可视化独立笔画重建"""
    import matplotlib.pyplot as plt

    model.eval()

    fig, axes = plt.subplots(2, num_samples, figsize=(12, 4))
    fig.suptitle('Independent Strokes Reconstruction (Phase 1.6)', fontsize=16)

    with torch.no_grad():
        for i in range(num_samples):
            img_tensor, _ = dataset[i * 100]
            img = img_tensor.numpy().squeeze() * 255

            img_batch = img_tensor.unsqueeze(0).to(device)

            # 使用统一的接口
            reconstructed, _ = model(img_batch)

            recon_img = reconstructed.cpu().squeeze().numpy() * 255

            diff = np.abs(img - recon_img).mean()

            # 原图
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')

            # 重建图
            axes[1, i].imshow(recon_img, cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title(f'Reconstructed (diff={diff:.1f})')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction_independent.png', dpi=150, bbox_inches='tight')
    print("  ✓ 保存图像: reconstruction_independent.png")
    plt.show()


if __name__ == '__main__':
    import numpy as np
    train_phase1_6()
