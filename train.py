"""
训练脚本：通过重建任务训练 Encoder
目标：让 Encoder 学到良好的中间表示
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import StrokeDataset
from model import StrokeEncoder
from pixel_decoder import PixelDecoder, ReconstructionModel


def train_reconstruction(
    num_epochs=50,
    batch_size=32,
    lr=1e-3,
    device='cpu',
    save_path='best_reconstruction.pth'
):
    """训练重建模型"""

    print("=" * 60)
    print("训练配置:")
    print(f"  设备: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  学习率: {lr}")
    print("=" * 60)

    # 1. 创建数据集和 DataLoader
    dataset = StrokeDataset(size=64, length=10000)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device != 'cpu' else False
    )

    # 2. 创建模型
    encoder = StrokeEncoder(
        in_channels=1,
        embed_dim=192,
        num_heads=4,
        num_layers=6,
        dropout=0.1
    )

    decoder = PixelDecoder(
        embed_dim=192,
        spatial_size=8,
        output_channels=1
    )

    model = ReconstructionModel(encoder, decoder)
    model = model.to(device)

    # 3. 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 4. 训练循环
    best_loss = float('inf')
    history = {'train_loss': []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)

            # 前向传播
            optimizer.zero_grad()
            reconstructed, embeddings = model(imgs)

            # 计算损失
            loss = criterion(reconstructed, imgs)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # 更新学习率
        scheduler.step()

        # 记录平均损失
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"  → 保存最佳模型 (loss: {best_loss:.6f})")

        # 每 10 个 epoch 可视化一次
        if epoch % 10 == 0:
            visualize_reconstruction(model, dataset, device, epoch)

    print("\n训练完成!")
    print(f"最佳损失: {best_loss:.6f}")

    # 绘制损失曲线
    plot_loss_curve(history)

    return model, history


def visualize_reconstruction(model, dataset, device, epoch, num_samples=4):
    """可视化重建效果"""
    model.eval()

    fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))
    fig.suptitle(f'Epoch {epoch} - Reconstruction Results', fontsize=16)

    with torch.no_grad():
        for i in range(num_samples):
            # 获取样本
            img, _, _, _ = dataset.render_stroke()
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

            # 重建
            reconstructed, _ = model(img_tensor)

            # 显示原图
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')

            # 显示重建图
            recon_img = reconstructed.cpu().squeeze().numpy()
            recon_img = (recon_img * 255).astype(np.uint8)
            axes[1, i].imshow(recon_img, cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f'reconstruction_epoch_{epoch}.png', dpi=100, bbox_inches='tight')
    plt.close()


def plot_loss_curve(history):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curve.png', dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # 检测设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = 'xpu'

    # 开始训练
    model, history = train_reconstruction(
        num_epochs=50,
        batch_size=32,
        lr=1e-3,
        device=device,
        save_path='best_reconstruction.pth'
    )
