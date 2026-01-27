"""
Phase 2: DETR 风格矢量化训练

训练流程：
1. 加载 Phase 1.6 的 Encoder
2. 创建 DETR Vector Decoder
3. 使用 Hungarian Matching Loss 训练
4. 冻结 Encoder → 端到端微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import StrokeEncoder
from detr_decoder import DETRVectorDecoder
from datasets import IndependentStrokesDataset
from losses import DETRLoss


def train_phase2_detr():
    """Phase 2 DETR 训练主函数"""

    config = {
        # 数据
        'dataset_size': 20000,
        'max_strokes': 8,
        'batch_size': 16,

        # 模型
        'num_slots': 8,
        'embed_dim': 128,

        # 阶段 1：冻结 Encoder
        'phase1_lr': 5e-4,
        'phase1_epochs': 50,

        # 阶段 2：端到端
        'phase2_lr': 1e-4,
        'phase2_epochs': 20,

        # 损失权重
        'coord_weight': 5.0,
        'width_weight': 1.0,
        'validity_weight': 1.0,
    }

    # 设备
    device = 'xpu'
    print(f"使用设备: {device}")

    # 加载 Phase 1.6 模型
    print("\n加载 Phase 1.6 模型...")
    try:
        checkpoint = torch.load('best_reconstruction_independent.pth', map_location=device)
        print("  使用 Phase 1.6 模型（独立多笔画）")
    except FileNotFoundError:
        try:
            checkpoint = torch.load('best_reconstruction_multi.pth', map_location=device)
            print("  使用 Phase 1.5 模型（连续多曲线）")
        except FileNotFoundError:
            checkpoint = torch.load('best_reconstruction.pth', map_location=device)
            print("  使用 Phase 1 模型（单曲线）")

    encoder = StrokeEncoder(
        in_channels=1, embed_dim=128, num_heads=4, num_layers=6, dropout=0.1
    ).to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")

    # 创建 DETR Decoder
    print("\n创建 DETR Vector Decoder...")
    detr_decoder = DETRVectorDecoder(
        embed_dim=config['embed_dim'],
        num_slots=config['num_slots'],
        num_layers=3,
        num_heads=4,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in detr_decoder.parameters())
    print(f"  Decoder 参数: {total_params:,}")

    # 创建数据集
    print("\n创建数据集...")
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
    print(f"  有效笔画数: {(targets[..., 10].sum(dim=1)).long()}")

    # 损失函数
    criterion = DETRLoss(
        coord_weight=config['coord_weight'],
        width_weight=config['width_weight'],
        validity_weight=config['validity_weight']
    )

    # ================================================================
    # 阶段 1：冻结 Encoder
    # ================================================================
    print("\n" + "=" * 60)
    print("阶段 1: 冻结 Encoder，训练 DETR Decoder")
    print("=" * 60)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    detr_decoder.train()

    optimizer = optim.Adam(detr_decoder.parameters(), lr=config['phase1_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['phase1_epochs'])

    best_loss = float('inf')

    for epoch in range(1, config['phase1_epochs'] + 1):
        epoch_loss = 0.0
        epoch_loss_dict = None

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # 前向传播
            with torch.no_grad():
                embeddings = encoder(imgs)

            pred_strokes, pred_validity = detr_decoder(embeddings)

            # 计算损失
            loss, loss_dict = criterion(pred_strokes, pred_validity, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if epoch_loss_dict is None:
                epoch_loss_dict = {key: 0.0 for key in loss_dict.keys()}

            for key in epoch_loss_dict:
                epoch_loss_dict[key] += loss_dict[key]

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.6f}, "
                      f"Coord: {loss_dict['coord']:.6f}, "
                      f"Width: {loss_dict['width']:.6f}")

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        for key in epoch_loss_dict:
            epoch_loss_dict[key] /= len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'detr_decoder_state_dict': detr_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_detr_vectorization.pth')
            print(f"  ✓ 保存最佳模型")

        print(f"\nEpoch {epoch}/{config['phase1_epochs']}")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  坐标: {epoch_loss_dict['coord']:.6f}, "
              f"宽度: {epoch_loss_dict['width']:.6f}, "
              f"有效性: {epoch_loss_dict['validity']:.6f}, "
              f"匹配代价: {epoch_loss_dict['matching_cost']:.6f}")
        print(f"  最佳损失: {best_loss:.6f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)

    # ================================================================
    # 阶段 2：端到端微调
    # ================================================================
    print("\n" + "=" * 60)
    print("阶段 2: 解冻 Encoder，端到端微调")
    print("=" * 60)

    for param in encoder.parameters():
        param.requires_grad = True

    encoder.train()
    detr_decoder.train()

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(detr_decoder.parameters()),
        lr=config['phase2_lr']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['phase2_epochs'])

    best_loss = float('inf')

    for epoch in range(1, config['phase2_epochs'] + 1):
        epoch_loss = 0.0
        epoch_loss_dict = None

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # 前向传播
            embeddings = encoder(imgs)
            pred_strokes, pred_validity = detr_decoder(embeddings)

            # 计算损失
            loss, loss_dict = criterion(pred_strokes, pred_validity, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if epoch_loss_dict is None:
                epoch_loss_dict = {key: 0.0 for key in loss_dict.keys()}

            for key in epoch_loss_dict:
                epoch_loss_dict[key] += loss_dict[key]

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.6f}")

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        for key in epoch_loss_dict:
            epoch_loss_dict[key] /= len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'detr_decoder_state_dict': detr_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_detr_vectorization.pth')
            print(f"  ✓ 保存完整模型")

        print(f"\nEpoch {epoch}/{config['phase2_epochs']}")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  坐标: {epoch_loss_dict['coord']:.6f}, "
              f"宽度: {epoch_loss_dict['width']:.6f}")
        print(f"  最佳损失: {best_loss:.6f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Phase 2 训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型已保存: best_detr_vectorization.pth")
    print("=" * 60)


if __name__ == '__main__':
    train_phase2_detr()
