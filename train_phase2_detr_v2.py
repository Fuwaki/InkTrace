"""
Phase 2: DETR 风格矢量化训练（使用统一接口）

训练流程：
1. 使用 ModelFactory 加载 Phase 1.6 的模型
2. 创建完整的 VectorizationModel
3. 使用统一的 freeze/unfreeze 接口训练
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import ModelFactory
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

    device = 'xpu'
    print(f"使用设备: {device}")

    # ================================================================
    # 加载/创建模型
    # ================================================================
    print("\n创建矢量化模型...")

    # 尝试加载 Phase 1.6 的 Encoder
    try:
        checkpoint = torch.load('best_reconstruction_independent.pth', map_location=device)
        print("  找到 Phase 1.6 模型，加载 Encoder...")

        # 创建模型并加载 Encoder
        model = ModelFactory.create_vectorization_model(
            embed_dim=config['embed_dim'],
            num_slots=config['num_slots'],
            device=device,
            include_pixel_decoder=True  # 包含 Pixel Decoder 用于重建损失
        )

        # 加载 Encoder 权重
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"  Encoder Epoch: {checkpoint['epoch']}")

    except FileNotFoundError:
        print("  未找到 Phase 1.6 模型，从头创建...")
        model = ModelFactory.create_vectorization_model(
            embed_dim=config['embed_dim'],
            num_slots=config['num_slots'],
            device=device,
            include_pixel_decoder=True
        )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # ================================================================
    # 创建数据集
    # ================================================================
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

    # 测试加载
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

    # 使用统一的接口冻结 Encoder
    model.freeze_encoder()
    model.unfreeze_detr_decoder()

    # 只优化 DETR Decoder
    optimizer = optim.Adam(model.detr_decoder.parameters(), lr=config['phase1_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['phase1_epochs'])

    best_loss = float('inf')

    for epoch in range(1, config['phase1_epochs'] + 1):
        avg_loss, loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            ModelFactory.save_model(
                model, 'best_detr_vectorization.pth',
                epoch, best_loss, optimizer
            )
            print(f"  ✓ 保存最佳模型")

        print(f"\nEpoch {epoch}/{config['phase1_epochs']}")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  坐标: {loss_dict['coord']:.6f}, "
              f"宽度: {loss_dict['width']:.6f}, "
              f"有效性: {loss_dict['validity']:.6f}")
        print(f"  最佳损失: {best_loss:.6f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)

    # ================================================================
    # 阶段 2：端到端微调
    # ================================================================
    print("\n" + "=" * 60)
    print("阶段 2: 解冻 Encoder，端到端微调")
    print("=" * 60)

    # 使用统一的接口解冻 Encoder
    model.unfreeze_encoder()
    model.unfreeze_detr_decoder()

    # 重新统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params:,}")

    # 优化所有参数
    optimizer = optim.Adam(model.parameters(), lr=config['phase2_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['phase2_epochs'])

    best_loss = float('inf')

    for epoch in range(1, config['phase2_epochs'] + 1):
        avg_loss, loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            ModelFactory.save_model(
                model, 'best_detr_vectorization.pth',
                epoch, best_loss, optimizer
            )
            print(f"  ✓ 保存完整模型")

        print(f"\nEpoch {epoch}/{config['phase2_epochs']}")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  坐标: {loss_dict['coord']:.6f}, "
              f"宽度: {loss_dict['width']:.6f}")
        print(f"  最佳损失: {best_loss:.6f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Phase 2 训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型已保存: best_detr_vectorization.pth")
    print("=" * 60)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()

    epoch_loss = 0.0
    epoch_loss_dict = None

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 前向传播（使用统一的接口）
        # mode='both'：同时输出矢量和重建
        strokes, validity, reconstructed = model(imgs, mode='both')

        # 计算损失
        loss, loss_dict = criterion(strokes, validity, targets)

        # 如果有重建，加上重建损失
        if reconstructed is not None:
            recon_loss = torch.nn.functional.mse_loss(reconstructed, imgs)
            loss = loss + 0.1 * recon_loss
            loss_dict['reconstruction'] = recon_loss.item()

        # 反向传播
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if epoch_loss_dict is None:
            epoch_loss_dict = {key: 0.0 for key in loss_dict.keys()}

        for key in epoch_loss_dict:
            epoch_loss_dict[key] += loss_dict[key]

        # 每 100 个 batch 打印一次
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.6f}, "
                  f"Coord: {loss_dict['coord']:.6f}, "
                  f"Width: {loss_dict['width']:.6f}")

    # 平均
    avg_loss = epoch_loss / len(train_loader)
    for key in epoch_loss_dict:
        epoch_loss_dict[key] /= len(train_loader)

    return avg_loss, epoch_loss_dict


if __name__ == '__main__':
    train_phase2_detr()
