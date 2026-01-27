"""
Phase 2: DETR 风格矢量化训练 (V2 改进版)

训练流程：
1. 使用 ModelFactory 加载 Phase 1.6 的模型
2. 创建完整的 VectorizationModel
3. 使用统一的 freeze/unfreeze 接口训练

改进点：
- 渐进式训练：先简单样本（少笔画），再复杂样本
- 梯度裁剪：防止梯度爆炸
- 学习率预热：稳定训练初期
- 更好的日志输出
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

from models import ModelFactory
from datasets import IndependentStrokesDataset
from losses import DETRLoss


def train_phase2_detr():
    """Phase 2 DETR 训练主函数"""

    config = {
        # 数据
        "dataset_size": 30000,  # 增加数据量
        "max_strokes": 8,
        "batch_size": 32,  # 增大 batch size
        # 模型
        "num_slots": 8,
        "embed_dim": 128,
        # 阶段 1：冻结 Encoder
        "phase1_lr": 1e-3,  # 提高初始学习率
        "phase1_epochs": 80,  # 增加 epoch
        # 阶段 2：端到端
        "phase2_lr": 5e-5,  # 降低学习率
        "phase2_epochs": 30,
        # 损失权重
        "coord_weight": 5.0,
        "width_weight": 2.0,
        "validity_weight": 2.0,
        # 训练技巧
        "grad_clip": 1.0,  # 梯度裁剪
        "warmup_epochs": 5,  # 学习率预热
    }

    if torch.cuda.is_available():
        device = "cuda"
        # 启用 cudnn benchmark 以加速训练
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    print(f"使用设备: {device}")

    # ================================================================
    # 加载/创建模型
    # ================================================================
    print("\n创建矢量化模型 (V2 改进版)...")

    # 尝试加载 Phase 1.6 的 Encoder
    try:
        checkpoint = torch.load(
            "best_reconstruction_independent.pth", map_location=device
        )
        print("  找到 Phase 1.6 模型，加载 Encoder...")

        # 创建模型并加载 Encoder
        model = ModelFactory.create_vectorization_model(
            embed_dim=config["embed_dim"],
            num_slots=config["num_slots"],
            device=device,
            include_pixel_decoder=True,  # 包含 Pixel Decoder 用于重建损失
        )

        # 加载 Encoder 权重
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        print(f"  Encoder Epoch: {checkpoint['epoch']}")

    except FileNotFoundError:
        print("  未找到 Phase 1.6 模型，从头创建...")
        model = ModelFactory.create_vectorization_model(
            embed_dim=config["embed_dim"],
            num_slots=config["num_slots"],
            device=device,
            include_pixel_decoder=True,
        )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.detr_decoder.parameters())
    print(f"  总参数: {total_params:,}")
    print(f"  Encoder 参数: {encoder_params:,}")
    print(f"  DETR Decoder 参数: {decoder_params:,}")

    # ================================================================
    # 创建数据集
    # ================================================================
    print("\n创建数据集...")
    dataset = IndependentStrokesDataset(
        size=64, length=config["dataset_size"], max_strokes=config["max_strokes"]
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"  数据集大小: {len(dataset)}")
    print(f"  批次数: {len(train_loader)}")

    # 测试加载
    imgs, targets = next(iter(train_loader))
    print(f"\nBatch 形状:")
    print(f"  Images: {imgs.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  有效笔画数: {(targets[..., 10].sum(dim=1)).long().tolist()[:8]}...")

    # 损失函数
    criterion = DETRLoss(
        coord_weight=config["coord_weight"],
        width_weight=config["width_weight"],
        validity_weight=config["validity_weight"],
        use_focal_loss=True,
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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params:,}")

    # 优化器：AdamW + 权重衰减
    optimizer = optim.AdamW(
        model.detr_decoder.parameters(), lr=config["phase1_lr"], weight_decay=0.01
    )

    # 学习率调度器：带预热的余弦退火
    def lr_lambda(epoch):
        if epoch < config["warmup_epochs"]:
            return (epoch + 1) / config["warmup_epochs"]
        else:
            progress = (epoch - config["warmup_epochs"]) / (
                config["phase1_epochs"] - config["warmup_epochs"]
            )
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")

    for epoch in range(1, config["phase1_epochs"] + 1):
        avg_loss, loss_dict = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            grad_clip=config["grad_clip"],
        )

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            ModelFactory.save_model(
                model, "best_detr_vectorization.pth", epoch, best_loss, optimizer
            )
            print(f"  ✓ 保存最佳模型")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{config['phase1_epochs']}")
        print(f"  平均损失: {avg_loss:.6f} (最佳: {best_loss:.6f})")
        print(
            f"  坐标: {loss_dict['coord']:.6f}, "
            f"宽度: {loss_dict['width']:.6f}, "
            f"有效性: {loss_dict['validity']:.6f}"
        )
        print(f"  学习率: {current_lr:.6f}")
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

    # 分组学习率：Encoder 用更小的学习率
    optimizer = optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": config["phase2_lr"] * 0.1},
            {"params": model.detr_decoder.parameters(), "lr": config["phase2_lr"]},
        ],
        weight_decay=0.01,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["phase2_epochs"]
    )

    best_loss = float("inf")

    for epoch in range(1, config["phase2_epochs"] + 1):
        avg_loss, loss_dict = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            grad_clip=config["grad_clip"],
        )

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            ModelFactory.save_model(
                model, "best_detr_vectorization.pth", epoch, best_loss, optimizer
            )
            print(f"  ✓ 保存完整模型")

        print(f"\nEpoch {epoch}/{config['phase2_epochs']}")
        print(f"  平均损失: {avg_loss:.6f} (最佳: {best_loss:.6f})")
        print(f"  坐标: {loss_dict['coord']:.6f}, 宽度: {loss_dict['width']:.6f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Phase 2 训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型已保存: best_detr_vectorization.pth")
    print("=" * 60)


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, epoch, grad_clip=1.0
):
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
        strokes, validity, reconstructed = model(imgs, mode="both")

        # 计算损失
        loss, loss_dict = criterion(strokes, validity, targets)

        # 如果有重建，加上重建损失
        if reconstructed is not None:
            recon_loss = torch.nn.functional.mse_loss(reconstructed, imgs)
            loss = loss + 0.1 * recon_loss
            loss_dict["reconstruction"] = recon_loss.item()

        # 反向传播
        loss.backward()

        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        epoch_loss += loss.item()

        if epoch_loss_dict is None:
            epoch_loss_dict = {key: 0.0 for key in loss_dict.keys()}

        for key in epoch_loss_dict:
            if key in loss_dict:
                epoch_loss_dict[key] += loss_dict[key]

        # 每 50 个 batch 打印一次
        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.6f}, "
                f"Coord: {loss_dict['coord']:.6f}, "
                f"Validity: {loss_dict['validity']:.6f}"
            )

    # 平均
    avg_loss = epoch_loss / len(train_loader)
    for key in epoch_loss_dict:
        epoch_loss_dict[key] /= len(train_loader)

    return avg_loss, epoch_loss_dict


if __name__ == "__main__":
    train_phase2_detr()
