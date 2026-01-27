"""
训练完整的矢量化模型（Phase 2）

训练策略：
1. 阶段 1：冻结 Encoder，只训练 Vector Decoder
2. 阶段 2：解冻 Encoder，端到端微调

损失函数：矢量损失 + 重建损失（防止 Encoder 遗忘）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

from model import StrokeEncoder
from pixel_decoder import PixelDecoder
from vector_decoder import VectorDecoder, VectorizationModel, train_freeze_encoder, train_end_to_end
from data import StrokeDataset


class MultiStrokeDataset(StrokeDataset):
    """
    多笔画数据集：继承自 StrokeDataset，生成多条连续的贝塞尔曲线
    """

    def __init__(self, size=64, length=10000, max_strokes=8):
        super().__init__(size, length)
        self.max_strokes = max_strokes

    def generate_multi_strokes(self):
        """生成多条连续的曲线"""
        # 随机决定笔画数量
        num_strokes = np.random.randint(1, self.max_strokes + 1)

        # 创建画布（2倍超采样）
        scale = 2
        canvas_size = self.size * scale
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        curves = []

        for i in range(num_strokes):
            # 随机起点（第一条笔画的起点随机，后续的可能接上一条）
            if i == 0:
                p0 = np.random.rand(2) * self.size
            else:
                # 70% 的概率接上一条的终点，30% 的概率随机起笔
                if np.random.rand() > 0.3:
                    p0 = curves[-1]['points'][3]  # 接上一条的终点
                else:
                    p0 = np.random.rand(2) * self.size

            # 随机生成控制点（P1, P2, P3）
            # 控制点可以在图像外部，允许曲线进出边界
            p1 = np.random.rand(2) * (self.size * 1.3) - (self.size * 0.15)
            p2 = np.random.rand(2) * (self.size * 1.3) - (self.size * 0.15)
            p3 = np.random.rand(2) * (self.size * 1.3) - (self.size * 0.15)

            points = np.stack([p0, p1, p2, p3])

            # 随机宽度
            w_start = np.random.uniform(1.0, 6.0)
            w_end = np.random.uniform(1.0, 6.0)

            # 渲染到画布
            self._render_bezier(canvas, points, w_start, w_end, scale)

            curves.append({
                'points': points,
                'w_start': w_start,
                'w_end': w_end
            })

        # 下采样回原始尺寸
        canvas = self.resize_canvas(canvas)

        return canvas, curves

    def _render_bezier(self, canvas, points, w_start, w_end, scale):
        """渲染单条贝塞尔曲线"""
        num_steps = 200

        for i in range(num_steps):
            t = i / (num_steps - 1)
            pt = self.get_bezier_point(t, points) * scale

            # 线性插值宽度
            current_width = w_start + (w_end - w_start) * t

            cv2_circle(
                canvas,
                (int(pt[0]), int(pt[1])),
                int(current_width * scale / 2),
                255,
                -1
            )

    def resize_canvas(self, canvas):
        """下采样画布"""
        import cv2
        return cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def sort_curves_spatially(self, curves):
        """按照起点位置排序：从上到下、从左到右"""
        if len(curves) == 0:
            return curves

        start_points = np.array([curve['points'][0] for curve in curves])
        # lexsort: 先按最后一列（Y），再按倒数第二列（X）
        sort_indices = np.lexsort((start_points[:, 0], start_points[:, 1]))
        return [curves[i] for i in sort_indices]

    def prepare_target(self, curves):
        """
        将曲线列表转为 tensor 格式

        Returns:
            target: [max_strokes, 9]  # 8 个参数 + 1 个有效性标志
        """
        num_curves = len(curves)

        # 创建目标数组
        target = np.zeros((self.max_strokes, 9), dtype=np.float32)

        for i in range(self.max_strokes):
            if i < num_curves:
                # 有效曲线
                curve = curves[i]
                points = curve['points'] / self.size  # 归一化到 [0, 1]

                # P1, P2, P3 的坐标（P0 可以推导或不需要）
                target[i, :6] = points[1:].flatten()  # [x1, y1, x2, y2, x3, y3]
                target[i, 6:8] = [
                    curve['w_start'] / 10.0,
                    curve['w_end'] / 10.0
                ]  # 宽度归一化到 [0, 1]
                target[i, 8] = 1.0  # 有效性标志
            else:
                # Padding（无效曲线）
                target[i, 8] = 0.0  # 无效标志

        return torch.from_numpy(target)

    def __getitem__(self, idx):
        # 1. 生成多条曲线
        canvas, curves = self.generate_multi_strokes()

        # 2. 排序（关键！）
        curves = self.sort_curves_spatially(curves)

        # 3. 准备图像 tensor
        img_tensor = torch.from_numpy(canvas).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        # 4. 准备 targets
        target = self.prepare_target(curves)

        return img_tensor, target


def cv2_circle(canvas, center, radius, color, thickness):
    """封装 cv2.circle"""
    import cv2
    cv2.circle(canvas, center, radius, color, thickness)


def load_phase1_model(checkpoint_path, device):
    """
    加载 Phase 1 训练好的模型

    Returns:
        encoder, pixel_decoder
    """
    print(f"加载 Phase 1 模型: {checkpoint_path}")

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

    # 加载权重
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    pixel_decoder.load_state_dict(checkpoint['decoder_state_dict'])

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")

    # 设置为评估模式
    encoder.eval()
    pixel_decoder.eval()

    return encoder, pixel_decoder


def train_phase2():
    """Phase 2 训练主函数"""

    # ==================== 配置 ====================
    config = {
        # 数据
        'dataset_size': 20000,
        'max_strokes': 8,
        'batch_size': 16,  # 小 batch size，因为模型更大了

        # 模型
        'embed_dim': 128,
        'max_curves': 8,  # 最多输出 8 条曲线（与数据集一致）

        # 训练阶段 1：冻结 Encoder
        'phase1_lr': 1e-3,
        'phase1_epochs': 30,

        # 训练阶段 2：端到端
        'phase2_lr': 1e-4,  # 更小的学习率，避免破坏 Encoder
        'phase2_epochs': 20,

        # 损失权重
        'reconstruction_weight': 0.1,  # 重建损失的权重
    }

    # ==================== 设备 ====================
    device = 'cpu'
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    #     device = 'xpu'

    print(f"使用设备: {device}")

    # ==================== 加载 Phase 1 模型 ====================
    encoder, pixel_decoder = load_phase1_model('best_reconstruction.pth', device)

    # ==================== 创建 Vector Decoder ====================
    vector_decoder = VectorDecoder(
        embed_dim=config['embed_dim'],
        max_curves=config['max_curves'],
        num_layers=3,
        num_heads=4,
        dropout=0.1
    ).to(device)

    # ==================== 创建完整模型 ====================
    model = VectorizationModel(
        encoder=encoder,
        vector_decoder=vector_decoder,
        pixel_decoder=pixel_decoder  # 保留用于计算重建损失
    ).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # ==================== 创建数据集 ====================
    print(f"\n创建数据集...")
    dataset = MultiStrokeDataset(
        size=64,
        length=config['dataset_size'],
        max_strokes=config['max_strokes']
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if device != 'cpu' else False
    )

    print(f"  数据集大小: {len(dataset)}")
    print(f"  批次数: {len(train_loader)}")

    # 测试加载一个 batch
    imgs, targets = next(iter(train_loader))
    print(f"\nBatch 形状:")
    print(f"  Images: {imgs.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  有效的曲线数: {(targets[..., 8].sum(dim=1)).long()}")

    # ==================== 训练阶段 1：冻结 Encoder ====================
    print("\n" + "=" * 60)
    print("阶段 1: 冻结 Encoder，训练 Vector Decoder")
    print("=" * 60)

    optimizer1 = optim.Adam(
        vector_decoder.parameters(),
        lr=config['phase1_lr']
    )

    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer1,
        T_max=config['phase1_epochs']
    )

    best_loss1 = float('inf')

    for epoch in range(1, config['phase1_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['phase1_epochs']}")
        print(f"  学习率: {optimizer1.param_groups[0]['lr']:.6f}")

        avg_loss, loss_dict = train_freeze_encoder(
            model, train_loader, optimizer1, device, epoch,
            reconstruction_weight=config['reconstruction_weight']
        )

        scheduler1.step()

        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  坐标损失: {loss_dict['coord']:.6f}")
        print(f"  宽度损失: {loss_dict['width']:.6f}")
        print(f"  有效性损失: {loss_dict['validity']:.6f}")
        print(f"  重建损失: {loss_dict['reconstruction']:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss1:
            best_loss1 = avg_loss
            torch.save({
                'epoch': epoch,
                'vector_decoder_state_dict': vector_decoder.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'loss': best_loss1,
            }, 'best_vector_decoder.pth')
            print(f"  ✓ 保存最佳模型 (Loss: {best_loss1:.6f})")

    # ==================== 训练阶段 2：端到端微调 ====================
    print("\n" + "=" * 60)
    print("阶段 2: 解冻 Encoder，端到端微调")
    print("=" * 60)

    # 解冻 Encoder
    for param in model.encoder.parameters():
        param.requires_grad = True

    # 重新统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params:,}")

    # 使用更小的学习率
    optimizer2 = optim.Adam(
        model.parameters(),
        lr=config['phase2_lr']
    )

    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2,
        T_max=config['phase2_epochs']
    )

    best_loss2 = float('inf')

    for epoch in range(1, config['phase2_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['phase2_epochs']}")
        print(f"  学习率: {optimizer2.param_groups[0]['lr']:.6f}")

        avg_loss, loss_dict = train_end_to_end(
            model, train_loader, optimizer2, device, epoch,
            reconstruction_weight=config['reconstruction_weight']
        )

        scheduler2.step()

        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  坐标损失: {loss_dict['coord']:.6f}")
        print(f"  宽度损失: {loss_dict['width']:.6f}")
        print(f"  有效性损失: {loss_dict['validity']:.6f}")
        print(f"  重建损失: {loss_dict['reconstruction']:.6f}")

        # 保存完整模型
        if avg_loss < best_loss2:
            best_loss2 = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'vector_decoder_state_dict': model.vector_decoder.state_dict(),
                'pixel_decoder_state_dict': model.pixel_decoder.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'loss': best_loss2,
            }, 'best_vectorization.pth')
            print(f"  ✓ 保存完整模型 (Loss: {best_loss2:.6f})")

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"阶段 1 最佳损失: {best_loss1:.6f}")
    print(f"阶段 2 最佳损失: {best_loss2:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    train_phase2()
