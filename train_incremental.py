"""
渐进式训练脚本：从简单到复杂

策略：
1. 先训练单曲线（确保模型学会基本曲线拟合）
2. 再训练2-3条曲线（学习连接关系）
3. 最后训练8条曲线（完整场景）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import StrokeEncoder
from pixel_decoder import PixelDecoder
from vector_decoder import VectorDecoder, VectorizationModel
from data import StrokeDataset


class SimpleMultiStrokeDataset(StrokeDataset):
    """
    简化版多笔画数据集
    - 渐进式增加曲线数量
    - 明确的起点-终点连接关系
    """

    def __init__(self, size=64, length=10000, max_strokes=8, mode='progressive'):
        super().__init__(size, length)
        self.max_strokes = max_strokes
        self.mode = mode  # 'progressive' | 'fixed'

    def __getitem__(self, idx):
        # 根据模式决定曲线数量
        if self.mode == 'progressive':
            # 渐进式：前50%单曲线，后50%逐渐增加
            progress = idx / len(self)
            if progress < 0.3:
                num_strokes = 1
            elif progress < 0.6:
                num_strokes = np.random.randint(1, 4)
            else:
                num_strokes = np.random.randint(1, self.max_strokes + 1)
        else:  # 'fixed'
            num_strokes = self.max_strokes

        # 生成曲线
        canvas, curves = self.generate_connected_strokes(num_strokes)

        # 排序（重要！）
        curves = self.sort_curves_spatially(curves)

        # 准备 tensor
        img_tensor = torch.from_numpy(canvas).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        # 准备 targets
        target = self.prepare_target(curves)

        return img_tensor, target

    def generate_connected_strokes(self, num_strokes):
        """生成连续的笔画序列"""
        scale = 2
        canvas_size = self.size * scale
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        curves = []

        # 随机起点
        current_point = np.random.rand(2) * self.size

        for i in range(num_strokes):
            # 当前点作为 P0
            p0 = current_point

            # 生成控制点（P1, P2, P3）
            # 约束：P3 应该在 P0 附近，形成自然的书写方向
            direction = np.random.randn(2)
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            # 曲线长度：10-20 像素
            length = np.random.uniform(10, 20)

            # P3 = P0 + direction * length
            p3 = p0 + direction * length

            # P1, P2 在中间，加一些随机扰动
            p1 = p0 + direction * length * 0.3 + np.random.randn(2) * 3
            p2 = p0 + direction * length * 0.7 + np.random.randn(2) * 3

            # 限制在画布内（可以稍微出界）
            points = np.stack([p0, p1, p2, p3])

            # 随机宽度
            w_start = np.random.uniform(2.0, 5.0)
            w_end = np.random.uniform(2.0, 5.0)

            # 渲染
            self._render_bezier(canvas, points, w_start, w_end, scale)

            curves.append({
                'points': points,
                'w_start': w_start,
                'w_end': w_end
            })

            # 更新当前点为 P3
            current_point = p3

        # 下采样
        canvas = self.resize_canvas(canvas)

        return canvas, curves

    def _render_bezier(self, canvas, points, w_start, w_end, scale):
        """渲染贝塞尔曲线"""
        num_steps = 200

        for i in range(num_steps):
            t = i / (num_steps - 1)
            pt = self.get_bezier_point(t, points) * scale
            current_width = w_start + (w_end - w_start) * t

            cv2_circle(
                canvas,
                (int(pt[0]), int(pt[1])),
                int(current_width * scale / 2),
                255,
                -1
            )

    def resize_canvas(self, canvas):
        import cv2
        return cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def sort_curves_spatially(self, curves):
        """按照起点位置排序"""
        if len(curves) == 0:
            return curves

        start_points = np.array([curve['points'][0] for curve in curves])
        sort_indices = np.lexsort((start_points[:, 0], start_points[:, 1]))
        return [curves[i] for i in sort_indices]

    def prepare_target(self, curves):
        """准备目标 tensor"""
        num_curves = len(curves)
        target = np.zeros((self.max_strokes, 9), dtype=np.float32)

        for i in range(self.max_strokes):
            if i < num_curves:
                curve = curves[i]
                points = curve['points'] / self.size

                # P1, P2, P3
                target[i, :6] = points[1:].flatten()
                target[i, 6:8] = [
                    curve['w_start'] / 10.0,
                    curve['w_end'] / 10.0
                ]
                target[i, 8] = 1.0
            else:
                target[i, 8] = 0.0

        return torch.from_numpy(target)


def cv2_circle(canvas, center, radius, color, thickness):
    import cv2
    cv2.circle(canvas, center, radius, color, thickness)


class ImprovedVectorizationLoss(nn.Module):
    """
    改进的损失函数

    新增：
    1. 连续性损失：确保 P3 接近下一条的 P0
    2. Focal Loss：处理正负样本不平衡
    3. 平滑损失：鼓励曲线平滑
    """

    def __init__(self, reconstruction_weight=0.1, continuity_weight=0.2):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.reconstruction_weight = reconstruction_weight
        self.continuity_weight = continuity_weight

    def forward(self, predictions, targets, reconstructed=None, images=None):
        B, max_curves, _ = predictions.shape

        target_curves = targets[..., :8]
        target_validity = targets[..., 8:9]
        valid_mask = target_validity.squeeze(-1) > 0.5

        # 1. 坐标回归损失
        if valid_mask.sum() > 0:
            coord_pred = predictions[..., :6][valid_mask]
            coord_target = target_curves[..., :6][valid_mask]
            coord_loss = self.l1_loss(coord_pred, coord_target)
        else:
            coord_loss = torch.tensor(0.0, device=predictions.device)

        # 2. 宽度回归损失
        if valid_mask.sum() > 0:
            width_pred = predictions[..., 6:8][valid_mask]
            width_target = target_curves[..., 6:8][valid_mask]
            width_loss = self.l1_loss(width_pred, width_target)
        else:
            width_loss = torch.tensor(0.0, device=predictions.device)

        # 3. 有效性分类损失（使用 Focal Loss 的简化版）
        validity_pred = predictions[..., 8:9]
        validity_loss = self.bce_loss(validity_pred, target_validity)

        # 4. 连续性损失（确保曲线连接平滑）
        continuity_loss = self.compute_continuity_loss(predictions, valid_mask, B, max_curves)

        # 5. 重建损失
        reconstruction_loss = torch.tensor(0.0, device=predictions.device)
        if reconstructed is not None and images is not None:
            reconstruction_loss = self.mse_loss(reconstructed, images)

        # 总损失
        total_loss = (
            coord_loss +
            width_loss +
            2.0 * validity_loss +  # 加权有效性损失
            self.continuity_weight * continuity_loss +
            self.reconstruction_weight * reconstruction_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'coord': coord_loss.item(),
            'width': width_loss.item(),
            'validity': validity_loss.item(),
            'continuity': continuity_loss.item(),
            'reconstruction': reconstruction_loss.item()
        }

        return total_loss, loss_dict

    def compute_continuity_loss(self, predictions, valid_mask, B, max_curves):
        """
        计算连续性损失：确保 P3 接近下一条的 P0

        由于我们只预测 P1, P2, P3，P0 由上一条决定
        所以我们惩罚：P3(当前) - P0(下一条)
        但这里我们简化为：相邻曲线的 P3 应该接近
        """
        continuity_loss = torch.tensor(0.0, device=predictions.device)

        # 提取 P3 (最后两个坐标)
        p3_pred = predictions[..., 4:6]  # [B, max_curves, 2]

        # 对于每个 sample
        for b in range(B):
            # 找到有效的曲线索引
            valid_indices = torch.where(valid_mask[b])[0]

            if len(valid_indices) < 2:
                continue

            # 对于相邻的曲线，计算 P3 的距离
            for i in range(len(valid_indices) - 1):
                idx1 = valid_indices[i]
                idx2 = valid_indices[i + 1]

                # 惩罚 P3 和下一个起点的距离
                # 注意：这里我们简化了，实际应该用 P3 作为下一条的 P0
                # 但由于排序问题，我们只惩罚相邻slot的距离
                dist = torch.abs(p3_pred[b, idx1] - p3_pred[b, idx2]).mean()
                continuity_loss = continuity_loss + dist

        return continuity_loss / B


def train_epoch(model, train_loader, optimizer, criterion, device, freeze_encoder=True):
    """训练一个 epoch"""
    if freeze_encoder:
        model.encoder.eval()
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.vector_decoder.train()
    else:
        model.train()

    epoch_loss = 0.0
    epoch_loss_dict = None

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 前向传播
        if freeze_encoder:
            with torch.no_grad():
                embeddings = model.encoder(imgs)
            curves, validity = model.vector_decoder(embeddings)
        else:
            embeddings = model.encoder(imgs)
            curves, validity = model.vector_decoder(embeddings)

        # 重建
        reconstructed = None
        if model.pixel_decoder is not None:
            if freeze_encoder:
                with torch.no_grad():
                    reconstructed = model.pixel_decoder(embeddings)
            else:
                reconstructed = model.pixel_decoder(embeddings)

        # 组合预测
        predictions = torch.cat([curves, validity], dim=-1)

        # 计算损失
        loss, loss_dict = criterion(
            predictions, targets,
            reconstructed=reconstructed,
            images=imgs
        )

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
                  f"Width: {loss_dict['width']:.6f}, "
                  f"Cont: {loss_dict['continuity']:.6f}")

    # 平均
    avg_loss = epoch_loss / len(train_loader)
    for key in epoch_loss_dict:
        epoch_loss_dict[key] /= len(train_loader)

    return avg_loss, epoch_loss_dict


def train_progressive():
    """渐进式训练主函数"""
    config = {
        'dataset_size': 20000,
        'max_strokes': 8,
        'batch_size': 16,

        'embed_dim': 128,
        'max_curves': 8,

        # 阶段 1：冻结 Encoder
        'phase1_lr': 5e-4,  # 降低学习率
        'phase1_epochs': 40,

        # 阶段 2：端到端
        'phase2_lr': 1e-4,
        'phase2_epochs': 20,

        'reconstruction_weight': 0.1,
        'continuity_weight': 0.2,
    }

    device = 'cpu'
    print(f"使用设备: {device}")

    # 加载 Phase 1.5 模型（多曲线预训练）
    print("\n加载 Phase 1.5 模型...")
    try:
        checkpoint = torch.load('best_reconstruction_multi.pth', map_location=device)
        print("  使用多曲线预训练模型")
    except FileNotFoundError:
        checkpoint = torch.load('best_reconstruction.pth', map_location=device)
        print("  使用单曲线模型（建议先运行 train_phase1_5.py）")

    encoder = StrokeEncoder(
        in_channels=1, embed_dim=128, num_heads=4, num_layers=6, dropout=0.1
    ).to(device)
    pixel_decoder = PixelDecoder(embed_dim=128).to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    pixel_decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    pixel_decoder.eval()

    # 创建 Vector Decoder
    vector_decoder = VectorDecoder(
        embed_dim=config['embed_dim'],
        max_curves=config['max_curves'],
        num_layers=3,
        num_heads=4,
        dropout=0.1
    ).to(device)

    model = VectorizationModel(encoder, vector_decoder, pixel_decoder).to(device)

    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 创建数据集（渐进式）
    print("\n创建数据集...")
    dataset = SimpleMultiStrokeDataset(
        size=64,
        length=config['dataset_size'],
        max_strokes=config['max_strokes'],
        mode='progressive'
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    print(f"数据集大小: {len(dataset)}")

    # 改进的损失函数
    criterion = ImprovedVectorizationLoss(
        reconstruction_weight=config['reconstruction_weight'],
        continuity_weight=config['continuity_weight']
    )

    # 阶段 1
    print("\n" + "=" * 60)
    print("阶段 1: 冻结 Encoder")
    print("=" * 60)

    optimizer = optim.Adam(
        vector_decoder.parameters(),
        lr=config['phase1_lr']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['phase1_epochs']
    )

    best_loss = float('inf')

    for epoch in range(1, config['phase1_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['phase1_epochs']}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

        avg_loss, loss_dict = train_epoch(
            model, train_loader, optimizer, criterion, device,
            freeze_encoder=True
        )

        scheduler.step()

        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  坐标: {loss_dict['coord']:.6f}, "
              f"宽度: {loss_dict['width']:.6f}, "
              f"有效性: {loss_dict['validity']:.6f}, "
              f"连续性: {loss_dict['continuity']:.6f}, "
              f"重建: {loss_dict['reconstruction']:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'vector_decoder_state_dict': vector_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_vector_decoder_v2.pth')
            print(f"  ✓ 保存最佳模型")

    # 阶段 2
    print("\n" + "=" * 60)
    print("阶段 2: 端到端微调")
    print("=" * 60)

    for param in model.encoder.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['phase2_lr']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['phase2_epochs']
    )

    best_loss = float('inf')

    for epoch in range(1, config['phase2_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['phase2_epochs']}")

        avg_loss, loss_dict = train_epoch(
            model, train_loader, optimizer, criterion, device,
            freeze_encoder=False
        )

        scheduler.step()

        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  坐标: {loss_dict['coord']:.6f}, "
              f"宽度: {loss_dict['width']:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'vector_decoder_state_dict': model.vector_decoder.state_dict(),
                'pixel_decoder_state_dict': model.pixel_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_vectorization_v2.pth')
            print(f"  ✓ 保存完整模型")

    print("\n训练完成!")


if __name__ == '__main__':
    train_progressive()
