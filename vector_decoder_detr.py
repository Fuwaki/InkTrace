"""
DETR 风格的矢量化解码器

核心特性：
1. Set Prediction：预测无序的笔画集合
2. Hungarian Matching：训练时找到最优的 Slot-GT 配对
3. 每个 Slot 预测一个完整的独立笔画
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


class DETRVectorDecoder(nn.Module):
    """
    DETR 风格的矢量化解码器

    架构：
    - Learnable Slot Queries：每个 Slot 负责预测一个笔画
    - Transformer Decoder：Slots attend to Encoder features
    - Prediction Head：输出笔画参数（P0, P1, P2, P3, w_start, w_end, is_valid）

    输出格式：
    - 每个笔画 10 个值：[x0, y0, x1, y1, x2, y2, x3, y3, w_start, w_end]
    - 外加一个有效性标志
    """

    def __init__(
        self,
        embed_dim=128,
        num_slots=8,      # 最多预测 8 个笔画
        num_layers=3,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_slots = num_slots

        # 1. Learnable Slot Queries
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, embed_dim) * 0.02)

        # 2. 位置编码（给 Slots 加上位置先验）
        self.register_buffer('position_ids', torch.arange(num_slots).expand((1, -1)))
        self.position_embeddings = nn.Embedding(num_slots, embed_dim)

        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. 笔画参数预测头
        # 输出 10 个值：x0, y0, x1, y1, x2, y2, x3, y3, w_start, w_end
        self.stroke_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 10)
        )

        # 5. 有效性预测头
        self.validity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

        # 6. Layer Norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: [B, seq_len, embed_dim]

        Returns:
            strokes: [B, num_slots, 10] 笔画参数
            validity: [B, num_slots, 1] 有效性
        """
        B = encoder_features.shape[0]
        device = encoder_features.device

        # 1. 初始化 Slot Queries + 位置编码
        slots = self.slot_queries.expand(B, -1, -1)  # [B, num_slots, embed_dim]
        pos_embeds = self.position_embeddings(self.position_ids).expand(B, -1, -1)
        slots = slots + pos_embeds

        # 2. Transformer Decoder
        decoded_slots = self.decoder(
            tgt=slots,               # [B, num_slots, embed_dim]
            memory=encoder_features  # [B, 64, embed_dim]
        )  # [B, num_slots, embed_dim]

        # 3. Layer Norm
        decoded_slots = self.norm(decoded_slots)

        # 4. 预测笔画参数
        strokes = self.stroke_head(decoded_slots)  # [B, num_slots, 10]

        # 归一化坐标到 [0, 1]
        strokes[..., :8] = torch.sigmoid(strokes[..., :8])
        # 归一化宽度到 [0, 10]
        strokes[..., 8:10] = torch.sigmoid(strokes[..., 8:10]) * 10.0

        # 5. 预测有效性
        validity_logits = self.validity_head(decoded_slots)
        validity = torch.sigmoid(validity_logits)  # [B, num_slots, 1]

        return strokes, validity


class DETRLoss(nn.Module):
    """
    DETR 风格的损失函数

    流程：
    1. Hungarian Matching：找到最优的 Slot-GT 配对
    2. 计算配对后的损失：
       - 坐标损失（L1）
       - 宽度损失（L1）
       - 有效性损失（BCE）
    """

    def __init__(self, coord_weight=5.0, width_weight=1.0, validity_weight=1.0):
        super().__init__()
        self.coord_weight = coord_weight
        self.width_weight = width_weight
        self.validity_weight = validity_weight

        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred_strokes, pred_validity, targets):
        """
        Args:
            pred_strokes: [B, num_slots, 10] 预测的笔画参数
            pred_validity: [B, num_slots, 1] 预测的有效性
            targets: [B, num_slots, 11] GT
                    前 10 维是笔画参数，最后 1 维是有效性标志

        Returns:
            loss: 总损失
            loss_dict: 各项损失的字典
        """
        B, num_slots, _ = pred_strokes.shape

        # 提取 GT
        gt_strokes = targets[..., :10]  # [B, num_slots, 10]
        gt_validity = targets[..., 10:11]  # [B, num_slots, 1]

        # 对于每个样本，进行 Hungarian Matching
        total_loss = 0.0
        loss_dict = {
            'coord': 0.0,
            'width': 0.0,
            'validity': 0.0,
            'matching_cost': 0.0
        }

        for b in range(B):
            # 当前样本的预测和 GT
            pred = pred_strokes[b]  # [num_slots, 10]
            pred_val = pred_validity[b]  # [num_slots, 1]
            gt = gt_strokes[b]  # [num_slots, 10]
            gt_val = gt_validity[b]  # [num_slots, 1]

            # 找到有效的 GT 笔画
            valid_gt_indices = torch.where(gt_val.squeeze(-1) > 0.5)[0]
            num_gt = len(valid_gt_indices)

            if num_gt == 0:
                # 没有 GT 笔画，只计算有效性损失
                validity_loss = self.bce_loss(pred_val, gt_val)
                total_loss += validity_loss
                loss_dict['validity'] += validity_loss.item()
                continue

            # 计算 Cost Matrix
            cost_matrix = self.compute_cost_matrix(
                pred, gt, pred_val, gt_val, valid_gt_indices
            )  # [num_slots, num_gt]

            # Hungarian Matching
            # 找到最优配对：最小化总 cost
            pred_indices, gt_indices = linear_sum_assignment(
                cost_matrix.detach().cpu().numpy()
            )  # pred_indices, gt_indices 都是长度为 num_gt 的数组

            # 根据 matching 计算损失
            # 将配对结果转为 tensor
            pred_indices = torch.from_numpy(pred_indices).to(pred.device)
            gt_indices = torch.from_numpy(gt_indices).to(gt.device)

            # 提取配对后的预测和 GT
            matched_pred = pred[pred_indices]  # [num_gt, 10]
            matched_gt = gt[gt_indices]  # [num_gt, 10]
            matched_pred_val = pred_val[pred_indices]  # [num_gt, 1]
            matched_gt_val = gt_val[gt_indices]  # [num_gt, 1]

            # 1. 坐标损失（前 8 个坐标）
            coord_loss = self.l1_loss(matched_pred[..., :8], matched_gt[..., :8])

            # 2. 宽度损失（后 2 个宽度）
            width_loss = self.l1_loss(matched_pred[..., 8:10], matched_gt[..., 8:10])

            # 3. 有效性损失
            validity_loss = self.bce_loss(matched_pred_val, matched_gt_val)

            # 对于没有匹配上的 GT，计算有效性损失（应该预测为 0）
            # 对于没有匹配上的预测，也要计算有效性损失
            # 这里简化处理：对所有 Slots 计算有效性损失
            global_validity_loss = self.bce_loss(pred_val, gt_val)

            # 总损失
            loss = (
                self.coord_weight * coord_loss +
                self.width_weight * width_loss +
                self.validity_weight * global_validity_loss
            )

            total_loss += loss
            loss_dict['coord'] += coord_loss.item()
            loss_dict['width'] += width_loss.item()
            loss_dict['validity'] += global_validity_loss.item()
            loss_dict['matching_cost'] += cost_matrix[pred_indices, gt_indices].sum().item()

        # 平均
        avg_loss = total_loss / B
        for key in loss_dict:
            loss_dict[key] /= B

        return avg_loss, loss_dict

    def compute_cost_matrix(self, pred, gt, pred_val, gt_val, valid_gt_indices):
        """
        计算预测和 GT 之间的 cost matrix

        Cost = 5 * L1(坐标) + 1 * L1(宽度) + 1 * BCE(有效性)

        Args:
            pred: [num_slots, 10] 所有 Slots 的预测
            gt: [num_slots, 10] 所有 GT（包括 padding）
            pred_val: [num_slots, 1] 预测的有效性
            gt_val: [num_slots, 1] GT 的有效性
            valid_gt_indices: 有效 GT 的索引

        Returns:
            cost_matrix: [num_slots, num_gt]
        """
        num_slots = pred.shape[0]
        num_gt = len(valid_gt_indices)

        # 提取有效的 GT
        valid_gt = gt[valid_gt_indices]  # [num_gt, 10]
        valid_gt_val = gt_val[valid_gt_indices]  # [num_gt, 1]

        # 初始化 cost matrix
        cost_matrix = torch.zeros(num_slots, num_gt, device=pred.device)

        for i in range(num_slots):
            for j in range(num_gt):
                # 坐标 cost
                coord_cost = torch.abs(pred[i, :8] - valid_gt[j, :8]).sum()

                # 宽度 cost
                width_cost = torch.abs(pred[i, 8:10] - valid_gt[j, 8:10]).sum()

                # 有效性 cost
                # 如果预测的有效性高，但 GT 不在这个位置，cost 应该高
                # 这里的 cost 只是用于 matching，简化处理
                validity_cost = 0.0  # 暂时不考虑

                # 总 cost
                cost_matrix[i, j] = (
                    5.0 * coord_cost +
                    1.0 * width_cost +
                    validity_cost
                )

        return cost_matrix


class IndependentStrokesDataset:
    """独立多笔画数据集（带 GT 标签）"""

    def __init__(self, size=64, length=10000, max_strokes=8):
        self.size = size
        self.length = length
        self.max_strokes = max_strokes

    def __len__(self):
        return self.length

    def get_bezier_point(self, t, points):
        p0, p1, p2, p3 = points
        return (
            (1 - t) ** 3 * p0 +
            3 * (1 - t) ** 2 * t * p1 +
            3 * (1 - t) * t**2 * p2 +
            t**3 * p3
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

        # 生成独立笔画
        canvas, strokes = self.generate_independent_strokes(num_strokes)

        # 转为 tensor
        img_tensor = torch.from_numpy(canvas).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        # 准备 GT
        target = self.prepare_target(strokes)

        return img_tensor, target

    def generate_independent_strokes(self, num_strokes):
        """生成多个独立的笔画"""
        scale = 2
        canvas_size = self.size * scale
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        strokes = []

        for _ in range(num_strokes):
            # 随机起点
            p0 = np.random.rand(2) * self.size * 0.8 + self.size * 0.1

            # 随机方向和长度
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            length = np.random.uniform(10, 25)

            # P3
            p3 = p0 + direction * length

            # P1, P2
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

            strokes.append({
                'points': points,
                'w_start': w_start,
                'w_end': w_end
            })

        # 下采样
        import cv2
        canvas = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return canvas, strokes

    def prepare_target(self, strokes):
        """
        准备 GT tensor

        Returns:
            target: [max_strokes, 11]
                    前 10 维：笔画参数
                    最后 1 维：有效性标志
        """
        num_strokes = len(strokes)
        target = np.zeros((self.max_strokes, 11), dtype=np.float32)

        for i in range(self.max_strokes):
            if i < num_strokes:
                stroke = strokes[i]
                points = stroke['points'] / self.size  # 归一化到 [0, 1]

                # P0, P1, P2, P3 的坐标
                target[i, :8] = points.flatten()
                target[i, 8:10] = [
                    stroke['w_start'] / 10.0,
                    stroke['w_end'] / 10.0
                ]
                target[i, 10] = 1.0  # 有效标志
            else:
                target[i, 10] = 0.0  # 无效

        return torch.from_numpy(target)


def train_detr_vectorization():
    """训练 DETR 风格的矢量化模型"""
    from model import StrokeEncoder

    config = {
        'dataset_size': 20000,
        'max_strokes': 8,
        'batch_size': 16,

        'num_slots': 8,

        # 阶段 1：冻结 Encoder
        'phase1_lr': 5e-4,
        'phase1_epochs': 50,

        # 阶段 2：端到端
        'phase2_lr': 1e-4,
        'phase2_epochs': 20,
    }

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
        embed_dim=128,
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

    # 测试
    imgs, targets = next(iter(train_loader))
    print(f"\nBatch 形状:")
    print(f"  Images: {imgs.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  有效笔画数: {(targets[..., 10].sum(dim=1)).long()}")

    # 损失函数
    criterion = DETRLoss(
        coord_weight=5.0,
        width_weight=1.0,
        validity_weight=1.0
    )

    # 训练
    print("\n" + "=" * 60)
    print("Phase 2: DETR 风格矢量化训练")
    print("=" * 60)

    best_loss = float('inf')

    # 阶段 1：冻结 Encoder
    print("\n阶段 1: 冻结 Encoder")
    optimizer = optim.Adam(detr_decoder.parameters(), lr=config['phase1_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['phase1_epochs'])

    for epoch in range(1, config['phase1_epochs'] + 1):
        detr_decoder.train()

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
        print("-" * 60)

    print("\n训练完成!")
    print(f"最佳损失: {best_loss:.6f}")

    return encoder, detr_decoder


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torch.optim as optim

    train_detr_vectorization()
