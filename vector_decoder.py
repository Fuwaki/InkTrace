import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorDecoder(nn.Module):
    """
    矢量解码器：将图像特征转换为贝塞尔曲线序列

    架构：
    - 使用固定的 Learned Slots（每个 slot 对应一条潜在的曲线）
    - Transformer Decoder：Slots attend to Encoder features
    - 并行输出：所有曲线同时预测

    输出格式：
    - 每个曲线包含 9 个值：[x1, y1, x2, y2, x3, y3, w_start, w_end, is_valid]
    - 坐标归一化到 [0, 1]
    - 宽度归一化到 [0, 1]（推理时乘以 10）
    """

    def __init__(
        self,
        embed_dim=128,      # 与 Encoder 一致
        max_curves=16,      # 最多输出 16 条曲线
        num_layers=3,       # Transformer Decoder 层数
        num_heads=4,        # 注意力头数
        dropout=0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_curves = max_curves

        # 1. Slot Embeddings：每个 slot 对应一条潜在的曲线
        # 加入位置编码，让模型知道"这是第几条曲线"
        self.slot_embed = nn.Parameter(torch.randn(1, max_curves, embed_dim) * 0.02)

        # 2. 位置编码（可选的周期性位置编码）
        self.register_buffer('position_ids', torch.arange(max_curves).expand((1, -1)))
        self.position_embeddings = nn.Embedding(max_curves, embed_dim)

        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN Transformer（更稳定）
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. 曲线参数预测头
        # 输出：[x1, y1, x2, y2, x3, y3, w_start, w_end]
        self.curve_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 8)
        )

        # 5. 有效性预测头（判断该 slot 是否对应真实曲线）
        self.validity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

        # 6. Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: [B, seq_len, embed_dim]  # 来自 Encoder 的特征序列

        Returns:
            curves: [B, max_curves, 8]  # 每个曲线的参数
            validity: [B, max_curves, 1]  # 每个 slot 的有效性概率
        """
        B = encoder_features.shape[0]
        device = encoder_features.device

        # 1. 初始化 Slot Queries + 位置编码
        slot_queries = self.slot_embed.expand(B, -1, -1)  # [B, max_curves, embed_dim]
        pos_embeds = self.position_embeddings(self.position_ids).expand(B, -1, -1)
        slot_queries = slot_queries + pos_embeds

        # 2. Transformer Decoder：Slots attend to Memory
        # 使用因果掩码，让第 i 个 slot 只能看到前 i 个 slots（可选）
        decoded_slots = self.decoder(
            tgt=slot_queries,           # [B, max_curves, embed_dim]
            memory=encoder_features,     # [B, 64, embed_dim]
        )  # [B, max_curves, embed_dim]

        # 3. Layer Norm
        decoded_slots = self.norm(decoded_slots)

        # 4. 预测曲线参数
        curves = self.curve_head(decoded_slots)  # [B, max_curves, 8]

        # 5. 预测有效性
        validity_logits = self.validity_head(decoded_slots)  # [B, max_curves, 1]
        validity = torch.sigmoid(validity_logits)

        return curves, validity


class VectorizationModel(nn.Module):
    """
    完整的矢量化模型：Encoder + Vector Decoder + (可选的) Pixel Decoder
    """

    def __init__(self, encoder, vector_decoder, pixel_decoder=None):
        super().__init__()
        self.encoder = encoder
        self.vector_decoder = vector_decoder
        self.pixel_decoder = pixel_decoder  # 用于计算重建损失

    def forward(self, x, return_pixels=False):
        """
        Args:
            x: [B, 1, 64, 64] 输入图像
            return_pixels: 是否返回重建图像

        Returns:
            if return_pixels:
                curves, validity, reconstructed
            else:
                curves, validity
        """
        # 1. 编码
        embeddings = self.encoder(x)  # [B, 64, embed_dim]

        # 2. 矢量解码
        curves, validity = self.vector_decoder(embeddings)

        # 3. 可选：像素重建（用于防止 Encoder 遗忘）
        if return_pixels and self.pixel_decoder is not None:
            reconstructed = self.pixel_decoder(embeddings)
            return curves, validity, reconstructed

        return curves, validity


class VectorizationLoss(nn.Module):
    """
    矢量化损失函数

    包含：
    1. 坐标回归损失（L1）
    2. 宽度回归损失（L1）
    3. 有效性分类损失（BCE）
    4. 可选：重建损失（MSE）
    """

    def __init__(self, reconstruction_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.reconstruction_weight = reconstruction_weight

    def forward(self, predictions, targets, reconstructed=None, images=None):
        """
        Args:
            predictions: [B, max_curves, 8]
            targets: [B, max_curves, 9]  # 前 8 个是参数，最后 1 个是有效性标志
            reconstructed: [B, 1, 64, 64] 重建图像（可选）
            images: [B, 1, 64, 64] 原始图像（可选）

        Returns:
            loss: 总损失
            loss_dict: 各项损失的字典
        """
        B, max_curves, _ = predictions.shape

        # 提取目标和有效性标志
        target_curves = targets[..., :8]  # [B, max_curves, 8]
        target_validity = targets[..., 8:9]  # [B, max_curves, 1]

        # 有效 mask（转换为 bool 类型）
        valid_mask = target_validity.squeeze(-1) > 0.5  # [B, max_curves]

        # 1. 坐标回归损失（只对有效曲线计算）
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

        # 3. 有效性分类损失
        # validity 已经在外部传入，不需要重新计算
        # 这里我们需要从 predictions 中提取 validity
        # predictions 的格式是 [curves (8维), validity (1维)] = 9维
        validity_pred = predictions[..., 8:9]  # 最后一维是有效性
        validity_loss = self.bce_loss(validity_pred, target_validity)

        # 4. 可选：重建损失
        reconstruction_loss = torch.tensor(0.0, device=predictions.device)
        if reconstructed is not None and images is not None:
            reconstruction_loss = self.mse_loss(reconstructed, images)

        # 总损失
        total_loss = (
            coord_loss +
            width_loss +
            validity_loss +
            self.reconstruction_weight * reconstruction_loss
        )

        # 记录各项损失
        loss_dict = {
            'total': total_loss.item(),
            'coord': coord_loss.item(),
            'width': width_loss.item(),
            'validity': validity_loss.item(),
            'reconstruction': reconstruction_loss.item()
        }

        return total_loss, loss_dict


def train_freeze_encoder(
    model, train_loader, optimizer, device, epoch,
    reconstruction_weight=0.1
):
    """
    阶段 1：冻结 Encoder，只训练 Vector Decoder
    """
    # 冻结 Encoder
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False

    model.vector_decoder.train()
    if model.pixel_decoder is not None:
        model.pixel_decoder.eval()

    criterion = VectorizationLoss(reconstruction_weight=reconstruction_weight)

    epoch_loss = 0.0
    epoch_loss_dict = {
        'coord': 0.0,
        'width': 0.0,
        'validity': 0.0,
        'reconstruction': 0.0
    }

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 前向传播
        with torch.no_grad():
            embeddings = model.encoder(imgs)

        curves, validity = model.vector_decoder(embeddings)

        # 如果有 Pixel Decoder，计算重建
        reconstructed = None
        if model.pixel_decoder is not None:
            with torch.no_grad():
                reconstructed = model.pixel_decoder(embeddings)

        # 组合预测：[B, max_curves, 9]
        predictions = torch.cat([curves, validity], dim=-1)

        # 计算损失
        loss, loss_dict = criterion(
            predictions, targets,
            reconstructed=reconstructed,
            images=imgs
        )

        # 反向传播（只更新 Vector Decoder）
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        for key in epoch_loss_dict:
            epoch_loss_dict[key] += loss_dict[key]

        # 每 50 个 batch 打印一次
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.6f}, "
                  f"Coord: {loss_dict['coord']:.6f}, "
                  f"Width: {loss_dict['width']:.6f}")

    # 平均损失
    avg_loss = epoch_loss / len(train_loader)
    for key in epoch_loss_dict:
        epoch_loss_dict[key] /= len(train_loader)

    return avg_loss, epoch_loss_dict


def train_end_to_end(
    model, train_loader, optimizer, device, epoch,
    reconstruction_weight=0.1
):
    """
    阶段 2：端到端训练，解冻 Encoder
    """
    model.train()
    criterion = VectorizationLoss(reconstruction_weight=reconstruction_weight)

    epoch_loss = 0.0
    epoch_loss_dict = {
        'coord': 0.0,
        'width': 0.0,
        'validity': 0.0,
        'reconstruction': 0.0
    }

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 前向传播
        embeddings = model.encoder(imgs)
        curves, validity = model.vector_decoder(embeddings)

        # 如果有 Pixel Decoder，计算重建
        reconstructed = None
        if model.pixel_decoder is not None:
            reconstructed = model.pixel_decoder(embeddings)

        # 组合预测：[B, max_curves, 9]
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
        for key in epoch_loss_dict:
            epoch_loss_dict[key] += loss_dict[key]

        # 每 50 个 batch 打印一次
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.6f}, "
                  f"Coord: {loss_dict['coord']:.6f}, "
                  f"Width: {loss_dict['width']:.6f}")

    # 平均损失
    avg_loss = epoch_loss / len(train_loader)
    for key in epoch_loss_dict:
        epoch_loss_dict[key] /= len(train_loader)

    return avg_loss, epoch_loss_dict
