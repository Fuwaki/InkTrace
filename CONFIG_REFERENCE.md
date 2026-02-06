# InkTrace V5 配置参数完全参考

本文档详细说明 `configs/default.yaml` 中所有参数的含义、默认值和代码对应关系。

---

## 📋 **参数分类总览**

```
configs/default.yaml
├── model/           # 模型架构参数
├── training/        # 训练超参数
├── data/            # 数据加载参数
├── logging/         # 日志配置
├── device/          # 硬件配置
├── pipeline/        # 多阶段训练流程
└── stages/          # 各阶段覆盖配置
```

---

## 1️⃣ **Model 模型参数**

### Encoder 配置 (StrokeEncoder)

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `embed_dim` | int | 128 | encoder.py:45 | Transformer embedding 维度 |
| `num_layers` | int | 6 | encoder.py:47 | Transformer Encoder 层数 |
| `num_heads` | int | 4 | encoder.py:46 | Attention 头数 (需整除 embed_dim) |
| `dropout` | float | 0.1 | encoder.py:48 | Dropout 率 |

**重要提示**：
- `embed_dim=128` 对于 5 个密集预测头可能不足，建议改为 **256**
- `num_layers=6` 是合理值，不要减少
- `num_heads=4` 配合 `embed_dim=128`，每个 head 的 dim=32

**参数关系**：
```
head_dim = embed_dim / num_heads
对于 embed_dim=128, num_heads=4: head_dim = 32 (标准值)
如果 embed_dim=256, 建议改为 num_heads=8: head_dim = 32
```

---

### Decoder 配置 (UniversalDecoder)

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `decoder_heads` | int | 64 | decoder.py:243 | Decoder head channels (固定) |
| `decoder_kernel` | int | 7 | decoder.py:236 | NeXtBlock 卷积核大小 (固定) |

**注意**：这些参数目前硬编码在代码中，yaml 中定义仅为文档目的。

---

### 训练模式

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `full_heads` | bool | true | models.py:179 | 是否输出全部 5 个预测头 |

- `false`: 只输出 skeleton + tangent (structural 阶段)
- `true`: 输出全部 5 个头 (dense 阶段)

---

### Structural 预训练配置

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `mask_ratio` | float | 0.6 | models.py:32 | 遮挡比例 (0.0-1.0) |
| `mask_strategy` | str | "block" | models.py:37 | 遮挡策略: "block" \| "random" |
| `mask_block_size` | int | 8 | models.py:41 | block 策略的块大小 (像素) |

**Mask 策略说明**：
- `"block"`: 随机遮挡若干矩形块（类似 MAE），**推荐**
- `"random"`: 随机像素遮挡

---

## 2️⃣ **Training 训练参数**

### 基础训练参数

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `lr` | float | 1e-3 | lightning_model.py:49 | 初始学习率 |
| `batch_size` | int | 128 | lightning_model.py | 批次大小 |
| `epochs` | int | 50 | train_pl.py:404 | 训练轮数 |
| `epoch_length` | int | 10000 | train_pl.py:303 | 每个 epoch 的样本数 |
| `weight_decay` | float | 1e-4 | lightning_model.py:50 | AdamW 权重衰减 |
| `grad_clip` | float | 1.0 | lightning_model.py:54 | 梯度裁剪阈值 (max norm) |

**重要计算**：
```
batches_per_epoch = epoch_length / batch_size
例如：10000 / 128 = 78 batches/epoch

total_batches = epochs * batches_per_epoch
例如：50 * 78 = 3900 batches
```

---

### 学习率调度器

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `scheduler.type` | str | "onecycle" | lightning_model.py:206 | 调度器类型 |
| `scheduler.warmup_epochs` | int | 2 | lightning_model.py:56 | 预热轮数 |
| `scheduler.pct_start` | float | 0.1 | lightning_model.py:57 | OneCycleLR warmup 占比 |

**调度器类型**：
- `"onecycle"`: OneCycleLR (推荐，训练效果最好)
- `"cosine"`: CosineAnnealingLR (适合微调)
- `"constant"`: 固定学习率 (调试用)

**OneCycleLR 参数计算**（lightning_model.py:206-215）：
```python
total_steps = trainer.estimated_stepping_batches  # 自动计算
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,                    # 最高学习率
    total_steps=total_steps,      # 总步数
    pct_start=pct_start,          # warmup 占比
    div_factor=25.0,              # 初始 lr = max_lr / 25
    final_div_factor=1e4,         # 最终 lr = max_lr / 1e4
)
```

---

### Checkpoint 管理

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `checkpoint.save_dir` | str | "checkpoints" | train_pl.py:321 | 保存目录 |
| `checkpoint.keep_top_k` | int | 3 | train_pl.py:334 | 保留 top-k 最优 |
| `checkpoint.save_last` | bool | true | train_pl.py:337 | 保存 last.ckpt |
| `checkpoint.monitor` | str | "val/loss" | train_pl.py:329 | 监控指标 |
| `checkpoint.mode` | str | "min" | train_pl.py:335 | "min" 或 "max" |

**监控指标说明**：
- `"train/loss"`: 训练 loss (structural 阶段，无验证集)
- `"val/loss"`: 验证 loss (dense 阶段)

---

### Loss 权重 (Dense 阶段)

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `loss_weights.skeleton` | float | 10.0 | losses.py | 骨架 loss 权重 |
| `loss_weights.keypoints` | float | 5.0 | losses.py | 关键点 loss 权重 |
| `loss_weights.tangent` | float | 2.0 | losses.py | 切向场 loss 权重 |
| `loss_weights.width` | float | 1.0 | losses.py | 宽度 loss 权重 |
| `loss_weights.offset` | float | 1.0 | losses.py | 偏移 loss 权重 |

**总 Loss 计算**（losses.py）：
```python
L_total = 10.0 * L_skeleton + 5.0 * L_keypoints + 2.0 * L_tangent
          + 1.0 * L_width + 1.0 * L_offset
```

---

### Curriculum Learning

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `curriculum.enabled` | bool | false | train_pl.py:353 | 是否启用 |
| `curriculum.start_stage` | int | 0 | train_pl.py:377 | 起始阶段 (0-9) |
| `curriculum.end_stage` | int | 6 | train_pl.py:378 | 结束阶段 (0-9) |
| `curriculum.epochs_per_stage` | int | 10 | train_pl.py:379 | 每阶段轮数 |

**Curriculum Stages 说明**：
- Stage 0: 单笔画 (最简单)
- Stage 1-3: 多独立笔画 (递增: 1-3, 2-5, 3-8 笔画)
- Stage 4-6: 多段连续笔画 (递增: 2-3, 3-5, 4-8 段)
- Stage 7-9: 混合模式 (多条多段路径, 最复杂)

**自动升级机制**（lightning_model.py:387-405）：
```python
target_stage = start_stage + (current_epoch // epochs_per_stage)
target_stage = min(target_stage, end_stage)
```

---

### 可视化配置

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `visualization.enabled` | bool | true | train_pl.py:369 | 是否启用 |
| `visualization.num_samples` | int | 4 | train_pl.py:371 | 每次可视化样本数 |
| `visualization.log_interval` | int | 1 | train_pl.py:373 | 每 N epoch 可视化 |
| `visualization.log_metrics` | bool | true | train_pl.py:372 | 记录 IoU/Precision/Recall |

---

## 3️⃣ **Data 数据参数**

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `img_size` | int | 64 | train_pl.py:482 | 图像尺寸 (方形) |
| `batch_size` | int | 128 | train_pl.py:483 | 批次大小 (与 training 保持一致) |
| `num_workers` | int | 8 | train_pl.py:486 | DataLoader worker 数量 |
| `pin_memory` | bool | true | train_pl.py:488 | 是否使用 pin_memory |
| `persistent_workers` | bool | true | train_pl.py:489 | 保持 worker 进程 |
| `rust_threads` | int\|null | null | train_pl.py:487 | Rust 生成器线程数 (null=自动) |
| `curriculum_stage` | int | 0 | train_pl.py:485 | 初始 curriculum 阶段 |
| `keypoint_sigma` | float | 1.5 | train_pl.py:490 | 高斯热力图标准差 |

**高斯热力图说明**（dense_heads.py:149-160）：
```python
GT heatmap = exp(-((x-x0)^2 + (y-y0)^2) / (2 * sigma^2))
```
- `sigma=1.5`: 标准配置
- `sigma=2.0`: 更平滑的抗噪声配置
- `sigma=1.0`: 更锐利的精确定位

---

## 4️⃣ **Logging & Device**

### Logging

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `logging.log_interval` | int | 10 | train_pl.py:417 | 每 N step 记录一次 |
| `logging.tensorboard_dir` | str | "runs" | train_pl.py:308 | TensorBoard 目录 |

---

### Device

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `device.accelerator` | str | "auto" | train_pl.py:405 | 加速器类型 |
| `device.precision` | str | "16-mixed" | train_pl.py:406 | 精度模式 |

**加速器类型**：
- `"auto"`: 自动检测 (推荐)
- `"gpu"`: NVIDIA GPU
- `"cpu"`: CPU (精度自动降为 32)
- `"mps"`: Apple Silicon GPU

**精度模式**：
- `"32"`: 32 位浮点 (FP32)
- `"16-mixed"`: 混合精度 (FP16 + FP32)，推荐
- `"bf16-mixed"`: BFloat16 混合精度 (新 GPU)

---

## 5️⃣ **Pipeline 多阶段训练**

| 参数 | 类型 | 默认值 | 代码位置 | 说明 |
|------|------|--------|----------|------|
| `pipeline.order` | list | ["structural", "dense"] | train_pl.py:557 | 训练顺序 |
| `pipeline.auto_transfer` | bool | true | train_pl.py:558 | 自动权重传递 |

**权重传递逻辑**（train_pl.py:577-595）：
```python
for stage in pipeline.order:
    init_from = last_best_checkpoint  # 上一阶段的最优 checkpoint
    best_ckpt = run_stage(config, stage, init_from=init_from)
    last_best_checkpoint = best_ckpt  # 传递给下一阶段
```

---

## 6️⃣ **Stages 阶段配置**

每个阶段可以覆盖全局默认值，支持深度合并。

### 阶段级特殊参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `init_from` | str\|"auto" | 初始化权重路径 ("auto" = 自动传递) |
| `freeze_encoder` | bool | 是否冻结 Encoder 权重 |
| `description` | str | 阶段描述 |

---

## 🔧 **参数优先级**

```
命令行参数 > 阶段配置 > 全局默认值
```

**示例**（train_pl.py:685-690）：
```python
# 命令行覆盖
python train_pl.py --lr 2e-4 --batch_size 64

# 等价于修改 yaml
training:
  lr: 2e-4
  batch_size: 64
```

---

## 📊 **推荐配置**

### 🔴 **高性能配置** (GPU充足)

```yaml
model:
  embed_dim: 256          # 提升特征容量
  num_heads: 8            # 匹配 256 维

training:
  epoch_length: 80000     # 8x 数据量
  batch_size: 64          # 为更大 embed_dim 腾空间

data:
  img_size: 128           # 4x 分辨率
  keypoint_sigma: 2.0     # 补偿分辨率提升
```

**预期效果**：
- 参数量: 700K → 1.1M
- 训练时间: ~2x
- 精度: 显著提升

---

### 🟢 **轻量配置** (CPU推理)

```yaml
model:
  embed_dim: 128          # 保持轻量
  num_layers: 6

training:
  epoch_length: 50000     # 5x 数据量
  epochs: 150             # 3x 训练轮数
  lr: 5e-4                # 保守学习率
```

**预期效果**：
- 参数量: ~700K
- CPU 推理: ~10ms/image
- 精度: 良好 (通过更长训练补偿)

---

### 🟡 **调试配置**

```yaml
model:
  embed_dim: 64           # 极小模型
  num_layers: 2

training:
  epoch_length: 1000      # 快速迭代
  epochs: 5
  batch_size: 32
```

---

## ⚠️ **常见陷阱**

### 1. embed_dim 与 num_heads 不匹配

❌ **错误**：
```yaml
model:
  embed_dim: 128
  num_heads: 8    # 128 / 8 = 16 (太小)
```

✅ **正确**：
```yaml
model:
  embed_dim: 128
  num_heads: 4     # 128 / 4 = 32 (标准)
```

---

### 2. batch_size 与 epoch_length 不匹配

❌ **错误**：
```yaml
training:
  batch_size: 128
  epoch_length: 100   # 只有 0.78 batch/epoch (太少)
```

✅ **正确**：
```yaml
training:
  batch_size: 128
  epoch_length: 10000  # 78 batches/epoch (合理)
```

---

### 3. mask_ratio 未放在 model 下

❌ **旧配置**：
```yaml
training:
  mask_ratio: 0.6   # 位置错误
```

✅ **新配置**：
```yaml
model:
  mask_ratio: 0.6   # 正确位置
```

---

## 📌 **总结**

### ✅ 已修复的问题

1. ✅ 统一 `num_layers` 默认值为 **6** (yaml + encoder.py + models.py)
2. ✅ 统一 `embed_dim` 默认值为 **128** (yaml + encoder.py + models.py)
3. ✅ 将 `mask_ratio` 和 `mask_strategy` 移到 `model` 下
4. ✅ 移除未使用的 `min_lr` 和 `warmup_start_lr`
5. ✅ 添加 `mask_block_size` 配置
6. ✅ 添加 `decoder_heads` 和 `decoder_kernel` 文档说明

### 🎯 参数一致性检查表

| 参数 | yaml | encoder.py | models.py | train_pl.py | 状态 |
|------|------|------------|-----------|-------------|------|
| embed_dim | 128 | 128 | 128 | 128 | ✅ |
| num_layers | 6 | 6 | 6 | 6 | ✅ |
| num_heads | 4 | 4 | 4 | 4 | ✅ |
| mask_ratio | model | - | 0.6 | model | ✅ |

### 🚀 下一步优化建议

1. **P0**: 将 `embed_dim` 提升到 **256**
2. **P0**: 将 `epoch_length` 提升到 **50k+**
3. **P1**: 考虑提升 `img_size` 到 **128**
4. **P2**: 添加更多可配置参数（如 `stem_channels`）
