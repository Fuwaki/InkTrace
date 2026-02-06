# InkTrace V5 训练指南

## 🎯 训练系统概述

InkTrace V5 使用 PyTorch Lightning 构建了一个完整的多阶段训练系统，支持：

- **统一配置管理**: YAML 文件定义全局默认 + 阶段覆盖
- **多阶段训练流水线**: structural → dense → finetune
- **Curriculum Learning**: 从简单样本逐渐过渡到复杂样本
- **自动 Checkpoint 管理**: Top-K + Last 保存策略
- **混合精度训练**: FP16 AMP 加速

---

## 📁 配置文件结构

配置文件采用 **默认值 + 覆盖** 的设计模式：

```yaml
# configs/default.yaml

# 全局默认配置（所有阶段的基础）
model:
  embed_dim: 128
  num_layers: 4
  full_heads: true

training:
  lr: 1e-3
  batch_size: 128
  epochs: 50
  # ... 更多参数

data:
  img_size: 64
  curriculum_stage: 0
  # ...

# 训练流水线
pipeline:
  order: ["structural", "dense"]
  auto_transfer: true  # 自动传递权重

# 阶段定义（覆盖默认配置）
stages:
  structural:
    model:
      full_heads: false  # 覆盖默认值
    training:
      epochs: 30
      lr: 1e-3
    # ...

  dense:
    init_from: "auto"    # 自动使用上一阶段最优
    training:
      epochs: 80
      lr: 5e-4
    curriculum:
      enabled: true
    # ...
```

---

## 🚀 使用方法

### 1. 单阶段训练

```bash
# 结构预训练
python train_pl.py --config configs/default.yaml --stage structural

# 密集预测训练
python train_pl.py --config configs/default.yaml --stage dense

# 从 checkpoint 初始化（迁移学习）
python train_pl.py --config configs/default.yaml --stage dense \
    --init_from checkpoints/structural/last.ckpt

# 断点续训
python train_pl.py --config configs/default.yaml --stage dense \
    --resume checkpoints/dense/last.ckpt
```

### 2. 多阶段自动训练

```bash
# 运行完整流水线（structural -> dense）
python train_pl.py --config configs/default.yaml --run-all-stages

# 从中间阶段恢复
python train_pl.py --config configs/default.yaml --run-all-stages \
    --start-from dense
```

### 3. 快速调试

```bash
# 使用 debug 阶段配置（小数据集，快速迭代）
python train_pl.py --config configs/default.yaml --stage debug
```

### 4. 命令行覆盖

```bash
# 覆盖学习率和训练轮数
python train_pl.py --config configs/default.yaml --stage dense \
    --lr 1e-4 --epochs 100 --batch_size 64
```

---

## 📊 训练阶段说明

### Phase 1: Structural Pretraining (结构预训练)

**目标**: 让 Encoder 学会从残缺输入推断完整结构

**方法**:
- Masking + Reconstruction（类似 MAE）
- 关闭跳连，强迫 Encoder 在 bottleneck 编码完整信息
- 只输出 skeleton + tangent

**推荐配置**:
- `epochs: 30`
- `lr: 1e-3`
- `mask_ratio: 0.6`
- `curriculum_stage: 2`（中等复杂度）

### Phase 2: Dense Prediction (密集预测)

**目标**: 训练完整的 5-head 密集预测

**方法**:
- 多任务学习 (Skeleton + Keypoints + Tangent + Width + Offset)
- 从 structural checkpoint 初始化
- 启用 Curriculum Learning

**推荐配置**:
- `epochs: 80`
- `lr: 5e-4`
- `curriculum: 0 -> 6, 10 epochs/stage`
- `loss_weights: skeleton=10, keypoints=5, tangent=2, width=1, offset=1`

### Phase 3: End-to-End Finetuning (可选)

**目标**: 全模型微调，适应极端情况

**方法**:
- 解冻 Encoder
- 低学习率全局微调
- 使用复杂数据

**推荐配置**:
- `epochs: 20`
- `lr: 1e-4`
- `curriculum_stage: 6`（复杂数据）

---

## 🎓 Curriculum Learning

渐进式训练从简单样本逐渐过渡到复杂样本：

| Stage | 描述 | 样本复杂度 |
|-------|------|------------|
| 0 | 单笔画 | ★☆☆☆☆ |
| 1-3 | 多独立笔画 | ★★☆☆☆ |
| 4-6 | 多段连续笔画 | ★★★☆☆ |
| 7-9 | 混合模式 | ★★★★★ |

配置示例：
```yaml
curriculum:
  enabled: true
  start_stage: 0
  end_stage: 6
  epochs_per_stage: 10  # 每 10 epoch 升级一次
```

---

## 📈 监控与可视化

### TensorBoard

```bash
tensorboard --logdir runs/
```

记录的指标：
- `train/loss`: 总训练损失
- `train/loss_skel`, `train/loss_keys`, etc.: 各任务损失
- `val/loss`: 验证损失
- `val/iou`, `val/precision`, `val/recall`, `val/f1`: 骨架分割指标
- `val/kp_topo_recall`, `val/kp_geo_recall`: 关键点召回率
- `curriculum/stage`: 当前 curriculum 阶段
- `train/grad_norm`: 梯度范数（每 100 步）

### 可视化回调

自动生成对比图：输入图像 | GT | 预测

配置：
```yaml
visualization:
  enabled: true
  num_samples: 4
  log_interval: 1  # 每个 epoch
```

---

## 💾 Checkpoint 管理

### 保存策略

```yaml
checkpoint:
  save_dir: "checkpoints/dense"
  keep_top_k: 3        # 保留最优的 3 个
  save_last: true      # 始终保存 last.ckpt
  monitor: "val/loss"  # 监控指标
  mode: "min"          # 越小越好
```

### 文件结构

```
checkpoints/
├── structural/
│   ├── epoch10-train_loss0.1234.ckpt
│   ├── epoch20-train_loss0.0987.ckpt
│   └── last.ckpt
└── dense/
    ├── epoch30-val_loss0.0567.ckpt
    ├── epoch40-val_loss0.0456.ckpt
    └── last.ckpt
```

---

## ⚙️ Loss 权重调优

各任务 Loss 的推荐权重：

| 任务 | 权重 | 说明 |
|------|------|------|
| skeleton | 10.0 | 最重要，骨架分割 |
| keypoints | 5.0 | 关键点检测 |
| tangent | 2.0 | 切向场，对曲线拟合重要 |
| width | 1.0 | 宽度预测 |
| offset | 1.0 | 亚像素偏移 |

---

## 🔧 常见问题

# 恢复训练（自动检测配置）
python train.py --resume checkpoints/structural/checkpoint_latest.pth

# 从 checkpoint 初始化新训练
python train.py --config configs/default.yaml --stage dense \
    --init_from checkpoints/structural/checkpoint_best.pth
```

---

## 📖 迁移指南

### 旧的训练方式（繁琐）

```bash
# 1. Structural pretrain
python train_structural.py --from-scratch \
    --embed-dim 128 --num-layers 4 --lr 1e-3 --epochs 30

# 2. Dense training（手动指定所有参数）
python train_dense.py \
    --init_from checkpoints/best_structural.pth \
    --embed_dim 128 --num_layers 4 \
    --lr 5e-4 --epochs 50 --stage 2
```

**问题**：
- ❌ 需要手动运行两个脚本
- ❌ 参数必须手动保持一致
- ❌ checkpoint 格式不一致
- ❌ 恢复训练需要记住所有参数

### 新的训练方式（简单）

```bash
# 方式 1：配置文件 + 单阶段
python train.py --config configs/default.yaml --stage structural

# 方式 2：一键运行所有阶段
python train.py --config configs/default.yaml --run-all-stages

# 方式 3：恢复训练（不需要记住参数！）
python train.py --resume checkpoints/structural/checkpoint_latest.pth
```

**优势**：
- ✅ 一条命令完成多阶段训练
- ✅ 配置统一管理，不会出错
- ✅ checkpoint 自动管理
- ✅ 恢复训练零参数

---

## 🛠️ 代码对比

### 旧代码：train_dense.py (362 行)

```python
# 1. 参数定义（40+ 行）
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=32)
# ... 15+ 个参数

# 2. Checkpoint 逻辑（50+ 行）
if args.resume:
    # 完全恢复
    checkpoint = torch.load(args.resume)
    config = checkpoint.get("config", {})
    embed_dim = config.get("embed_dim", args.embed_dim)
    # ... 复杂的加载逻辑
elif args.init_from:
    # 跨 stage 继续
    # ... 另一套逻辑

# 3. 训练循环（100+ 行）
for epoch in range(start_epoch, args.epochs):
    model.train()
    for imgs, targets in pbar:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Checkpoint 保存
    if avg_losses["total"] < best_loss:
        torch.save({...}, "best_dense_model.pth")
```

### 新代码：train.py (~300 行，包含两个 trainer)

```python
# 1. 配置加载（1 行）
config = Config.from_yaml(args.config)

# 2. Trainer 创建（几行）
trainer = DenseTrainer(config, init_from=args.init_from)

# 3. 训练（1 行）
trainer.train(dataloader)

# Checkpoint 管理？自动！
# - 定期保存
# - 最佳保存
# - 自动清理
```

---

## 📝 最佳实践总结

### 1. 配置管理

**✅ DO：** 使用 YAML 配置文件
```yaml
model:
  embed_dim: 128
  lr: 1e-3
```

**❌ DON'T：** 所有参数都用命令行
```bash
python train.py --embed_dim 128 --lr 1e-3 --batch_size 32 ...
```

### 2. Checkpoint 管理

**✅ DO：** 统一格式，完整保存
```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "config": full_config,
}
```

**❌ DON'T：** 只保存模型
```python
torch.save(model.state_dict(), "model.pth")  # 无法恢复训练！
```

### 3. 恢复训练

**✅ DO：** 自动检测配置
```bash
python train.py --resume checkpoints/latest.pth  # 从 checkpoint 读取配置
```

**❌ DON'T：** 手动指定参数
```bash
python train.py --resume checkpoints/latest.pth \
    --lr 1e-3 --batch_size 32 ...  # 容易出错！
```

### 4. 分阶段训练

**✅ DO：** 统一脚本管理
```python
for stage in config.stages:
    trainer = create_trainer(stage)
    trainer.train()
```

**❌ DON'T：** 分离的脚本
```bash
python train_structural.py ...
python train_dense.py ...  # 需要手动管理
```

---

## 🔧 如何迁移现有代码

### 步骤 1：创建配置文件

```bash
cp configs/default.yaml configs/my_experiment.yaml
# 编辑配置
```

### 步骤 2：使用新脚本

```bash
# 旧方式
python train_dense.py --lr 1e-3 --epochs 50 --stage 2

# 新方式
python train.py --config configs/my_experiment.yaml --stage dense
```

### 步骤 3：旧 checkpoint 兼容

新系统支持加载旧 checkpoint（会尝试兼容）：

```python
# 在 train.py 中
if args.resume:
    trainer.load_checkpoint(args.resume)  # 自动处理旧格式
```

---

## 💡 进阶功能

### 1. 配置继承

```yaml
# configs/base.yaml
model:
  embed_dim: 128
  num_layers: 4

training:
  lr: 1e-3
  weight_decay: 1e-4
```

```yaml
# configs/experiment1.yaml（继承 base）
extends: base.yaml

training:
  lr: 5e-4  # 覆盖学习率

data:
  batch_size: 64  # 新增配置
```

### 2. 超参数搜索

```bash
for lr in 1e-3 5e-4 1e-4; do
    python train.py --config configs/default.yaml \
        --stage dense --lr $lr \
        --save_dir sweeps/lr_$lr
done
```

### 3. 实验对比

```bash
# TensorBoard 对比多个实验
tensorboard --logdir runs/

# 或比较 checkpoint
python scripts/compare_checkpoints.py \
    checkpoints/exp1/best.pth \
    checkpoints/exp2/best.pth
```

---

## 📚 文件结构

```
InkTrace/
├── configs/
│   ├── default.yaml          # 默认配置
│   ├── structural.yaml        # Structural pretrain 配置
│   └── dense.yaml             # Dense training 配置
├── train_lib.py               # 训练框架（Config, CheckpointManager, BaseTrainer）
├── train.py                   # 统一训练脚本
├── train_structural.py        # 旧脚本（保留兼容）
├── train_dense.py             # 旧脚本（保留兼容）
└── TRAINING_GUIDE.md          # 本文档
```

---

## 🎓 推荐工作流

### 日常开发

```bash
# 1. 编辑配置
vim configs/my_experiment.yaml

# 2. 训练
python train.py --config configs/my_experiment.yaml --run-all-stages

# 3. 监控
tensorboard --logdir runs/

# 4. 如果训练中断
python train.py --resume checkpoints/latest.pth
```

### 实验管理

```bash
# 为每个实验创建配置
configs/
├── exp_baseline.yaml
├── exp_large_model.yaml
├── exp_high_lr.yaml
└── exp_long_train.yaml

# 运行多个实验
python train.py --config configs/exp_baseline.yaml --run-all-stages &
python train.py --config configs/exp_large_model.yaml --run-all-stages &
```

---

## ❓ 常见问题

**Q: 旧的 train_dense.py 还能用吗？**
A: 可以，新系统兼容。但建议迁移到新系统。

**Q: 如何恢复旧 checkpoint 到新系统？**
A: 直接使用 `--resume`，会自动兼容：
```bash
python train.py --resume checkpoints/old/best_model.pth
```

**Q: 配置文件和命令行参数冲突时？**
A: 命令行参数优先：
```bash
python train.py --config config.yaml --lr 1e-2  # 1e-2 会覆盖 config 中的 lr
```

**Q: 如何只运行某个阶段？**
A: 使用 `--stage`：
```bash
python train.py --config config.yaml --stage structural
```

---

## 🚀 下一步

1. **安装依赖**：新系统需要 PyYAML
   ```bash
   pip install pyyaml
   ```

2. **测试新系统**：
   ```bash
   python train.py --config configs/default.yaml --stage structural --epochs 1
   ```

3. **迁移配置**：将你的常用参数写入 YAML

4. **享受简化**！训练再也不会繁琐了 ✨
