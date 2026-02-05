# 训练系统重构指南

## 🎯 问题总结

原来的 `train_structural.py` 和 `train_dense.py` 存在以下问题：

1. **Checkpoint 管理混乱**
   - 格式不一致（一个有 scheduler，一个没有）
   - 缺少定期保存
   - 没有自动清理
   - train_dense.py 甚至没保存 config

2. **配置管理分散**
   - 超参数在：命令行参数、checkpoint 内、硬编码
   - 没有配置文件
   - 跨脚本需要手动保持参数一致

3. **分阶段训练割裂**
   - 需要手动运行两个脚本
   - `--init_from` 和 `--resume` 容易混淆
   - 模型架构切换缺少验证

4. **大量代码重复**
   - 训练循环逻辑几乎完全一样
   - 参数定义、设备选择、dataloader 都是复制粘贴

---

## ✅ 新系统特性

### 1. 统一配置文件 (YAML)

```yaml
# configs/default.yaml
model:
  embed_dim: 128
  num_layers: 4

training:
  lr: 1e-3
  batch_size: 32
  epochs: 50
  save_interval: 5
  keep_last_n: 3

data:
  img_size: 64
  num_workers: 4

# 多阶段配置
stages:
  - name: "structural"
    epochs: 30
    model:
      full_heads: false
    training:
      mask_ratio: 0.6

  - name: "dense"
    epochs: 50
    init_from: "best_structural.pth"
    model:
      full_heads: true
```

### 2. 智能 checkpoint 管理

```python
# 自动功能：
✓ 定期保存（每 N epoch）
✓ 保存最佳模型
✓ 自动清理旧 checkpoint（保留最近 N 个）
✓ 完整保存：model + optimizer + scheduler + config
✓ 一键恢复训练
```

checkpoint 结构：
```python
{
    "version": 1,
    "epoch": 10,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "metric": 0.123,
    "config": {...},  # 完整配置
    "metadata": {"stage": "structural"},
    "timestamp": "2024-01-01T12:00:00",
}
```

### 3. 统一训练脚本

```bash
# 单阶段训练
python train.py --config configs/default.yaml --stage structural

# 多阶段自动训练
python train.py --config configs/default.yaml --run-all-stages

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
