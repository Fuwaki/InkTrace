# 训练可视化集成说明

## 架构概览

```
vis_core.py (核心渲染层)
    ↓
lightning_vis.py (PyTorch Lightning Callback)
    ↓
train_pl.py + lightning_model.py (训练入口)
    ↓
TensorBoard (可视化展示)
```

## 使用方法

### 1. 启动训练（自动启用可视化）

```bash
# Dense 阶段训练（会自动生成对比图）
python train_pl.py --config configs/default.yaml --stage dense
```

### 2. 启动 TensorBoard

```bash
tensorboard --logdir runs
```

### 3. 查看可视化结果

在浏览器打开 `http://localhost:6006`，可以查看：

#### **Images 标签页**
- `Validation/Visualization`: 每个验证 epoch 的对比图
  - Input: 输入图像
  - GT Skel: Ground Truth 骨架
  - Pred Skel: 预测骨架
  - GT Tan: Ground Truth 切向场（HSV 彩色）
  - Pred Tan: 预测切向场
  - Overlay: 叠加对比（红色=预测，绿色=GT）

#### **Scalars 标签页**
- **Loss 指标**:
  - `train/loss`: 训练 loss
  - `val/loss`: 验证 loss
  - `train/loss_skel`, `train/loss_tan`, etc.

- **评估指标** (每个 validation epoch):
  - `val/iou`: 骨架 IoU
  - `val/precision`: 精确率
  - `val/recall`: 召回率
  - `val/f1`: F1 分数
  - `val/kp_topo_recall`: 拓扑关键点召回率
  - `val/kp_geo_recall`: 几何关键点召回率

- **Curriculum 进度**:
  - `curriculum/stage`: 当前训练阶段 (0-9)

## 配置选项

编辑 `configs/default.yaml`:

```yaml
training:
  # 可视化配置
  visualization:
    enabled: true           # 是否启用可视化
    num_samples: 4          # 每次可视化的样本数
    log_metrics: true       # 是否记录详细指标

  # 验证频率（控制可视化生成频率）
logging:
  vis_interval: 2           # 每 2 个 epoch 验证一次
```

## 离线可视化

训练完成后，可以用独立脚本生成高清图片：

```bash
# 生成 8 个样本的网格图
python visualize.py --ckpt checkpoints/dense/last.ckpt --samples 8

# 生成单样本详细可视化（4x5 网格）
python visualize.py --ckpt best.ckpt --samples 4 --detailed

# 计算 100 个样本的统计指标
python visualize.py --ckpt best.ckpt --stats-samples 100 --stage 5
```

## 技术细节

### VisualizationCallback 工作流程

1. 每个 `validation_epoch_end` 触发
2. 从 `trainer.val_dataloaders` 获取一个 batch
3. 调用 `pl_module(imgs)` 进行推理
4. 调用 `vis_core.create_grid_image()` 生成对比图
5. 转换为 Tensor 并通过 `logger.experiment.add_image()` 写入 TensorBoard
6. 调用 `vis_core.compute_metrics()` 计算指标并记录

### 分布式训练支持

- 使用 `torchmetrics` 的指标函数自动处理多 GPU 同步
- `self.log(..., sync_dist=True)` 确保分布式聚合正确

### 内存管理

- 所有 matplotlib figure 绘制后调用 `plt.close(fig)` 防止内存泄漏
- 使用 `torch.no_grad()` 避免验证时计算梯度
