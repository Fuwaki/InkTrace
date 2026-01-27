# InkTrace 项目文件结构

## 📁 核心文件组织

### 模型实现
```
├── model.py                    # StrokeEncoder (RepViT + Transformer)
├── pixel_decoder.py            # PixelDecoder (重建解码器)
├── detr_decoder.py             # DETRVectorDecoder (DETR 风格矢量解码器)
└── RepVit.py                   # RepViT Backbone
```

### 数据集
```
└── datasets.py                 # 所有数据集类
    ├── StrokeDataset           # 单笔画 (Phase 1)
    ├── MultiStrokeReconstructionDataset  # 连续多笔画 (Phase 1.5)
    └── IndependentStrokesDataset        # 独立多笔画 (Phase 1.6, 2)
```

### 损失函数
```
└── losses.py                   # DETRLoss (Hungarian Matching)
```

### 训练入口
```
├── train_encoder.ipynb         # Phase 1: 单笔画重建
├── train_phase1_5.py           # Phase 1.5: 连续多笔画重建
├── train_phase1_6.py           # Phase 1.6: 独立多笔画重建
└── train_phase2_detr.py        # Phase 2: DETR 矢量化
```

### 可视化
```
├── visualize_multi_stroke.py           # Phase 1.5 可视化
├── visualize_independent_strokes.py    # Phase 1.6 可视化
└── visualize_detr.py                   # Phase 2 可视化
```



## 🚀 使用流程

### 训练流程
```bash
# Phase 1: 单笔画重建
jupyter notebook train_encoder.ipynb

# Phase 1.5: 连续多笔画重建
python train_phase1_5.py

# Phase 1.6: 独立多笔画重建
python train_phase1_6.py

# Phase 2: DETR 矢量化
python train_phase2_detr.py
```

### 可视化
```bash
# Phase 1.5 可视化
python visualize_multi_stroke.py

# Phase 1.6 可视化
python visualize_independent_strokes.py

# Phase 2 可视化
python visualize_detr.py
```

---

## 📊 模型文件

训练后会生成以下模型文件：

```
best_reconstruction.pth              # Phase 1 模型
best_reconstruction_multi.pth        # Phase 1.5 模型
best_reconstruction_independent.pth  # Phase 1.6 模型
best_detr_vectorization.pth          # Phase 2 模型
```

---

## 🎯 核心改进

### 1. 模块化
- 每个模块职责单一
- 易于维护和扩展

### 2. 可复用
- 数据集、模型、损失函数分离
- 不同的训练脚本可以共享组件

### 3. 清晰的训练流程
- 每个阶段有独立的训练脚本
- 明确的输入输出

### 4. 完整的可视化
- 每个阶段有专门的可视化脚本
- 便于调试和评估
