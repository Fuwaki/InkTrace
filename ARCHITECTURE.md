# InkTrace 项目架构说明

## 🎯 核心设计理念

### 统一接口 + 模块化设计

```
训练/可视化脚本
        ↓
统一工厂类 (ModelFactory)
        ↓
完整的模型类 (ReconstructionModel / VectorizationModel)
        ↓
独立的模块 (Encoder / Decoder / Dataset / Loss)
```

---

## 📁 核心模块

### 1. 模型实现
```
model.py              # StrokeEncoder (RepViT + Transformer)
pixel_decoder.py      # PixelDecoder (重建解码器)
detr_decoder.py       # DETRVectorDecoder (DETR 风格矢量解码器)
RepVit.py             # RepViT Backbone
```

### 2. 统一接口 ⭐ 新增
```
models.py
├── ReconstructionModel    # Encoder + PixelDecoder
├── VectorizationModel     # Encoder + DETR Decoder + PixelDecoder
└── ModelFactory           # 工厂类（创建、加载、保存）
```

### 3. 数据集
```
datasets.py
├── StrokeDataset                     # 单笔画
├── MultiStrokeReconstructionDataset  # 连续多笔画
└── IndependentStrokesDataset         # 独立多笔画
```

### 4. 损失函数
```
losses.py
└── DETRLoss  # Hungarian Matching
```

---

## 🚀 使用示例

### 创建模型

```python
from models import ModelFactory

# 方式 1：从头创建
model = ModelFactory.create_vectorization_model(
    embed_dim=128,
    num_slots=8,
    device='xpu',
    include_pixel_decoder=True  # 是否包含 Pixel Decoder
)

# 方式 2：加载已有模型
model = ModelFactory.load_vectorization_model(
    'best_detr_vectorization.pth',
    device='xpu'
)
```

### 使用模型

```python
# 训练/推理
strokes, validity, reconstructed = model(images, mode='both')

# mode 选项：
# - 'vectorize':  只输出矢量
# - 'reconstruct': 只输出重建
# - 'both': 同时输出矢量和重建
```

### 冻结/解冻模块

```python
# 冻结 Encoder
model.freeze_encoder()

# 解冻 Encoder
model.unfreeze_encoder()

# 冻结 DETR Decoder
model.freeze_detr_decoder()

# 解冻 DETR Decoder
model.unfreeze_detr_decoder()
```

### 保存模型

```python
ModelFactory.save_model(
    model,
    'best_model.pth',
    epoch=50,
    loss=0.001,
    optimizer=optimizer
)
```

---

## 📝 训练流程

### Phase 1: 单笔画重建
```bash
jupyter notebook train_encoder.ipynb
```

### Phase 1.5: 连续多笔画重建
```bash
python train_phase1_5_v2.py
```

### Phase 1.6: 独立多笔画重建
```bash
python train_phase1_6_v2.py
```

### Phase 2: DETR 矢量化
```bash
python train_phase2_detr_v2.py
```

---

## 🎨 可视化

```bash
# Phase 1.5
python visualize_multi_stroke.py

# Phase 1.6
python visualize_independent_strokes.py

# Phase 2
python visualize_detr_v2.py
```

---

## 🔑 核心优势

### 1. 统一接口 ⭐
```python
# 旧方式（需要手动管理）
encoder = StrokeEncoder(...)
decoder = DETRVectorDecoder(...)
embeddings = encoder(images)
strokes, validity = decoder(embeddings)

# 新方式（统一接口）
model = ModelFactory.create_vectorization_model(...)
strokes, validity, reconstructed = model(images, mode='both')
```

### 2. 易于扩展
```python
# 添加新的 Decoder
class NewVectorizationModel(nn.Module):
    def __init__(self, encoder, new_decoder):
        self.encoder = encoder
        self.new_decoder = new_decoder

# 在 ModelFactory 中添加
@staticmethod
def create_new_model(...):
    return NewVectorizationModel(encoder, new_decoder)
```

### 3. 一致性
```python
# 训练
model = ModelFactory.load_vectorization_model(...)
model.freeze_encoder()  # 统一的接口

# 可视化
model = ModelFactory.load_vectorization_model(...)  # 相同的接口
strokes, validity, _ = model(images, mode='vectorize')
```

### 4. 灵活性
```python
# 可以根据需要选择模式
# 只需要矢量
strokes, validity, _ = model(x, mode='vectorize')

# 只需要重建
_, _, reconstructed = model(x, mode='reconstruct')

# 都需要
strokes, validity, reconstructed = model(x, mode='both')
```

---

## 📊 模型对比

| 特性 | 旧架构 | 新架构 |
|------|--------|--------|
| 统一接口 | ❌ 需要手动组合 | ✅ ModelFactory |
| 训练一致性 | ❌ 代码重复 | ✅ 统一的 freeze/unfreeze |
| 可扩展性 | ❌ 难以添加新 decoder | ✅ 易于扩展 |
| 加载/保存 | ❌ 分散在各处 | ✅ 统一在工厂类 |
| 灵活性 | ❌ 固定的输出 | ✅ 可选的 mode |

---

## 🎯 最佳实践

### 1. 始终使用 ModelFactory
```python
# ✅ 推荐
model = ModelFactory.load_vectorization_model(...)

# ❌ 不推荐
encoder = StrokeEncoder(...)
decoder = DETRVectorDecoder(...)
```

### 2. 使用 mode 参数
```python
# ✅ 推荐
strokes, _, _ = model(x, mode='vectorize')

# ❌ 不推荐
embeddings = model.encoder(x)
strokes, _ = model.detr_decoder(embeddings)
```

### 3. 使用统一的 freeze/unfreeze
```python
# ✅ 推荐
model.freeze_encoder()

# ❌ 不推荐
for param in model.encoder.parameters():
    param.requires_grad = False
model.encoder.eval()
```

---

## 📈 未来扩展

这个架构可以轻松支持：

1. **新的 Decoder**
   ```python
   class TransformerDecoder(nn.Module):
       ...

   class VectorizationModelV2(nn.Module):
       def __init__(self, encoder, transformer_decoder):
           ...
   ```

2. **多任务学习**
   ```python
   strokes, validity, reconstructed = model(x, mode='both')
   loss = vector_loss + 0.1 * reconstruction_loss
   ```

3. **不同的输出格式**
   ```python
   def forward(self, x, output_format='bezier'):
       if output_format == 'bezier':
           return strokes, validity
       elif output_format == 'spline':
           return spline_params
   ```

---

## 总结

这个新的架构设计：
- ✅ 统一的接口
- ✅ 更好的可维护性
- ✅ 更容易扩展
- ✅ 保持一致性
- ✅ 灵活的使用方式

完美符合你的要求！🎉
