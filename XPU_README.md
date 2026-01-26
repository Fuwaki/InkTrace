# Intel Iris Xe Graphics (XPU) PyTorch 训练环境

## 硬件配置

- **CPU**: Intel Core i5-13500H (16线程)
- **GPU**: Intel Iris Xe Graphics (12.13 GB 可用内存)

## 环境配置

### 虚拟环境
```bash
# 激活虚拟环境
source .venv/bin/activate

# 或重新安装
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 关键依赖
- Python 3.10.18
- PyTorch 2.7.0+xpu (带XPU支持)
- Intel Extension for PyTorch 2.7.0

## 性能测试结果

**简单CNN训练 (600批次)**
- **CPU**: 20.98s
- **XPU**: 9.25s
- **加速比**: 2.27x

Iris Xe Graphics 显著提升了训练速度！

## 使用示例

### 1. 检查设备
```bash
source .venv/bin/activate
python check_devices.py
```

### 2. 运行性能测试
```bash
python xpu_train_test.py
```

### 3. 训练MNIST
```bash
python train_mnist.py
```

## 代码示例

### 自动选择设备
```python
import torch

def get_device():
    if torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# 使用
device = get_device()
model = MyModel().to(device)

# 训练时记得同步
for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    # ... 训练代码 ...

    if device.type == "xpu":
        torch.xpu.synchronize()
```

### XPU信息查询
```python
import torch

# 检查可用性
print(f"XPU可用: {torch.xpu.is_available()}")

# 设备数量
print(f"设备数: {torch.xpu.device_count()}")

# 设备名称
print(f"设备名称: {torch.xpu.get_device_name(0)}")

# 设备属性
props = torch.xpu.get_device_properties(0)
print(f"总内存: {props.total_memory / 1024**3:.2f} GB")
print(f"计算单元: {props.max_compute_units}")
```

## 注意事项

1. **设备同步**: XPU训练后需要 `torch.xpu.synchronize()` 来确保操作完成
2. **内存管理**: Iris核显共享系统内存，训练时注意批次大小
3. **版本兼容**: 必须使用 xpu 版本的 PyTorch (torch==2.7.0+xpu)

## 故障排除

### XPU不可用
检查是否安装了正确的PyTorch版本：
```bash
pip list | grep torch
# 应该显示: torch 2.7.0+xpu
```

### 导入错误
确保安装了所有运行时依赖：
```bash
pip install torch==2.7.0+xpu torchvision --index-url https://download.pytorch.org/whl
```

## 下一步

- 尝试更大的模型
- 使用混合精度训练 (FP16)
- 探索更多 Intel Extension for PyTorch 的优化功能
