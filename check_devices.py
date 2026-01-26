#!/usr/bin/env python3
"""检查可用的计算资源"""

import torch
import platform
import os

print("=" * 50)
print("系统信息")
print("=" * 50)
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python版本: {platform.python_version()}")
print(f"PyTorch版本: {torch.__version__}")

print("\n" + "=" * 50)
print("CPU信息")
print("=" * 50)
print(f"CPU核心数: {os.cpu_count()}")

print("\n" + "=" * 50)
print("GPU可用性检查")
print("=" * 50)
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备数: {torch.cuda.device_count()}")
    print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")

print(f"\nMPS可用 (Apple Silicon): {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

# 检查Intel XPU支持
try:
    import intel_extension_for_pytorch as ipex
    print(f"\nIPEX版本: {ipex.__version__}")
    print(f"XPU可用: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU设备数: {torch.xpu.device_count()}")
        for i in range(torch.xpu.device_count()):
            print(f"  XPU设备 {i}: {torch.xpu.get_device_name(i)}")
            # 获取设备属性
            device = torch.xpu.device(i)
            props = torch.xpu.get_device_properties(i)
            print(f"    总内存: {props.total_memory / 1024**3:.2f} GB")
            print(f"    计算单元: {props.max_compute_units}")
except ImportError:
    print("\nIPEX未安装")
except Exception as e:
    print(f"\nXPU检查出错: {e}")

print("\n" + "=" * 50)
print("推荐设备")
print("=" * 50)
device = torch.device("cpu")
try:
    if torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"推荐使用: Intel XPU (Iris核显)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"推荐使用: CUDA GPU")
    else:
        print(f"推荐使用: CPU ({os.cpu_count()}核心)")
except:
    print(f"推荐使用: CPU ({os.cpu_count()}核心)")

print(f"\n最终选择: {device}")
