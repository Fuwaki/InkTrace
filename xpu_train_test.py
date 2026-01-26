#!/usr/bin/env python3
"""
简单的PyTorch训练脚本，测试Intel Iris Xe Graphics (XPU)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import datasets, transforms


class SimpleNet(nn.Module):
    """简单的CNN模型"""
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output


def create_dummy_data(batch_size=64, num_batches=100):
    """创建虚拟数据用于测试"""
    for _ in range(num_batches):
        yield torch.randn(batch_size, 1, 28, 28), torch.randint(0, 10, (batch_size,))


def train(device_name="xpu"):
    """训练函数"""
    print(f"\n{'='*60}")
    print(f"使用设备: {device_name.upper()}")
    print(f"{'='*60}")

    device = torch.device(device_name)

    # 创建模型
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    # 训练参数
    batch_size = 64
    num_batches = 200
    num_epochs = 3

    print(f"\n训练配置:")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 每epoch批次数: {num_batches}")
    print(f"  - Epoch数: {num_epochs}")
    print(f"  - 总训练批次: {num_batches * num_epochs}")

    # 预热
    print("\n预热运行...")
    for data, target in create_dummy_data(batch_size, 10):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 同步设备
    if device_name == "xpu":
        torch.xpu.synchronize()

    # 开始训练计时
    print(f"\n开始训练...")
    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(create_dummy_data(batch_size, num_batches)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 同步设备
        if device_name == "xpu":
            torch.xpu.synchronize()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches

        print(f"Epoch {epoch+1}/{num_epochs} - 耗时: {epoch_time:.2f}s - 平均Loss: {avg_loss:.4f}")

    total_time = time.time() - total_start
    print(f"\n总训练时间: {total_time:.2f}s")
    print(f"每批次平均时间: {total_time/(num_batches*num_epochs):.4f}s")

    return total_time


def benchmark():
    """对比CPU和XPU性能"""
    print("\n" + "="*60)
    print("PyTorch XPU 训练性能测试")
    print("="*60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"XPU可用: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU设备: {torch.xpu.get_device_name(0)}")
        print(f"XPU内存: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    results = {}

    # 测试XPU
    if torch.xpu.is_available():
        try:
            xpu_time = train("xpu")
            results["XPU"] = xpu_time
        except Exception as e:
            print(f"\nXPU训练失败: {e}")
            results["XPU"] = None

    # 测试CPU
    try:
        cpu_time = train("cpu")
        results["CPU"] = cpu_time
    except Exception as e:
        print(f"\nCPU训练失败: {e}")
        results["CPU"] = None

    # 性能对比
    print("\n" + "="*60)
    print("性能对比总结")
    print("="*60)
    if results.get("XPU") and results.get("CPU"):
        speedup = results["CPU"] / results["XPU"]
        print(f"CPU时间: {results['CPU']:.2f}s")
        print(f"XPU时间: {results['XPU']:.2f}s")
        print(f"加速比: {speedup:.2f}x")

        if speedup > 1:
            print(f"\n✓ XPU比CPU快 {speedup:.2f} 倍!")
        else:
            print(f"\n✗ XPU比CPU慢 {1/speedup:.2f} 倍")
    else:
        print("无法完成对比")


if __name__ == "__main__":
    benchmark()
