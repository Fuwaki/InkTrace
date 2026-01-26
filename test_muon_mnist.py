"""
MNIST + Muon 优化器测试
对比 Adam 和 Muon 在简单分类任务上的表现

注意：PyTorch 的 Muon 只支持 2D 参数（矩阵），不支持 4D 卷积核
因此这里使用 MLP（全连接网络）而不是 CNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt


# ==================== 模型定义 ====================
class SimpleMLP(nn.Module):
    """
    简单的 MLP 用于 MNIST 分类
    使用全连接层，因为 Muon 只支持 2D 参数
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        output = torch.log_softmax(x, dim=1)
        return output


# ==================== 训练函数 ====================
def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'  Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\n  Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy


# ==================== 主训练循环 ====================
def run_experiment(optimizer_name='adam', num_epochs=5, batch_size=64, lr=0.001, device='cpu'):
    """运行单个实验"""

    print(f"\n{'='*60}")
    print(f"开始实验: {optimizer_name.upper()}")
    print(f"  Device: {device.upper()}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"{'='*60}\n")

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./mnist_data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 模型
    model = SimpleMLP().to(device)

    # 优化器
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'muon':
        # Muon 只支持 2D 参数（权重），不支持 1D 参数（bias）
        # 所以使用混合优化器：Muon 优化权重，Adam 优化 bias
        weight_params = [p for n, p in model.named_parameters() if 'weight' in n]
        bias_params = [p for n, p in model.named_parameters() if 'bias' in n]

        optimizer = optim.Muon(weight_params, lr=lr * 100)
        # 为 bias 创建额外的 Adam 优化器
        bias_optimizer = optim.Adam(bias_params, lr=lr)

        # 包装成组合优化器
        class CombinedOptimizer:
            def __init__(self, main_opt, bias_opt):
                self.main_opt = main_opt
                self.bias_opt = bias_opt

            def zero_grad(self):
                self.main_opt.zero_grad()
                self.bias_opt.zero_grad()

            def step(self):
                self.main_opt.step()
                self.bias_opt.step()

        optimizer = CombinedOptimizer(optimizer, bias_optimizer)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # 训练
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'  Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Acc: {test_acc:.2f}%')

    total_time = time.time() - start_time
    print(f'\n  训练完成! 总耗时: {total_time:.2f}s')
    print(f'  最终测试准确率: {history["test_acc"][-1]:.2f}%')

    return history, total_time


# ==================== 对比实验 ====================
def compare_optimizers():
    """对比 Adam, Muon, SGD"""

    print("\n" + "="*60)
    print("MNIST 优化器对比实验")
    print("="*60)

    # 强制使用 CPU（避免 XPU 兼容性问题）
    device = 'cpu'

    # 配置
    num_epochs = 5
    batch_size = 64

    results = {}

    # # 实验 1: Adam
    # history_adam, time_adam = run_experiment('adam', num_epochs, batch_size, lr=0.001, device=device)
    # results['adam'] = {'history': history_adam, 'time': time_adam}

    # 实验 2: Muon
    history_muon, time_muon = run_experiment('muon', num_epochs, batch_size, lr=0.001, device=device)
    results['muon'] = {'history': history_muon, 'time': time_muon}

    # 实验 3: SGD (baseline)
    history_sgd, time_sgd = run_experiment('sgd', num_epochs, batch_size, lr=0.01, device=device)
    results['sgd'] = {'history': history_sgd, 'time': time_sgd}

    # 绘图对比
    plot_comparison(results)

    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for name, res in results.items():
        final_acc = res['history']['test_acc'][-1]
        print(f"  {name.upper():8s}: 最终准确率 {final_acc:5.2f}%, 耗时 {res['time']:5.2f}s")
    print("="*60)

    return results


def plot_comparison(results):
    """绘制对比图表"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 测试准确率对比
    for opt_name, res in results.items():
        axes[0].plot(res['history']['test_acc'], label=opt_name.upper(), marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Test Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 训练损失对比
    for opt_name, res in results.items():
        axes[1].plot(res['history']['train_loss'], label=opt_name.upper(), marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Train Loss')
    axes[1].set_title('Train Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 时间对比
    opt_names = list(results.keys())
    times = [results[name]['time'] for name in opt_names]
    axes[2].bar([name.upper() for name in opt_names], times)
    axes[2].set_ylabel('Time (s)')
    axes[2].set_title('Training Time Comparison')

    plt.tight_layout()
    plt.savefig('mnist_optimizer_comparison.png', dpi=100)
    print("\n对比图表已保存: mnist_optimizer_comparison.png")
    plt.show()


if __name__ == "__main__":
    # 快速测试单个优化器
    # run_experiment('muon', num_epochs=3, batch_size=64, lr=0.001)

    # 完整对比实验
    compare_optimizers()
