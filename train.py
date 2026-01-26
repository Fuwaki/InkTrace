import torch
from torch.utils.data import DataLoader
from data import StrokeDataset


def main():
    # 创建数据集
    dataset = StrokeDataset(size=64, length=10000)

    # 用 DataLoader 封装
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 测试加载数据
    print(f"数据集大小: {len(dataset)}")
    print(f"批次数: {len(dataloader)}")

    for batch_idx, (imgs, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {imgs.shape}")  # [32, 1, 64, 64]
        print(f"  Labels shape: {labels.shape}")  # [32, 10]

        if batch_idx >= 1:  # 只打印前3个batch
            break


if __name__ == "__main__":
    main()
