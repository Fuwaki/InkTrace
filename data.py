import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class StrokeDataset(Dataset):
    def __init__(self, size=64, length=10000):
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def get_bezier_point(self, t, points):
        # 三次贝塞尔曲线公式 B(t)
        # points: [p0, p1, p2, p3], 每个 p 是 2D 点
        p0, p1, p2, p3 = points
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * p3
        )

    def render_stroke(self):
        # 1. 随机生成控制点 (P0, P1, P2, P3)
        # 为了保证线条主要在图中间，坐标在 [-10, 74] 之间随机，允许稍微出界
        points = np.random.rand(4, 2) * (self.size * 1.5) - (self.size * 0.25)

        # 2. 随机生成宽度 (Start Width, End Width)
        # 模拟笔画：有的头重脚轻，有的均匀
        w_start = np.random.uniform(1.0, 6.0)
        w_end = np.random.uniform(1.0, 6.0)

        # 创建画布 (使用 2倍超采样以获得更好的抗锯齿，最后缩小)
        scale = 2
        canvas_size = self.size * scale
        img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        # 3. 密集采样画圆 (Dense Circle Stamping)
        num_steps = 200  # 步数越多越平滑
        for i in range(num_steps):
            t = i / (num_steps - 1)

            # 计算当前坐标
            pt = self.get_bezier_point(t, points) * scale

            # 计算当前宽度 (简单的线性插值，你可以改为非线性)
            current_width = w_start + (w_end - w_start) * t

            # 绘制实心圆
            # cv2.circle(img, center, radius, color, thickness)
            cv2.circle(
                img, (int(pt[0]), int(pt[1])), int(current_width * scale / 2), 255, -1
            )

        # 4. 下采样回 64x64 (这就是高质量抗锯齿的来源)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return img, points, w_start, w_end

    def to_tensor(self, img, points, w_start, w_end):
        # 归一化图像到 0-1 并转为 Tensor
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        # 归一化坐标到 0-1 之间方便模型回归
        norm_points = points / self.size
        label = np.concatenate([norm_points.flatten(), [w_start / 10.0, w_end / 10.0]])

        return img_tensor, torch.tensor(label, dtype=torch.float32)

    def __getitem__(self, idx):

        img, points, w_start, w_end = self.render_stroke()
        return self.to_tensor(img, points, w_start, w_end)
