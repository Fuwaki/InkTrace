"""
数据集模块

包含所有数据集类：
- StrokeDataset: 单笔画数据集
- MultiStrokeReconstructionDataset: 连续多笔画重建数据集（Phase 1.5）
- IndependentStrokesDataset: 独立多笔画数据集（Phase 1.6, Phase 2）
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class StrokeDataset(Dataset):
    """
    单笔画数据集（Phase 1 使用）

    生成单条贝塞尔曲线的笔画
    """

    def __init__(self, size=64, length=10000):
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def get_bezier_point(self, t, points):
        # 三次贝塞尔曲线公式 B(t)
        p0, p1, p2, p3 = points
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * p3
        )

    def render_stroke(self):
        # 1. 随机生成控制点
        points = np.random.rand(4, 2) * (self.size * 1.5) - (self.size * 0.25)

        # 2. 随机生成宽度
        w_start = np.random.uniform(1.0, 6.0)
        w_end = np.random.uniform(1.0, 6.0)

        # 创建画布（2倍超采样）
        scale = 2
        canvas_size = self.size * scale
        img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        # 3. 密集采样画圆
        num_steps = 200
        for i in range(num_steps):
            t = i / (num_steps - 1)
            pt = self.get_bezier_point(t, points) * scale
            current_width = w_start + (w_end - w_start) * t

            cv2.circle(
                img, (int(pt[0]), int(pt[1])), int(current_width * scale / 2), 255, -1
            )

        # 4. 下采样
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return img, points, w_start, w_end

    def to_tensor(self, img, points, w_start, w_end):
        # 归一化图像
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        # 归一化坐标和宽度
        norm_points = points / self.size
        label = np.concatenate([norm_points.flatten(), [w_start / 10.0, w_end / 10.0]])

        return img_tensor, torch.tensor(label, dtype=torch.float32)

    def __getitem__(self, idx):
        img, points, w_start, w_end = self.render_stroke()
        return self.to_tensor(img, points, w_start, w_end)


class MultiStrokeReconstructionDataset(Dataset):
    """
    连续多笔画重建数据集（Phase 1.5 使用）

    关键：笔画是连续的，P0接上一条的P3
    """

    def __init__(self, size=64, length=10000, max_strokes=8):
        self.size = size
        self.length = length
        self.max_strokes = max_strokes

    def __len__(self):
        return self.length

    def get_bezier_point(self, t, points):
        p0, p1, p2, p3 = points
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * p3
        )

    def __getitem__(self, idx):
        # 渐进式增加曲线数量
        progress = idx / self.length
        if progress < 0.3:
            num_strokes = 1
        elif progress < 0.6:
            num_strokes = np.random.randint(1, 4)
        else:
            num_strokes = np.random.randint(1, self.max_strokes + 1)

        # 生成连续的多笔画
        scale = 2
        canvas_size = self.size * scale
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        current_point = np.random.rand(2) * self.size

        for _ in range(num_strokes):
            p0 = current_point

            # 生成方向和长度
            direction = np.random.randn(2)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            length = np.random.uniform(8, 20)

            p3 = p0 + direction * length

            p1 = (
                p0
                + direction * length * np.random.uniform(0.2, 0.4)
                + np.random.randn(2) * 3
            )
            p2 = (
                p0
                + direction * length * np.random.uniform(0.6, 0.8)
                + np.random.randn(2) * 3
            )

            points = np.stack([p0, p1, p2, p3])

            w_start = np.random.uniform(2.0, 5.0)
            w_end = np.random.uniform(2.0, 5.0)

            # 渲染
            num_steps = 200
            for i in range(num_steps):
                t = i / (num_steps - 1)
                pt = self.get_bezier_point(t, points) * scale
                current_width = w_start + (w_end - w_start) * t

                cv2.circle(
                    canvas,
                    (int(pt[0]), int(pt[1])),
                    int(current_width * scale / 2),
                    255,
                    -1,
                )

            current_point = p3

        # 下采样
        canvas = cv2.resize(
            canvas, (self.size, self.size), interpolation=cv2.INTER_AREA
        )

        img_tensor = torch.from_numpy(canvas).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        return img_tensor, img_tensor.clone()


class IndependentStrokesDataset(Dataset):
    """
    独立多笔画数据集（Phase 1.6, Phase 2 使用）

    关键：每个笔画是独立的，不连接
    用于 DETR 风格的 Set Prediction
    """

    def __init__(self, size=64, length=10000, max_strokes=8):
        self.size = size
        self.length = length
        self.max_strokes = max_strokes

    def __len__(self):
        return self.length

    def get_bezier_point(self, t, points):
        p0, p1, p2, p3 = points
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * p3
        )

    def __getitem__(self, idx):
        # 渐进式增加笔画数量
        progress = idx / self.length
        if progress < 0.3:
            num_strokes = np.random.randint(1, 3)
        elif progress < 0.6:
            num_strokes = np.random.randint(1, 5)
        else:
            num_strokes = np.random.randint(1, self.max_strokes + 1)

        # 生成独立笔画
        canvas, strokes = self.generate_independent_strokes(num_strokes)

        # 转为 tensor
        img_tensor = torch.from_numpy(canvas).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 64, 64]

        # 准备 GT
        target = self.prepare_target(strokes)

        return img_tensor, target

    def generate_independent_strokes(self, num_strokes):
        """生成多个独立的笔画"""
        scale = 2
        canvas_size = self.size * scale
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        strokes = []

        for _ in range(num_strokes):
            # 每个笔画完全独立，随机起点
            p0 = np.random.rand(2) * self.size * 0.8 + self.size * 0.1

            # 随机方向和长度
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            length = np.random.uniform(10, 25)

            # P3
            p3 = p0 + direction * length

            # P1, P2
            p1 = (
                p0
                + direction * length * np.random.uniform(0.2, 0.4)
                + np.random.randn(2) * 2
            )
            p2 = (
                p0
                + direction * length * np.random.uniform(0.6, 0.8)
                + np.random.randn(2) * 2
            )

            points = np.stack([p0, p1, p2, p3])

            # 随机宽度
            w_start = np.random.uniform(2.0, 5.0)
            w_end = np.random.uniform(2.0, 5.0)

            # 渲染
            num_steps = 200
            for i in range(num_steps):
                t = i / (num_steps - 1)
                pt = self.get_bezier_point(t, points) * scale
                current_width = w_start + (w_end - w_start) * t

                cv2.circle(
                    canvas,
                    (int(pt[0]), int(pt[1])),
                    int(current_width * scale / 2),
                    255,
                    -1,
                )

            strokes.append({"points": points, "w_start": w_start, "w_end": w_end})

        # 按 P0 的 Y 坐标排序，使得 GT 顺序相对固定，减少匈牙利匹配的震荡
        strokes.sort(key=lambda s: s["points"][0, 1])

        # 下采样
        canvas = cv2.resize(
            canvas, (self.size, self.size), interpolation=cv2.INTER_AREA
        )

        return canvas, strokes

    def prepare_target(self, strokes):
        """
        准备 GT tensor

        Returns:
            target: [max_strokes, 11]
                    前 10 维：笔画参数 (P0, P1, P2, P3, w_start, w_end)
                    最后 1 维：有效性标志
        """
        num_strokes = len(strokes)
        target = np.zeros((self.max_strokes, 11), dtype=np.float32)

        for i in range(self.max_strokes):
            if i < num_strokes:
                stroke = strokes[i]
                points = stroke["points"] / self.size  # 归一化到 [0, 1]

                # P0, P1, P2, P3 的坐标
                target[i, :8] = points.flatten()
                target[i, 8:10] = [stroke["w_start"] / 10.0, stroke["w_end"] / 10.0]
                target[i, 10] = 1.0  # 有效标志
            else:
                target[i, 10] = 0.0  # 无效

        return torch.from_numpy(target)
