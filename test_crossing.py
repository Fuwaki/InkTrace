#!/usr/bin/env python3
"""
测试交叉笔画的重建效果

创建简单的合成数据（十字形），测试算法是否能正确穿越交叉点
"""

import numpy as np
import matplotlib.pyplot as plt
from graph_reconstruction import SimpleGraphReconstructor, fit_bezier_curves


def create_cross_skeleton(size=64, thickness=1):
    """创建十字形骨架"""
    skeleton = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    # 水平线
    skeleton[center, 10:size-10] = 1.0
    # 垂直线
    skeleton[10:size-10, center] = 1.0
    
    return skeleton


def create_x_skeleton(size=64):
    """创建X形骨架"""
    skeleton = np.zeros((size, size), dtype=np.float32)
    
    # 对角线1 (左上到右下)
    for i in range(10, size-10):
        skeleton[i, i] = 1.0
    
    # 对角线2 (右上到左下)
    for i in range(10, size-10):
        skeleton[i, size-1-i] = 1.0
    
    return skeleton


def create_t_skeleton(size=64):
    """创建T形骨架"""
    skeleton = np.zeros((size, size), dtype=np.float32)
    
    # 水平线（顶部）
    skeleton[15, 10:size-10] = 1.0
    # 垂直线（中间向下）
    skeleton[15:size-10, size//2] = 1.0
    
    return skeleton


def create_tangent_map(skeleton, size=64):
    """为骨架创建简单的切线图"""
    tangent = np.zeros((2, size, size), dtype=np.float32)
    
    # 找到骨架像素
    coords = np.argwhere(skeleton > 0.5)
    
    for i, (r, c) in enumerate(coords):
        # 找邻居来估计方向
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and skeleton[nr, nc] > 0.5:
                    neighbors.append((nr - r, nc - c))
        
        if len(neighbors) >= 1:
            # 使用第一个邻居方向
            dr, dc = neighbors[0]
            theta = np.arctan2(dr, dc)
            tangent[0, r, c] = np.cos(2 * theta)
            tangent[1, r, c] = np.sin(2 * theta)
    
    return tangent


def test_shape(name, skeleton, cross_junctions=True):
    """测试一个形状"""
    print(f"\n{'='*50}")
    print(f"Testing: {name} (cross_junctions={cross_junctions})")
    print(f"{'='*50}")
    
    size = skeleton.shape[0]
    tangent = create_tangent_map(skeleton, size)
    
    pred_maps = {
        'skeleton': skeleton,
        'junction': np.zeros_like(skeleton),  # 不使用junction map
        'tangent': tangent,
        'width': np.ones_like(skeleton) * 2.0,
        'offset': np.zeros((2, size, size)),
    }
    
    reconstructor = SimpleGraphReconstructor(config={
        'skeleton_threshold': 0.5,
        'junction_threshold': 0.5,
        'min_stroke_length': 3,
        'use_thinning': False,  # 骨架已经是1像素宽
        'cross_junctions': cross_junctions,
    })
    
    strokes = reconstructor.process(pred_maps)
    bezier_curves = fit_bezier_curves(strokes, tolerance=2.0)
    
    print(f"  Number of strokes: {len(strokes)}")
    print(f"  Number of curves: {len(bezier_curves)}")
    for i, stroke in enumerate(strokes):
        print(f"  Stroke {i}: {len(stroke['points'])} points")
    
    return strokes, bezier_curves, pred_maps


def visualize_results(name, skeleton, strokes, curves, save_path):
    """可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 原始骨架
    axes[0].imshow(skeleton, cmap='gray')
    axes[0].set_title(f'{name} - Skeleton')
    axes[0].axis('off')
    
    # 追踪的笔画
    axes[1].imshow(skeleton, cmap='gray', alpha=0.3)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(strokes), 1)))
    for i, stroke in enumerate(strokes):
        pts = np.array(stroke['points'])
        axes[1].plot(pts[:, 0], pts[:, 1], '-', color=colors[i], linewidth=2)
        axes[1].plot(pts[0, 0], pts[0, 1], 'o', color=colors[i], markersize=8)
        axes[1].plot(pts[-1, 0], pts[-1, 1], 's', color=colors[i], markersize=8)
    axes[1].set_title(f'Traced Strokes ({len(strokes)})')
    axes[1].set_xlim(0, skeleton.shape[1])
    axes[1].set_ylim(skeleton.shape[0], 0)
    axes[1].axis('off')
    
    # 贝塞尔曲线
    axes[2].imshow(skeleton, cmap='gray', alpha=0.3)
    curve_colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves), 1)))
    for i, curve in enumerate(curves):
        p0, p1, p2, p3 = [np.array(p) for p in curve['points']]
        t = np.linspace(0, 1, 50)
        pts = np.outer((1-t)**3, p0) + np.outer(3*(1-t)**2*t, p1) + \
              np.outer(3*(1-t)*t**2, p2) + np.outer(t**3, p3)
        axes[2].plot(pts[:, 0], pts[:, 1], '-', color=curve_colors[i % len(curve_colors)], linewidth=2)
    axes[2].set_title(f'Bezier Curves ({len(curves)})')
    axes[2].set_xlim(0, skeleton.shape[1])
    axes[2].set_ylim(skeleton.shape[0], 0)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    import os
    os.makedirs('results_crossing', exist_ok=True)
    
    # 测试十字形
    cross = create_cross_skeleton()
    
    # 不穿越交叉点
    strokes1, curves1, _ = test_shape("Cross (no crossing)", cross, cross_junctions=False)
    visualize_results("Cross_NoCrossing", cross, strokes1, curves1, 
                      'results_crossing/cross_no_crossing.png')
    
    # 穿越交叉点
    strokes2, curves2, _ = test_shape("Cross (with crossing)", cross, cross_junctions=True)
    visualize_results("Cross_WithCrossing", cross, strokes2, curves2, 
                      'results_crossing/cross_with_crossing.png')
    
    # 测试X形
    x_shape = create_x_skeleton()
    strokes3, curves3, _ = test_shape("X-shape (no crossing)", x_shape, cross_junctions=False)
    visualize_results("X_NoCrossing", x_shape, strokes3, curves3,
                      'results_crossing/x_no_crossing.png')
    
    strokes4, curves4, _ = test_shape("X-shape (with crossing)", x_shape, cross_junctions=True)
    visualize_results("X_WithCrossing", x_shape, strokes4, curves4,
                      'results_crossing/x_with_crossing.png')
    
    # 测试T形
    t_shape = create_t_skeleton()
    strokes5, curves5, _ = test_shape("T-shape (no crossing)", t_shape, cross_junctions=False)
    visualize_results("T_NoCrossing", t_shape, strokes5, curves5,
                      'results_crossing/t_no_crossing.png')
    
    strokes6, curves6, _ = test_shape("T-shape (with crossing)", t_shape, cross_junctions=True)
    visualize_results("T_WithCrossing", t_shape, strokes6, curves6,
                      'results_crossing/t_with_crossing.png')
    
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    print(f"Cross without crossing: {len(strokes1)} strokes")
    print(f"Cross with crossing: {len(strokes2)} strokes")
    print(f"X without crossing: {len(strokes3)} strokes")
    print(f"X with crossing: {len(strokes4)} strokes")
    print(f"T without crossing: {len(strokes5)} strokes")
    print(f"T with crossing: {len(strokes6)} strokes")


if __name__ == '__main__':
    main()
