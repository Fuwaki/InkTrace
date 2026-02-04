#!/usr/bin/env python3
"""
InkTrace Graph Reconstruction Algorithm (Python Prototype v3)

改进版本：
1. 使用形态学骨架细化
2. 更鲁棒的端点/交叉点检测
3. 正确的坐标系统处理
4. 利用 offset map 进行亚像素修正
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import label
from skimage.morphology import skeletonize, thin


class SimpleGraphReconstructor:
    """
    简化的图重建算法
    
    坐标约定: 所有内部处理使用 (row, col) 即 (y, x) 格式
    输出时转换为 (x, y) 格式用于可视化
    """

    def __init__(self, config=None):
        """
        Args:
            config: dict of parameters
                - skeleton_threshold: float, default 0.3
                - junction_threshold: float, default 0.5
                - min_stroke_length: int, default 5
                - use_thinning: bool, default True
                - cross_junctions: bool, default True - 是否穿越交叉点
        """
        self.config = config or {}
        self.skeleton_threshold = self.config.get('skeleton_threshold', 0.3)
        self.junction_threshold = self.config.get('junction_threshold', 0.5)
        self.min_stroke_length = self.config.get('min_stroke_length', 5)
        self.use_thinning = self.config.get('use_thinning', True)
        self.cross_junctions = self.config.get('cross_junctions', True)  # 默认穿越交叉点

    def process(self, pred_maps):
        """
        Main pipeline

        Args:
            pred_maps: dict of prediction arrays
                - skeleton: [H, W]
                - junction: [H, W] (optional)
                - tangent: [2, H, W] (optional)
                - offset: [2, H, W] (optional)
                - width: [H, W] (optional)

        Returns:
            strokes: list of dict
                - points: [(x, y), ...] - ordered pixel chain (x, y format for viz)
                - width: float
                - start_junction: bool
                - end_junction: bool
        """
        # Convert to numpy if tensors
        pred = self._to_numpy(pred_maps)

        # Phase 1: Binarize and thin skeleton
        skeleton_binary = pred['skeleton'] > self.skeleton_threshold
        
        # Apply morphological thinning to get 1-pixel wide skeleton
        if self.use_thinning:
            skeleton_thin = thin(skeleton_binary)
        else:
            skeleton_thin = skeleton_binary
            
        print(f"  Skeleton pixels before/after thinning: {skeleton_binary.sum()} -> {skeleton_thin.sum()}")

        # Phase 2: Find endpoints and junctions from thinned skeleton
        endpoints, junctions = self._find_endpoints_and_junctions(
            skeleton_thin, pred.get('junction')
        )

        print(f"  Found {len(endpoints)} endpoints, {len(junctions)} junctions")

        # Phase 3: Trace complete strokes
        strokes = self._trace_strokes(skeleton_thin, endpoints, junctions, pred)

        print(f"  Traced {len(strokes)} strokes")

        return strokes

    def _to_numpy(self, pred_maps):
        """Convert tensors to numpy arrays"""
        pred = {}
        for key, value in pred_maps.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            if isinstance(value, np.ndarray):
                arr = value
            else:  # torch.Tensor
                arr = value.numpy()

            # Remove channel dimension if present
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]

            pred[key] = arr
        return pred

    def _find_endpoints_and_junctions(self, skeleton, junction_map=None):
        """
        找到skeleton的端点和交叉点
        
        使用 (row, col) 即 (y, x) 坐标格式

        Returns:
            endpoints: list of (row, col) tuples
            junctions: list of (row, col) tuples
        """
        H, W = skeleton.shape
        endpoint_list = []
        junction_list = []
        
        # 构建邻居计数矩阵 (更高效)
        # 使用卷积计算每个像素的8邻居数量
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.float32)
        neighbor_count = ndimage.convolve(skeleton.astype(np.float32), kernel, mode='constant')
        
        # 只在skeleton像素上计算
        skeleton_coords = np.argwhere(skeleton)  # (N, 2) -> (row, col)
        
        for coord in skeleton_coords:
            row, col = coord
            n_neighbors = int(neighbor_count[row, col])
            
            if n_neighbors == 1:
                # Endpoint: 只有1个邻居
                endpoint_list.append((row, col))
            elif n_neighbors >= 3:
                # Junction: 3个或更多邻居
                # 但需要进一步验证，避免锯齿导致的假阳性
                # 检查是否是真正的分叉点
                if self._is_true_junction(skeleton, row, col):
                    junction_list.append((row, col))
        
        # 如果提供了junction_map，用它来辅助验证
        if junction_map is not None:
            # 只保留在junction_map中也有高响应的点
            validated_junctions = []
            for junc in junction_list:
                row, col = junc
                # 检查3x3邻域内的最大junction响应
                r_min, r_max = max(0, row-1), min(H, row+2)
                c_min, c_max = max(0, col-1), min(W, col+2)
                local_max = junction_map[r_min:r_max, c_min:c_max].max()
                if local_max > self.junction_threshold:
                    validated_junctions.append(junc)
            
            # 如果验证后junction太少，使用原始检测结果
            if len(validated_junctions) > 0:
                junction_list = validated_junctions

        return endpoint_list, junction_list
    
    def _is_true_junction(self, skeleton, row, col):
        """
        验证是否是真正的分叉点（而不是锯齿）
        
        使用连通性分析：真正的分叉点周围应该有3个或更多独立的连通区域
        """
        H, W = skeleton.shape
        
        # 提取3x3邻域
        r_min, r_max = max(0, row-1), min(H, row+2)
        c_min, c_max = max(0, col-1), min(W, col+2)
        
        # 创建邻域副本，移除中心点
        neighborhood = skeleton[r_min:r_max, c_min:c_max].copy()
        local_row = row - r_min
        local_col = col - c_min
        neighborhood[local_row, local_col] = False
        
        # 计算连通分量数量
        labeled, num_features = label(neighborhood)
        
        # 真正的分叉点应该有3个或更多独立的分支
        return num_features >= 3

    def _trace_strokes(self, skeleton, endpoints, junctions, pred):
        """
        追踪所有完整的笔画
        
        坐标使用 (row, col) 格式

        Strategy:
        1. Start from each endpoint
        2. Walk along skeleton until hitting another endpoint
        3. If cross_junctions=True, allow passing through junction pixels
        """
        # 所有skeleton像素集合 (row, col) 格式
        all_pixels = set(map(tuple, np.argwhere(skeleton)))
        
        # 分别追踪已使用的端点和已完全追踪的像素
        used_endpoints = set()
        visited_pixels = set()  # 已追踪的非交叉点像素
        junction_set = set(junctions)
        endpoint_set = set(endpoints)
        
        strokes = []

        # Combine all special points
        all_nodes = set(endpoints + junctions)

        # Start from each endpoint
        for start_point in endpoints:
            if start_point in used_endpoints:
                continue

            # Trace from this endpoint
            stroke_pixels = self._walk_from_point_v2(
                start_point, skeleton, endpoint_set, junction_set, 
                visited_pixels, used_endpoints, pred
            )

            if len(stroke_pixels) >= self.min_stroke_length:
                # 转换为 (x, y) 格式用于输出
                points_xy = [(col, row) for (row, col) in stroke_pixels]
                width = self._estimate_width(stroke_pixels, pred.get('width'))
                strokes.append({
                    'points': points_xy,
                    'points_rc': stroke_pixels,
                    'width': width,
                    'start_junction': start_point in junction_set,
                    'end_junction': stroke_pixels[-1] in all_nodes
                })

        # Handle isolated loops or unvisited segments
        remaining = all_pixels - visited_pixels - junction_set
        while remaining:
            start = remaining.pop()
            stroke_pixels = self._walk_from_point_v2(
                start, skeleton, endpoint_set, junction_set,
                visited_pixels, used_endpoints, pred, max_steps=1000
            )

            if len(stroke_pixels) >= self.min_stroke_length:
                points_xy = [(col, row) for (row, col) in stroke_pixels]
                width = self._estimate_width(stroke_pixels, pred.get('width'))
                strokes.append({
                    'points': points_xy,
                    'points_rc': stroke_pixels,
                    'width': width,
                    'start_junction': False,
                    'end_junction': False
                })

            remaining = all_pixels - visited_pixels - junction_set

        return strokes

    def _walk_from_point_v2(self, start_point, skeleton, endpoints, junctions, 
                            visited_pixels, used_endpoints, pred, max_steps=1000):
        """
        改进版的追踪函数：
        - visited_pixels: 已完全追踪的非交叉点像素（不能再访问）
        - junctions: 交叉点像素（可以穿越，不标记为已访问）
        - endpoints: 端点像素（作为停止条件）
        
        坐标使用 (row, col) 格式
        """
        all_pixels = set(map(tuple, np.argwhere(skeleton)))

        current = start_point
        pixels = [current]
        
        # 标记起点
        if current in endpoints:
            used_endpoints.add(current)
        if current not in junctions:
            visited_pixels.add(current)

        prev_direction = None
        local_visited = {current}  # 本次追踪中访问过的点，防止循环

        for _ in range(max_steps):
            # 获取所有邻居
            neighbors = self._get_8neighbors(current, all_pixels)
            
            # 筛选可访问的邻居
            valid_neighbors = []
            for n in neighbors:
                # 跳过本次追踪已访问的点（防止循环）
                if n in local_visited:
                    continue
                    
                # 如果是交叉点，允许穿越（即使全局已访问）
                if n in junctions:
                    if self.cross_junctions:
                        valid_neighbors.append(n)
                    continue
                    
                # 非交叉点：检查是否全局已访问
                if n not in visited_pixels:
                    valid_neighbors.append(n)

            if len(valid_neighbors) == 0:
                break

            if len(valid_neighbors) == 1:
                next_pixel = valid_neighbors[0]
            else:
                # Multiple choices - use tangent to guide
                next_pixel = self._select_by_tangent_or_direction(
                    current, valid_neighbors, prev_direction, pred.get('tangent')
                )

            pixels.append(next_pixel)
            local_visited.add(next_pixel)
            
            # 更新全局状态
            if next_pixel not in junctions:
                visited_pixels.add(next_pixel)
            if next_pixel in endpoints:
                used_endpoints.add(next_pixel)

            # Update direction
            prev_direction = (next_pixel[0] - current[0], next_pixel[1] - current[1])
            current = next_pixel

            # 只在端点处停止（起点除外）
            if current in endpoints and current != start_point:
                break

        return pixels
    
    def _is_endpoint(self, point, skeleton):
        """判断一个点是否是端点（只有1个邻居）"""
        row, col = point
        H, W = skeleton.shape
        neighbors = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < H and 0 <= nc < W and skeleton[nr, nc]:
                    neighbors += 1
        return neighbors == 1

    def _get_8neighbors(self, pixel, pixel_set):
        """Get 8-connected neighbors in (row, col) format"""
        row, col = pixel
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (nr, nc) in pixel_set:
                    neighbors.append((nr, nc))
        return neighbors

    def _select_by_tangent_or_direction(self, current, candidates, prev_direction, tangent_map):
        """
        选择下一个像素，优先考虑切线方向或继续直行
        
        current和candidates使用 (row, col) 格式
        tangent_map使用 [2, H, W] 格式，存储 (cos2θ, sin2θ)
        """
        if prev_direction is None:
            return candidates[0]

        # 如果有tangent map，优先使用它
        if tangent_map is not None:
            row, col = current
            # tangent是 (cos2θ, sin2θ)，需要还原为方向
            cos2t = tangent_map[0, row, col]
            sin2t = tangent_map[1, row, col]
            # 还原角度 (有180度歧义，用prev_direction解决)
            theta = np.arctan2(sin2t, cos2t) / 2
            
            # 两个可能的方向
            dir1 = np.array([np.sin(theta), np.cos(theta)])  # (row_dir, col_dir)
            dir2 = -dir1
            
            # 选择与prev_direction更一致的那个
            prev_vec = np.array(prev_direction)
            prev_vec = prev_vec / (np.linalg.norm(prev_vec) + 1e-6)
            
            if np.dot(prev_vec, dir1) > np.dot(prev_vec, dir2):
                tangent_dir = dir1
            else:
                tangent_dir = dir2
            
            # 选择与tangent方向最一致的候选点
            best_candidate = candidates[0]
            best_score = -2
            
            for cand in candidates:
                cand_vec = np.array([cand[0] - current[0], cand[1] - current[1]])
                cand_vec = cand_vec / (np.linalg.norm(cand_vec) + 1e-6)
                score = np.dot(tangent_dir, cand_vec)
                
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            return best_candidate

        # Fallback: 继续沿着之前的方向走
        prev_vec = np.array(prev_direction)
        prev_vec = prev_vec / (np.linalg.norm(prev_vec) + 1e-6)

        best_candidate = candidates[0]
        best_score = -1

        for cand in candidates:
            cand_vec = np.array([cand[0] - current[0], cand[1] - current[1]])
            cand_vec = cand_vec / (np.linalg.norm(cand_vec) + 1e-6)

            # Prefer forward direction (dot product close to 1)
            score = np.dot(prev_vec, cand_vec)

            if score > best_score:
                best_score = score
                best_candidate = cand

        return best_candidate

    def _estimate_width(self, points, width_map):
        """估计路径的平均宽度
        
        points使用 (row, col) 格式
        """
        if width_map is None:
            return 2.0

        widths = []
        for p in points:
            row, col = int(p[0]), int(p[1])
            if 0 <= row < width_map.shape[0] and 0 <= col < width_map.shape[1]:
                widths.append(width_map[row, col])

        return np.mean(widths) if widths else 2.0


def fit_bezier_curves(strokes, tolerance=1.0):
    """
    将追踪到的笔画拟合成贝塞尔曲线

    Args:
        strokes: list of dict with 'points' key
        tolerance: fitting error tolerance

    Returns:
        curves: list of dict with 'points' (4 control points) and 'width'
    """
    all_curves = []

    for stroke in strokes:
        points = stroke['points']

        if len(points) < 4:
            # Too short, just return line
            p0, p3 = points[0], points[-1]
            p1 = ((2*p0[0] + p3[0]) / 3, (2*p0[1] + p3[1]) / 3)
            p2 = ((p0[0] + 2*p3[0]) / 3, (p0[1] + 2*p3[1]) / 3)

            all_curves.append({
                'points': [p0, p1, p2, p3],
                'width': stroke['width']
            })
        else:
            # Fit one or multiple curves
            curves = fit_bezier_chain(points, tolerance)
            for curve in curves:
                all_curves.append({
                    'points': curve,
                    'width': stroke['width']
                })

    return all_curves


def fit_bezier_chain(points, tolerance, max_depth=5, depth=0):
    """
    递归拟合贝塞尔曲线链
    
    Args:
        points: list of (x, y) tuples
        tolerance: 最大允许误差
        max_depth: 最大递归深度，避免过度分割
        depth: 当前递归深度
    """
    points_arr = np.array(points)
    
    if len(points) < 4:
        return [fit_single_bezier(points_arr)]

    # Try single curve
    control_points = fit_single_bezier(points_arr)
    errors = compute_fitting_errors(points_arr, control_points)
    max_error = np.max(errors)

    # 停止条件：误差足够小，或者点太少，或者递归太深
    if max_error <= tolerance or len(points) < 8 or depth >= max_depth:
        return [control_points]

    # Split at worst point
    split_idx = np.argmax(errors)
    
    # 确保分割点不在太靠近端点的位置
    min_segment = max(4, len(points) // 4)
    if split_idx < min_segment:
        split_idx = min_segment
    elif split_idx > len(points) - min_segment:
        split_idx = len(points) - min_segment
    
    # 如果分割后段太短，就不分割了
    if split_idx < 4 or len(points) - split_idx < 4:
        return [control_points]

    # Recursively fit
    left_curves = fit_bezier_chain(points[:split_idx+1], tolerance, max_depth, depth+1)
    right_curves = fit_bezier_chain(points[split_idx:], tolerance, max_depth, depth+1)

    return left_curves + right_curves


def fit_single_bezier(points):
    """
    拟合单条三次贝塞尔曲线 (使用改进的最小二乘法)
    
    三次贝塞尔曲线: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    
    Args:
        points: numpy array of shape (n, 2)
        
    Returns:
        [P₀, P₁, P₂, P₃] - 四个控制点
    """
    n = len(points)
    p0 = points[0]
    p3 = points[-1]

    if n <= 2:
        # 太短，返回直线
        p1 = ((2*p0[0] + p3[0]) / 3, (2*p0[1] + p3[1]) / 3)
        p2 = ((p0[0] + 2*p3[0]) / 3, (p0[1] + 2*p3[1]) / 3)
        return [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]

    # Chord length parameterization
    chord_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    total_length = np.sum(chord_lengths)
    
    if total_length < 1e-6:
        # 所有点重合
        p1 = tuple(p0)
        p2 = tuple(p3)
        return [tuple(p0), p1, p2, tuple(p3)]
    
    cumulative = np.cumsum(chord_lengths)
    t = np.zeros(n)
    t[1:] = cumulative / total_length

    # 构建线性系统来求解 P1 和 P2
    # B(t) - (1-t)³P₀ - t³P₃ = 3(1-t)²t·P₁ + 3(1-t)t²·P₂
    
    # 基函数
    B1 = 3 * (1 - t) ** 2 * t  # P1 的系数
    B2 = 3 * (1 - t) * t ** 2  # P2 的系数
    
    # 构建 A 矩阵 [B1, B2]
    A = np.column_stack([B1, B2])
    
    # 目标向量: 原始点 - 端点贡献
    B0 = (1 - t) ** 3  # P0 的系数
    B3 = t ** 3        # P3 的系数
    
    b = points - np.outer(B0, p0) - np.outer(B3, p3)

    try:
        # 分别求解 x 和 y 坐标
        sol_x, _, _, _ = np.linalg.lstsq(A, b[:, 0], rcond=None)
        sol_y, _, _, _ = np.linalg.lstsq(A, b[:, 1], rcond=None)

        p1 = (float(sol_x[0]), float(sol_y[0]))
        p2 = (float(sol_x[1]), float(sol_y[1]))
        
        # 检查控制点是否合理 (不要离端点太远)
        max_dist = np.linalg.norm(p3 - p0) * 2
        if np.linalg.norm(np.array(p1) - p0) > max_dist or \
           np.linalg.norm(np.array(p2) - p3) > max_dist:
            # 控制点异常，退化为直线
            p1 = ((2*p0[0] + p3[0]) / 3, (2*p0[1] + p3[1]) / 3)
            p2 = ((p0[0] + 2*p3[0]) / 3, (p0[1] + 2*p3[1]) / 3)
            
    except Exception:
        # 拟合失败，返回直线
        p1 = ((2*p0[0] + p3[0]) / 3, (2*p0[1] + p3[1]) / 3)
        p2 = ((p0[0] + 2*p3[0]) / 3, (p0[1] + 2*p3[1]) / 3)

    return [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]


def compute_fitting_errors(points, control_points):
    """计算拟合误差"""
    p0, p1, p2, p3 = [np.array(p) for p in control_points]

    # Sample curve
    t = np.linspace(0, 1, len(points) * 2)
    curve = np.outer((1-t)**3, p0) + \
            np.outer(3*(1-t)**2*t, p1) + \
            np.outer(3*(1-t)*t**2, p2) + \
            np.outer(t**3, p3)

    # Find min distance for each point
    errors = []
    for p in points:
        dists = np.linalg.norm(curve - p, axis=1)
        errors.append(dists.min())

    return np.array(errors)
