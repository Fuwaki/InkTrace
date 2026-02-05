#!/usr/bin/env python3
"""
InkTrace Graph Reconstruction Algorithm (Simplified)

基于 Skeleton Topology 的重建：
1. 使用形态学骨架细化
2. 基于邻居数的特异点检测 (端点=1, 连接点/交叉点>=3)
3. 路径追踪与合并
4. 贝塞尔曲线拟合

V1.1 更新：
- 添加 Stroke 级别的智能合并，解决模型预测多节点分割问题
- 基于切线连续性判断是否应该合并相邻 segment
- 区分真正的交叉点和虚假的中间节点
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import label
from skimage.morphology import skeletonize, thin
from collections import defaultdict


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
        self.skeleton_threshold = self.config.get("skeleton_threshold", 0.3)
        self.junction_threshold = self.config.get("junction_threshold", 0.5)
        self.min_stroke_length = self.config.get("min_stroke_length", 5)
        self.use_thinning = self.config.get("use_thinning", True)
        self.cross_junctions = self.config.get(
            "cross_junctions", True
        )  # 默认穿越交叉点
        # 新增：合并参数
        self.merge_strokes = self.config.get(
            "merge_strokes", True
        )  # 是否合并被分割的笔画
        self.merge_angle_threshold = self.config.get(
            "merge_angle_threshold", 45.0
        )  # 合并角度阈值(度)
        self.min_branch_neighbors = self.config.get(
            "min_branch_neighbors", 3
        )  # 认定为真正分支点的最小邻居数

    def process(self, pred_maps):
        """
        Main pipeline

        Args:
            pred_maps: dict of prediction arrays
                - skeleton: [H, W]
                - junction: [H, W] (Marks both endpoints and internal nodes)
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
        skeleton_binary = pred["skeleton"] > self.skeleton_threshold

        # Apply morphological thinning to get 1-pixel wide skeleton
        if self.use_thinning:
            skeleton_thin = thin(skeleton_binary)
        else:
            skeleton_thin = skeleton_binary

        # Phase 2: Find special points (endpoints and internal nodes)
        # We classify them based on topology primarily, validated by junction map
        endpoints, internal_nodes = self._find_topology_points(
            skeleton_thin, pred.get("junction")
        )

        # Phase 3: Trace complete strokes
        strokes = self._trace_strokes(skeleton_thin, endpoints, internal_nodes, pred)

        # Phase 4: Merge fragmented strokes (智能合并被分割的笔画)
        if self.merge_strokes and len(strokes) > 1:
            strokes = self._merge_fragmented_strokes(strokes, internal_nodes, pred)

        return strokes

    def _find_topology_points(self, skeleton, junction_map=None):
        """
        根据骨架拓扑结构寻找特殊点

        Endpoint: 1 neighbor
        Internal Node (Connection/Junction): >= 3 neighbors OR >= 2 neighbors & high junction score

        改进：区分"真正的交叉点"和"虚假的中间节点"
        - 真正的交叉点：拓扑上有 >= 3 个邻居
        - 虚假的中间节点：只有 2 个邻居但 junction_map 分数高（可能是模型误判）

        Returns:
            endpoints: list of (row, col)
            internal_nodes: list of (row, col) (includes connections and crossings)
        """
        H, W = skeleton.shape
        endpoint_list = []
        internal_list = []

        # 新增：区分真正的分支点和可能的虚假节点
        true_branch_list = []  # 拓扑上确定是分支的点
        possible_junction_list = []  # 可能是虚假的点（只有2邻居但junction分数高）

        # 构建邻居计数矩阵
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
        neighbor_count = ndimage.convolve(
            skeleton.astype(np.float32), kernel, mode="constant"
        )

        # 只在skeleton像素上计算
        skeleton_coords = np.argwhere(skeleton)

        for coord in skeleton_coords:
            row, col = coord
            n_neighbors = int(neighbor_count[row, col])

            if n_neighbors == 1:
                endpoint_list.append((row, col))
            elif n_neighbors >= self.min_branch_neighbors:
                # Topologically a junction/connection (真正的分支点)
                true_branch_list.append((row, col))
            elif n_neighbors == 2 and junction_map is not None:
                # Check if it's a "virtual" breakdown point (sharp turn or segmentation point)
                # 这些点可能是模型误判，需要在后续合并时处理
                if junction_map[row, col] > self.junction_threshold:
                    possible_junction_list.append((row, col))

        # 合并所有 internal nodes
        # 但我们会在元数据中标记哪些是真正的分支点
        internal_list = true_branch_list + possible_junction_list

        # Junction map validation for endpoints (optional but good for noise removal)
        if junction_map is not None:
            validated_endpoints = []
            for ep in endpoint_list:
                row, col = ep
                # Search local max in junction map
                r_min, r_max = max(0, row - 2), min(H, row + 3)
                c_min, c_max = max(0, col - 2), min(W, col + 3)
                if (
                    junction_map[r_min:r_max, c_min:c_max].max()
                    > self.junction_threshold
                ):
                    validated_endpoints.append(ep)

            # If we lose too many endpoints, fallback to raw topological endpoints
            if len(validated_endpoints) > 0:
                endpoint_list = validated_endpoints

        # 存储真正的分支点集合，供后续合并使用
        self._true_branch_points = set(true_branch_list)
        self._possible_junction_points = set(possible_junction_list)

        return endpoint_list, internal_list

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

    def _trace_strokes(self, skeleton, endpoints, internal_nodes, pred):
        """
        追踪所有完整的笔画
        Strategy:
        1. Break skeleton into segments at internal nodes
        2. Trace each segment from node to node (or node to endpoint)
        """
        # 所有skeleton像素集合
        all_pixels = set(map(tuple, np.argwhere(skeleton)))

        # Nodes where tracing stops
        stop_nodes = set(endpoints + internal_nodes)

        visited_segments = (
            set()
        )  # Store as sets of frozensets of points to avoid duplicates
        visited_pixels = set()

        strokes = []

        # Seeds for tracing: endpoints and internal nodes
        # We need to trace from every neighbor of every node
        seeds = []

        # 1. Add all endpoints as seeds
        seeds.extend(endpoints)

        # 2. Add all internal nodes as seeds
        seeds.extend(internal_nodes)

        for start_node in seeds:
            # Get all neighbors of this node that are skeleton pixels
            neighbors = self._get_8neighbors(start_node, all_pixels)

            for neighbor in neighbors:
                # Unique segment ID: tuple of sorted (start_node, neighbor)
                # This only works for the first step. For full segment deduplication we check visited pixels.

                if neighbor in visited_pixels:
                    # Heuristic: if neighbor is already fully processed, skip.
                    # Note: internal nodes are never "visited" in the sense of being consumed.
                    if neighbor not in stop_nodes:
                        continue

                # Trace this segment
                segment_pixels = self._trace_segment(
                    start_node, neighbor, skeleton, stop_nodes, pred
                )

                if segment_pixels and len(segment_pixels) >= self.min_stroke_length:
                    # Check if we've seen this segment (reverse check)
                    # We use midpoint to roughly identify the segment
                    mid_idx = len(segment_pixels) // 2
                    mid_point = segment_pixels[mid_idx]

                    if mid_point in visited_pixels and mid_point not in stop_nodes:
                        continue

                    # Mark pixels as visited
                    for p in segment_pixels:
                        if p not in stop_nodes:
                            visited_pixels.add(p)

                    # Output stroke
                    points_xy = [(col, row) for (row, col) in segment_pixels]
                    width = self._estimate_width(segment_pixels, pred.get("width"))

                    strokes.append(
                        {
                            "points": points_xy,
                            "points_rc": segment_pixels,
                            "width": width,
                        }
                    )

        return strokes

    def _trace_segment(self, start_node, second_pixel, skeleton, stop_nodes, pred):
        """
        Trace a single segment from start_node passing through second_pixel
        Stops when hitting any node in stop_nodes
        """
        if second_pixel in stop_nodes:
            return [start_node, second_pixel]

        current = second_pixel
        pixels = [start_node, current]
        prev_direction = (current[0] - start_node[0], current[1] - start_node[1])

        all_pixels = set(map(tuple, np.argwhere(skeleton)))

        max_steps = 1000
        for _ in range(max_steps):
            # Get neighbors
            neighbors = self._get_8neighbors(current, all_pixels)

            # Filter neighbors: not the one we came from
            valid_candidates = [n for n in neighbors if n != pixels[-2]]

            if not valid_candidates:
                break  # Dead end

            # If any neighbor is a stop node, go there and finish
            stop_candidates = [n for n in valid_candidates if n in stop_nodes]
            if stop_candidates:
                # Pick the best one if multiple (rare)
                next_pixel = stop_candidates[0]
                pixels.append(next_pixel)
                break

            # Regular traversal
            if len(valid_candidates) == 1:
                next_pixel = valid_candidates[0]
            else:
                # Junction logic (should be rare if internal_nodes are correct)
                next_pixel = self._select_by_tangent_or_direction(
                    current, valid_candidates, prev_direction, pred.get("tangent")
                )

            pixels.append(next_pixel)
            prev_direction = (next_pixel[0] - current[0], next_pixel[1] - current[1])
            current = next_pixel

            # Safety check (should be caught by stop_candidates but just in case)
            if current in stop_nodes:
                break

        return pixels

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

    def _select_by_tangent_or_direction(
        self, current, candidates, prev_direction, tangent_map
    ):
        """
        选择下一个像素，优先考虑切线方向
        """
        if prev_direction is None:
            return candidates[0]

        # 如果有tangent map，优先使用它
        if tangent_map is not None:
            row, col = current
            cos2t = tangent_map[0, row, col]
            sin2t = tangent_map[1, row, col]
            theta = np.arctan2(sin2t, cos2t) / 2

            dir1 = np.array([np.sin(theta), np.cos(theta)])
            dir2 = -dir1

            prev_vec = np.array(prev_direction)
            prev_vec = prev_vec / (np.linalg.norm(prev_vec) + 1e-6)

            if np.dot(prev_vec, dir1) > np.dot(prev_vec, dir2):
                tangent_dir = dir1
            else:
                tangent_dir = dir2

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

        # Fallback: Forward direction
        prev_vec = np.array(prev_direction)
        prev_vec = prev_vec / (np.linalg.norm(prev_vec) + 1e-6)

        best_candidate = candidates[0]
        best_score = -1

        for cand in candidates:
            cand_vec = np.array([cand[0] - current[0], cand[1] - current[1]])
            cand_vec = cand_vec / (np.linalg.norm(cand_vec) + 1e-6)
            score = np.dot(prev_vec, cand_vec)
            if score > best_score:
                best_score = score
                best_candidate = cand

        return best_candidate

    def _estimate_width(self, points, width_map):
        """估计路径的平均宽度"""
        if width_map is None:
            return 2.0

        widths = []
        for p in points:
            row, col = int(p[0]), int(p[1])
            if 0 <= row < width_map.shape[0] and 0 <= col < width_map.shape[1]:
                widths.append(width_map[row, col])

        return np.mean(widths) if widths else 2.0

    def _merge_fragmented_strokes(self, strokes, internal_nodes, pred):
        """
        智能合并被分割的笔画段

        策略：
        1. 建立 stroke 端点到 stroke 的索引
        2. 对于每个 internal_node，检查连接到它的所有 stroke
        3. 如果该节点是"虚假的中间节点"（2邻居但被标记为junction），直接合并
        4. 如果是真正的分支点，只在切线方向一致时合并
        5. 使用 Union-Find 结构处理传递性合并

        Args:
            strokes: list of stroke dicts
            internal_nodes: list of (row, col) internal junction points
            pred: prediction maps including tangent

        Returns:
            merged_strokes: list of merged stroke dicts
        """
        if len(strokes) <= 1:
            return strokes

        n = len(strokes)

        # Union-Find 结构
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 建立端点索引: (row, col) -> [(stroke_idx, 'start'|'end'), ...]
        endpoint_index = defaultdict(list)

        for idx, stroke in enumerate(strokes):
            if len(stroke["points"]) < 2:
                continue
            # points 是 (x, y) 格式，转换回 (row, col)
            start_xy = stroke["points"][0]
            end_xy = stroke["points"][-1]
            start_rc = (int(round(start_xy[1])), int(round(start_xy[0])))
            end_rc = (int(round(end_xy[1])), int(round(end_xy[0])))

            endpoint_index[start_rc].append((idx, "start"))
            endpoint_index[end_rc].append((idx, "end"))

        # 获取真正的分支点和可能的虚假节点
        true_branches = getattr(self, "_true_branch_points", set())
        possible_junctions = getattr(self, "_possible_junction_points", set())

        # 对于每个 internal_node，检查连接的 strokes
        for node in internal_nodes:
            # 查找连接到这个节点的所有 stroke 端点
            connected = []

            # 搜索节点附近的端点（考虑舍入误差）
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    neighbor = (node[0] + dr, node[1] + dc)
                    if neighbor in endpoint_index:
                        connected.extend(endpoint_index[neighbor])

            if len(connected) < 2:
                continue

            # 判断这个节点是真正的分支点还是虚假的中间节点
            is_true_branch = node in true_branches
            is_possible_junction = node in possible_junctions

            # 如果是虚假的中间节点（只有2个邻居但被标记为junction），直接合并
            if is_possible_junction and len(connected) == 2:
                idx_i, type_i = connected[0]
                idx_j, type_j = connected[1]
                if find(idx_i) != find(idx_j):
                    union(idx_i, idx_j)
                continue

            # 计算每个连接的切线方向
            directions = []
            for stroke_idx, end_type in connected:
                stroke = strokes[stroke_idx]
                pts = stroke["points"]

                if len(pts) < 2:
                    continue

                # 计算端点处的切线方向（指向笔画内部）
                if end_type == "start":
                    # 从 start 指向内部
                    p1 = np.array(pts[0])
                    p2 = np.array(pts[min(3, len(pts) - 1)])  # 取几个点的平均方向
                    direction = p2 - p1
                else:
                    # 从 end 指向内部（反向）
                    p1 = np.array(pts[-1])
                    p2 = np.array(pts[max(-4, -len(pts))])
                    direction = p2 - p1

                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm
                    directions.append((stroke_idx, end_type, direction))

            # 尝试配对：找切线方向一致的 stroke 对
            # 对于真正的分支点，使用更严格的角度阈值
            angle_threshold = self.merge_angle_threshold
            if is_true_branch:
                angle_threshold = min(angle_threshold, 30.0)  # 对真正的分支点更严格

            for i in range(len(directions)):
                for j in range(i + 1, len(directions)):
                    idx_i, type_i, dir_i = directions[i]
                    idx_j, type_j, dir_j = directions[j]

                    if find(idx_i) == find(idx_j):
                        continue  # 已经在同一组

                    # 检查是否可以合并：
                    # 计算方向一致性
                    if (type_i == "end" and type_j == "start") or (
                        type_i == "start" and type_j == "end"
                    ):
                        # 头尾连接：方向应该相似
                        angle = np.arccos(np.clip(np.dot(dir_i, dir_j), -1, 1))
                    else:
                        # 同向连接：方向应该相反
                        angle = np.arccos(np.clip(-np.dot(dir_i, dir_j), -1, 1))

                    angle_deg = np.degrees(angle)

                    if angle_deg < angle_threshold:
                        # 可以合并
                        union(idx_i, idx_j)

        # 根据 Union-Find 结果合并 strokes
        groups = defaultdict(list)
        for idx in range(n):
            groups[find(idx)].append(idx)

        merged_strokes = []

        for group_root, indices in groups.items():
            if len(indices) == 1:
                # 单独的 stroke，直接保留
                merged_strokes.append(strokes[indices[0]])
            else:
                # 合并多个 strokes
                merged = self._merge_stroke_group(strokes, indices, endpoint_index)
                merged_strokes.append(merged)

        return merged_strokes

    def _merge_stroke_group(self, strokes, indices, endpoint_index):
        """
        合并一组 strokes 成为一个连续的笔画

        使用贪心算法构建路径：
        1. 选择一个端点作为起点
        2. 依次连接相邻的 stroke segment
        """
        if len(indices) == 1:
            return strokes[indices[0]]

        # 收集所有 segment 的端点信息
        segments = []
        for idx in indices:
            stroke = strokes[idx]
            pts = stroke["points"]
            if len(pts) < 2:
                continue
            segments.append(
                {
                    "idx": idx,
                    "points": pts,
                    "start": (
                        int(round(pts[0][1])),
                        int(round(pts[0][0])),
                    ),  # (row, col)
                    "end": (int(round(pts[-1][1])), int(round(pts[-1][0]))),
                    "width": stroke["width"],
                    "used": False,
                    "reversed": False,
                }
            )

        if not segments:
            return strokes[indices[0]]

        # 建立端点邻接关系
        def points_close(p1, p2, threshold=2):
            return abs(p1[0] - p2[0]) <= threshold and abs(p1[1] - p2[1]) <= threshold

        # 找到一个"真正的端点"作为起始（只连接一个其他 segment 的端点）
        endpoint_counts = defaultdict(int)
        for seg in segments:
            endpoint_counts[seg["start"]] += 1
            endpoint_counts[seg["end"]] += 1

        # 找连接数最少的端点开始
        start_seg = None
        start_from_end = False

        for seg in segments:
            # 检查 start 和 end 的连接数
            start_conn = sum(
                1
                for s in segments
                if s != seg
                and (
                    points_close(s["start"], seg["start"])
                    or points_close(s["end"], seg["start"])
                )
            )
            end_conn = sum(
                1
                for s in segments
                if s != seg
                and (
                    points_close(s["start"], seg["end"])
                    or points_close(s["end"], seg["end"])
                )
            )

            if start_conn == 0:
                start_seg = seg
                start_from_end = False
                break
            elif end_conn == 0:
                start_seg = seg
                start_from_end = True
                break

        if start_seg is None:
            start_seg = segments[0]
            start_from_end = False

        # 贪心构建路径
        merged_points = []
        current_seg = start_seg
        current_seg["used"] = True

        if start_from_end:
            merged_points = list(reversed(current_seg["points"]))
            current_end = current_seg["start"]
        else:
            merged_points = list(current_seg["points"])
            current_end = current_seg["end"]

        total_width = current_seg["width"]
        width_count = 1

        # 继续连接
        while True:
            next_seg = None
            connect_to_start = None

            for seg in segments:
                if seg["used"]:
                    continue

                # 检查是否可以连接
                if points_close(seg["start"], current_end):
                    next_seg = seg
                    connect_to_start = True
                    break
                elif points_close(seg["end"], current_end):
                    next_seg = seg
                    connect_to_start = False
                    break

            if next_seg is None:
                break

            next_seg["used"] = True

            if connect_to_start:
                # 正向连接
                merged_points.extend(next_seg["points"][1:])  # 跳过第一个点，避免重复
                current_end = next_seg["end"]
            else:
                # 反向连接
                merged_points.extend(reversed(list(next_seg["points"])[:-1]))
                current_end = next_seg["start"]

            total_width += next_seg["width"]
            width_count += 1

        return {
            "points": merged_points,
            "width": total_width / width_count,
        }


def fit_bezier_curves(
    strokes, tolerance=1.0, merge_connected=True, connection_tolerance=3.0
):
    """
    将追踪到的笔画拟合成贝塞尔曲线，并根据拓扑重组路径
    """
    all_curves = []

    for stroke_idx, stroke in enumerate(strokes):
        points = stroke["points"]

        if len(points) < 4:
            # Too short, straight line
            p0, p3 = points[0], points[-1]
            p1 = ((2 * p0[0] + p3[0]) / 3, (2 * p0[1] + p3[1]) / 3)
            p2 = ((p0[0] + 2 * p3[0]) / 3, (p0[1] + 2 * p3[1]) / 3)

            all_curves.append(
                {
                    "points": [p0, p1, p2, p3],
                    "width": stroke["width"],
                    "stroke_idx": stroke_idx,
                }
            )
        else:
            # Fitting
            curves = fit_bezier_chain(points, tolerance)
            for curve in curves:
                all_curves.append(
                    {
                        "points": curve,
                        "width": stroke["width"],
                        "stroke_idx": stroke_idx,
                    }
                )

    # Merge connected curves into paths
    if merge_connected and len(all_curves) > 1:
        all_curves = _merge_connected_curves(all_curves, connection_tolerance)

    return all_curves


def _merge_connected_curves(curves, tolerance=3.0):
    """
    Merge curves that are connected (endpoint of one == startpoint of another).
    """
    n = len(curves)
    if n == 0:
        return curves

    # Simple Greedy Merging
    # Note: Complex topology handling is often better done on the vectorized graph structure
    # Here we just chain simple G0 connections

    adjacency = {i: [] for i in range(n)}
    reverse_adj = {i: [] for i in range(n)}

    for i in range(n):
        end_i = np.array(curves[i]["points"][3])
        for j in range(n):
            if i == j:
                continue
            start_j = np.array(curves[j]["points"][0])
            dist = np.linalg.norm(end_i - start_j)
            if dist < tolerance:
                adjacency[i].append(j)
                reverse_adj[j].append(i)

    visited = set()
    path_id = 0

    # Helper to find path start
    def find_start(idx):
        curr = idx
        path_set = {curr}
        while True:
            preds = [p for p in reverse_adj[curr] if p not in path_set]  # Prevent loops
            if not preds:
                break
            curr = preds[0]
            path_set.add(curr)
        return curr

    # Assign path IDs
    for i in range(n):
        if i in visited:
            continue

        start_node = find_start(i)

        # Walk forward
        curr = start_node
        path_order = 0

        while curr is not None:
            if curr in visited:
                break
            visited.add(curr)

            curves[curr]["path_id"] = path_id
            curves[curr]["path_order"] = path_order
            path_order += 1

            succs = [s for s in adjacency[curr] if s not in visited]
            curr = succs[0] if succs else None

        path_id += 1

    return curves


def get_continuous_paths(curves):
    """Group curves by path_id"""
    if not curves:
        return []
    path_dict = {}
    for curve in curves:
        pid = curve.get("path_id", 0)
        if pid not in path_dict:
            path_dict[pid] = []
        path_dict[pid].append(curve)

    paths = []
    for pid in sorted(path_dict.keys()):
        paths.append(sorted(path_dict[pid], key=lambda c: c.get("path_order", 0)))
    return paths


def fit_bezier_chain(points, tolerance, max_depth=5, depth=0):
    """Recursively fit Bezier chain"""
    points_arr = np.array(points)
    if len(points) < 4:
        return [fit_single_bezier(points_arr)]

    control_points = fit_single_bezier(points_arr)
    errors = compute_fitting_errors(points_arr, control_points)
    if np.max(errors) <= tolerance or len(points) < 8 or depth >= max_depth:
        return [control_points]

    split_idx = np.argmax(errors)
    # Safety bounds
    min_seg = max(4, len(points) // 4)
    split_idx = max(min_seg, min(len(points) - min_seg, split_idx))

    # If can't split effectively
    if split_idx < 4 or len(points) - split_idx < 4:
        return [control_points]

    left = fit_bezier_chain(points[: split_idx + 1], tolerance, max_depth, depth + 1)
    right = fit_bezier_chain(points[split_idx:], tolerance, max_depth, depth + 1)
    return left + right


def fit_single_bezier(points):
    """Least-squares fit cubic Bezier"""
    n = len(points)
    p0, p3 = points[0], points[-1]

    if n <= 2:
        p1 = ((2 * p0[0] + p3[0]) / 3, (2 * p0[1] + p3[1]) / 3)
        p2 = ((p0[0] + 2 * p3[0]) / 3, (p0[1] + 2 * p3[1]) / 3)
        return [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]

    # Chord length parameterization
    dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
    total = np.sum(dists)
    if total < 1e-6:
        return [tuple(p0), tuple(p0), tuple(p3), tuple(p3)]

    t = np.zeros(n)
    t[1:] = np.cumsum(dists) / total

    # Basis functions
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t**2
    A = np.column_stack([B1, B2])

    # Residuals
    B0 = (1 - t) ** 3
    B3 = t**3
    b = points - np.outer(B0, p0) - np.outer(B3, p3)

    try:
        res = np.linalg.lstsq(A, b, rcond=None)[0]
        p1 = tuple(res[0])
        p2 = tuple(res[1])
    except:
        p1 = ((2 * p0[0] + p3[0]) / 3, (2 * p0[1] + p3[1]) / 3)
        p2 = ((p0[0] + 2 * p3[0]) / 3, (p0[1] + 2 * p3[1]) / 3)

    return [tuple(p0), p1, p2, tuple(p3)]


def compute_fitting_errors(points, control_points):
    """Compute min distance from points to curve"""
    p0, p1, p2, p3 = [np.array(p) for p in control_points]
    t = np.linspace(0, 1, max(10, len(points) // 2))
    curve = (
        np.outer((1 - t) ** 3, p0)
        + np.outer(3 * (1 - t) ** 2 * t, p1)
        + np.outer(3 * (1 - t) * t**2, p2)
        + np.outer(t**3, p3)
    )

    # Simple approx: distance to closest sampled point
    # Ideally should use point-to-curve distance
    from scipy.spatial import cKDTree

    tree = cKDTree(curve)
    dists, _ = tree.query(points)
    return dists
