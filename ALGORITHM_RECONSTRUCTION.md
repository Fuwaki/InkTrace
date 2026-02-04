# InkTrace Reconstruction Algorithm: From Pixels to Bezier Curves

**Version**: 1.1
**Status**: Implementation (Phase 2 Active)
**Scope**: Dense Map Generation (Ground Truth) & Post-processing (Reconstruction) logic.

---

## 1. Overview (总览)

我们的输入是神经网络预测出的 5 张特征图（本阶段暂时忽略 Width Map），目标是输出一组有序的贝塞尔曲线路径。

### Inputs (Dense Prediction Maps)
这些图由 `dense_gen.py` 或 `DenseVectorNet` 生成。

*   **Skeleton Map ($S$)**: $H \times W \times 1$, $\in [0, 1]$. 描述像素是否在笔画骨架上。
*   **Junction Map ($J$)**: $H \times W \times 1$, $\in [0, 1]$. 描述像素是否是端点或交叉点。
*   **Tangent Map ($T$)**: $H \times W \times 2$, $(\cos 2\theta, \sin 2\theta)$. 描述骨架的切线方向（消除方向模糊，使用双倍角表示）。
*   **Offset Map ($O$)**: $H \times W \times 2$, $(\delta x, \delta y)$. 描述像素中心到真实骨架中心的亚像素偏移 ($\in [-0.5, 0.5]$)。
*   **Width Map ($W$)**: $H \times W \times 1$. 描述笔画在该点的宽度。

### Outputs
*   **Graph Structure**: $G=(V, E)$, where $V$ are junctions/endpoints, $E$ are strokes.
*   **Vector Paths**: List of Cubic Bezier Curves representing each edge $E$.

---

## 1.5. Data Representation & Generation (Ground Truth Algorithm)

为了训练 Dense Prediction 网络，我们首先实现了逆向过程：从矢量数据生成 Dense Maps (`dense_gen.py`)。这定义了重建算法的“标准答案”。

### Algorithm: Vector to Dense Maps
**Input**: Vector Strokes (Bézier Control Points $P_0, P_1, P_2, P_3$, Widths $w_{start}, w_{end}$).
**Output**: Dense Maps ($S, J, T, O, W$).

1.  **Curve Sampling**:
    *   估算贝塞尔曲线的弦长与控制网格长度，确定采样步数 $N$。
    *   保证采样间隔 $\Delta t$ 使得像素空间步长 $< 0.5$ px，从而生成连续的骨架。

2.  **Attribute Calculation**:
    *   **Tangent**: 计算曲线导数 $B'(t)$，归一化得到切向量 $(u_x, u_y)$。转换为双倍角表示：
        *   $T_1 = \cos 2\theta = u_x^2 - u_y^2$
        *   $T_2 = \sin 2\theta = 2 u_x u_y$
        *   *优势*: 解决了 $0^\circ$ 和 $180^\circ$ 的方向模糊问题，且在 $0$ 附近连续。
    *   **Offset**: 对于采样点位置 $P(t) = (b_x, b_y)$，其落在像素 $(x, y) = (\lfloor b_x \rfloor, \lfloor b_y \rfloor)$。
        *   Pixel Center: $(x+0.5, y+0.5)$
        *   Offset: $(\delta x, \delta y) = (b_x, b_y) - (x+0.5, y+0.5)$
    *   **Width**: 沿 $t$ 线性插值宽度。

3.  **Rasterization (Splatting)**:
    *   遍历所有采样点，更新对应的 Feature Maps。
    *   对于重叠区域（交叉点），通常保留最后绘制的值或根据深度排序（当前实现为简单覆盖）。
    *   **Junctions**: 仅在笔画的 $P_0$ 和 $P_3$ 处标记 Junction Map。

---

## 2. Phase 1: Graph Construction (图构建)

这一步的目标是将光栅图像转化为拓扑图结构。

### Step 1.1: Skeleton Preprocessing (骨架预处理)
Skeleton Map 输出的骨架可能在转弯处有几个像素宽，需要细化为单像素宽度。

1.  **Thresholding**: 二值化 $S$ (e.g., threshold > 0.2)。
2.  **Morphological Thinning**: 使用形态学细化算法（Zhang-Suen 或 `skimage.morphology.thin`）将骨架细化为单像素宽度。
    *   $S_{thin} = \text{thin}(S > \text{threshold})$
    *   目的：确保后续的邻域分析和端点检测准确。

### Step 1.2: Extract Vertices (Node Extraction)
从细化的 Skeleton Map 中自动检测端点和交叉点（Nodes）。

**方法：基于 8-邻域连通性分析**

1.  **Neighbor Counting**: 使用卷积核计算每个骨架像素的 8-邻居数量。
    ```python
    kernel = [[1, 1, 1],
              [1, 0, 1],
              [1, 1, 1]]
    neighbor_count = convolve(S_thin, kernel)
    ```

2.  **Endpoint Detection**:
    *   端点定义：恰好有 1 个骨架像素邻居
    *   $V_{endpoint} = \{p \in S_{thin} \mid \text{neighbor\_count}(p) = 1\}$

3.  **Junction Detection**:
    *   交叉点定义：有 3 个或更多骨架像素邻居
    *   候选：$V_{junction}^{cand} = \{p \in S_{thin} \mid \text{neighbor\_count}(p) \ge 3\}$
    *   **验证**: 使用连通分量分析（CCL）验证真正的分叉点
        *   提取 3×3 邻域，移除中心点
        *   计算剩余连通分量数，$\ge 3$ 则确认为真交叉点
        *   目的：消除锯齿导致的假阳性
    *   **可选验证**: 如果提供了 Junction Map $J$，用其响应值验证候选点
        *   在 3×3 邻域内搜索 $\max(J) > \text{junction\_threshold}$

### Step 1.3: Stroke Tracing Strategy (笔画追踪策略)

**核心思想**：从每个端点出发，沿骨架行走直到遇到另一个端点或交叉点。

**关键设计决策**：
- **交叉点穿越** (`cross_junctions`): 当遇到交叉点时，允许穿过它继续追踪，而不是停止。
  - 优势：保持长笔画的完整性，避免在交叉点处断裂
  - 实现：交叉点像素不被标记为"已访问"，可以多次经过
- **已访问像素管理**:
  - `visited_pixels`: 已完全追踪的非交叉点像素（不能再访问）
  - `junctions`: 交叉点像素（可以穿越，不标记为已访问）

---

## 3. Phase 2: Path Tracing (路径追踪)

这一步的目标是将骨架像素转化为有序的笔画路径（Strokes）。

### Step 2.1: Iterative Stroke Tracing

**算法流程**（`_trace_strokes` 和 `_walk_from_point_v2`）：

1.  **Main Loop**: 遍历所有端点，从每个未使用的端点开始追踪

2.  **Walking Algorithm** (`_walk_from_point_v2`):
    ```python
    def walk(start_point, skeleton, endpoints, junctions, visited_pixels, ...):
        current = start_point
        pixels = [current]
        local_visited = {current}  # 防止循环

        for step in range(max_steps):
            # 获取 8-neighbor
            neighbors = get_8neighbors(current, skeleton_pixels)

            # 筛选可访问的邻居
            valid_neighbors = []
            for n in neighbors:
                if n in local_visited:
                    continue  # 本次追踪已访问
                if n in junctions:
                    valid_neighbors.append(n)  # 交叉点可以穿越
                    continue
                if n not in visited_pixels:
                    valid_neighbors.append(n)  # 未访问的非交叉点

            if len(valid_neighbors) == 0:
                break  # 死路

            # 选择下一个像素
            if len(valid_neighbors) == 1:
                next_pixel = valid_neighbors[0]
            else:
                # 多个选择：使用 tangent 或方向一致性
                next_pixel = select_by_tangent_or_direction(
                    current, valid_neighbors, prev_direction, tangent_map
                )

            pixels.append(next_pixel)
            local_visited.add(next_pixel)

            # 更新全局状态
            if next_pixel not in junctions:
                visited_pixels.add(next_pixel)  # 标记为已访问

            prev_direction = (next_pixel - current)
            current = next_pixel

            # 停止条件：到达另一个端点
            if current in endpoints and current != start_point:
                break

        return pixels
    ```

3.  **Ambiguity Resolution** (多邻居选择策略):

    *   **Strategy A: Tangent Guidance** (如果提供 Tangent Map)
        *   从 Tangent Map $T = (\cos 2\theta, \cos 2\theta)$ 还原切线方向
        *   $\theta = \frac{1}{2} \arctan2(T_1, T_0)$
        *   由于双倍角表示存在 $180^\circ$ 模糊，选择与 `prev_direction` 更一致的方向
        *   计算各候选方向与切线方向的点积，选择最大者

    *   **Strategy B: Direction Consistency** (Fallback)
        *   如果没有 Tangent Map，优先选择与 `prev_direction` 点积最大的邻居
        *   即倾向于继续沿当前方向直行

4.  **Loop Handling**:
    *   使用 `local_visited` 集合记录本次追踪访问的点
    *   如果遇到已访问的点，跳过以防止无限循环

5.  **Remaining Segments**:
    *   完成所有端点追踪后，检查剩余未访问的骨架像素
    *   对孤立环（无端点）进行额外追踪

### Step 2.2: Width Estimation

对每条追踪到的笔画，估计其平均宽度：

```python
widths = [width_map[row, col] for (row, col) in stroke_pixels]
stroke_width = np.mean(widths) if widths else 2.0
```

**注意**：当前实现未使用 Offset Map 进行亚像素修正（可扩展）。

### Step 2.3: Coordinate System

**内部处理**: 使用 `(row, col)` 即 `(y, x)` 格式，符合 NumPy 数组索引习惯。

**输出格式**: 转换为 `(x, y)` 格式用于可视化和后续拟合：
```python
points_xy = [(col, row) for (row, col) in stroke_pixels_rc]
```

---

## 4. Phase 3: Curve Fitting (几何拟合)

这一步的目标是用最少的贝塞尔曲线段拟合点序列 $P$。

### Algorithm: Adaptive Least Squares Fitting (自适应最小二乘法)

**实现**：`fit_bezier_curves`, `fit_bezier_chain`, `fit_single_bezier`

**Input**: Ordered Point Chain $P = [(x_0, y_0), (x_1, y_1), \dots, (x_N, y_N)]$.
**Output**: List of Cubic Bezier Curves, each with 4 control points $[P_0, P_1, P_2, P_3]$.

### Step 3.1: Single Curve Fitting (`fit_single_bezier`)

**方法**：基于弦长参数化的最小二乘法

1.  **Fix Endpoints**:
    *   $P_0 = p_0$ (起点)
    *   $P_3 = p_N$ (终点)

2.  **Chord Length Parameterization** (弦长参数化):
    *   计算相邻点之间的弦长：$L_i = \|p_{i+1} - p_i\|_2$
    *   累积长度：$t_0 = 0, \quad t_i = \frac{\sum_{j=0}^{i-1} L_j}{\sum_{j=0}^{N-1} L_j}$

3.  **Least Squares Formulation**:
    *   三次贝塞尔曲线：$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t)t^2 P_2 + t^3 P_3$
    *   移项得到：$B(t) - (1-t)^3 P_0 - t^3 P_3 = 3(1-t)^2 t \cdot P_1 + 3(1-t)t^2 \cdot P_2$
    *   构建线性系统 $A \cdot [P_1, P_2] = b$，其中：
        *   $A = [3(1-t_i)^2 t_i, \quad 3(1-t_i)t_i^2]$ (基函数矩阵)
        *   $b = p_i - (1-t_i)^3 P_0 - t_i^3 P_3$ (目标向量)
    *   使用 `np.linalg.lstsq` 分别求解 $x$ 和 $y$ 坐标

4.  **Sanity Check**:
    *   检查控制点是否合理（不要离端点太远）
    *   如果 $\|P_1 - P_0\| > 2\|P_3 - P_0\|$ 或 $\|P_2 - P_3\| > 2\|P_3 - P_0\|$，退化为直线

### Step 3.2: Error Computation (`compute_fitting_errors`)

计算拟合曲线与原始点集的距离：

1.  **Sample Curve**: 在 $t \in [0, 1]$ 上密集采样曲线（采样点数 = $2N$）
2.  **Point-to-Curve Distance**: 对每个原始点 $p_i$，计算到曲线采样点的最小距离
3.  **Return**: 误差数组 $[e_0, e_1, \dots, e_N]$

### Step 3.3: Adaptive Recursive Fitting (`fit_bezier_chain`)

递归分割策略：

```python
def fit_bezier_chain(points, tolerance, max_depth=5, depth=0):
    # 1. 尝试单曲线拟合
    control_points = fit_single_bezier(points)
    errors = compute_fitting_errors(points, control_points)
    max_error = np.max(errors)

    # 2. 停止条件
    if (max_error <= tolerance or          # 误差足够小
        len(points) < 8 or                # 点太少
        depth >= max_depth):              # 递归太深
        return [control_points]

    # 3. 在最大误差点分割
    split_idx = np.argmax(errors)

    # 确保分割点不靠近端点（至少 4 个点或 1/4 长度）
    split_idx = np.clip(split_idx, min_segment, len(points) - min_segment)

    # 4. 递归拟合两段
    left_curves = fit_bezier_chain(points[:split_idx+1], tolerance, max_depth, depth+1)
    right_curves = fit_bezier_chain(points[split_idx:], tolerance, max_depth, depth+1)

    return left_curves + right_curves
```

**参数说明**：
- `tolerance`: 最大允许误差（默认 3.0 像素）
- `max_depth`: 最大递归深度（默认 5），避免过度分割
- `min_segment`: 最小段长（$\max(4, N/4)$），确保每段至少有 4 个点

### Step 3.4: Post-processing

对每条笔画（Stroke）：

1.  **Short Strokes** ($< 4$ 个点):
    *   直接返回直线贝塞尔（控制点均匀分布在端点之间）

2.  **Normal Strokes**:
    *   调用 `fit_bezier_chain` 进行递归拟合
    *   继承笔画宽度属性

**Note**: 当前实现**未使用 Tangent Map 约束**控制点方向，完全基于几何拟合。未来可以扩展为在 $P_1, P_2$ 求解时加入切线方向约束。

---

## 5. Implementation Roadmap

### Completed (Phase 2 Preparation)
- [x] **Data Generation (`dense_gen.py`)**: 实现从矢量数据到 Dense Maps 的高精度转换。
    - 实现了双倍角切线表示。
    - 实现了亚像素 Offset 计算。
- [x] **Dataset (`datasets.py`)**: 封装了 Rust 数据流与 Dense Gen，支持渐进学习。
- [x] **Visualization (`visualize_dense_gt.py`)**: 验证了 GT Maps 的正确性（HSV 可视化切线场）。
- [x] **Network (`DenseVectorNet`)**: 实现了配套的 Hybrid U-Net 架构训练管线 `train_dense.py`。

### Completed (Phase 3 - Reconstruction)
- [x] **Python Prototype (`graph_reconstruction.py`)**: 实现完整的图重建算法。
    - `SimpleGraphReconstructor`: 骨架细化、节点检测、笔画追踪
    - `fit_bezier_curves`: 自适应贝塞尔曲线拟合
    - 支持交叉点穿越、Tangent 引导、宽度估计
- [x] **End-to-End Test (`test_reconstruction.py`)**: 完整的测试和可视化流程。
    - 支持从数据集随机采样测试
    - 支持自定义图片测试（`--image` 参数）
    - 生成详细的可视化结果（Input, Skeleton, Junction, Bezier Curves, Statistics）

### Pending (Future Improvements)
- [ ] **Offset Map Integration**: 在路径追踪中使用 Offset Map 进行亚像素坐标修正
- [ ] **Tangent Constraint in Fitting**: 在贝塞尔曲线拟合时使用 Tangent Map 约束控制点方向
- [ ] **SVG Export**: 将拟合结果导出为标准 SVG 格式
- [ ] **Rust Porting**: 将核心算法移植到 `ink_trace_rs` 以提升性能
- [ ] **Error Metrics**: 实现重建质量的定量评估（Chamfer Distance, F-score 等）

---

## 6. Configuration Parameters

### Graph Reconstruction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `skeleton_threshold` | 0.2 | Skeleton binarization threshold |
| `junction_threshold` | 0.5 | Junction Map validation threshold |
| `min_stroke_length` | 5 | Minimum pixels for a valid stroke |
| `use_thinning` | True | Apply morphological thinning to skeleton |
| `cross_junctions` | True | Allow passing through junction pixels |

### Curve Fitting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tolerance` | 3.0 | Max fitting error (pixels) before splitting |
| `max_depth` | 5 | Maximum recursion depth for curve splitting |
| `min_segment` | max(4, N/4) | Minimum segment length after splitting |

---

## 7. Usage Examples

### Testing on Custom Image

```bash
python test_reconstruction.py \
    --checkpoint best_dense_model.pth \
    --image path/to/your/image.png \
    --output results/ \
    --size 64 \
    --threshold 0.2 \
    --invert  # If image has white background
```

### Testing on Dataset Samples

```bash
python test_reconstruction.py \
    --checkpoint best_dense_model.pth \
    --num-samples 10 \
    --output results/
```

### Programmatic Usage

```python
from graph_reconstruction import SimpleGraphReconstructor, fit_bezier_curves

# Initialize reconstructor
reconstructor = SimpleGraphReconstructor(config={
    'skeleton_threshold': 0.2,
    'junction_threshold': 0.5,
    'cross_junctions': True,
})

# Process prediction maps
strokes = reconstructor.process(pred_maps)

# Fit bezier curves
bezier_curves = fit_bezier_curves(strokes, tolerance=3.0)

# Output format
for curve in bezier_curves:
    p0, p1, p2, p3 = curve['points']  # (x, y) tuples
    width = curve['width']
```
