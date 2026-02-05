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

### Step 1.1: Extract Vertices (Node Extraction)
从 Junction Map 中提取关键点（Nodes）。

1.  **Thresholding**: 二值化 $J$ (e.g., threshold > 0.3)。
2.  **Connected Components (CCL)**: 对二值化后的 Junction 区域进行连通域分析。
    *   因为一个交叉点在 heatmap 上可能是一团 3x3 或 5x5 的像素簇。
3.  **Centroid Calculation**: 计算每个连通域的质心 (Centroid)，作为初步的 Node 坐标 $v_i$。
    *   *Refinement*: 利用 Offset Map 对质心坐标进行亚像素修正：$v'_i = v_i + O(v_i)$。

### Step 1.2: Skeleton Thinning / Cleaning (骨架预处理)
Skeleton Map 输出的骨架可能在转弯处有几个像素宽，或者有毛刺。

1.  **Thresholding**: 二值化 $S$ (e.g., threshold > 0.4)。
2.  **Subtract Junctions**: 将 Junction 区域从 Skeleton 中“挖掉”。
    *   $S_{clean} = S \setminus J_{dilated}$
    *   目的：打断交叉处的连接，使每条笔画成为独立的连通域（Segments）。

### Step 1.3: Build Adjacency (建立邻接关系)
现在我们需要把 Node 和 Segment 连起来。

1.  **Label Segments**: 对 $S_{clean}$ 进行连通域分析，每个连通域代表一条潜在的边 $e_k$。
2.  **Proximity Search**: 对于每个 Segment $e_k$ 的两个端点像素，搜索附近的 Node $v_i$。
    *   如果距离 $dist(end\_pixel, v_i) < R$ (搜索半径)，则认为边 $e_k$ 连接到了节点 $v_i$。
3.  **Graph Representation**: 构建邻接表。
    *   Node $v_i$: `{coord: (x, y), transitions: [edge_id_1, edge_id_2, ...]}`
    *   Edge $e_k$: `{nodes: (u, v), pixel_chain: [...]}`

---

## 3. Phase 2: Path Tracing (路径追踪)

这一步的目标是将 Segment 的像素点变成有序的、高精度的点序列。

### Step 2.1: Ordered Pixel Walking
对于每条边 $e_k$（此时只是无序的像素堆）：

1.  **Find Start**: 从连接的一个 Node $u$ 开始。
2.  **Greedy Walk**: 寻找与当前像素相邻（8邻域）且在 Skeleton Mask 内的未访问像素。
    *   *Ambiguity Handling*: 如果有多个相邻像素怎么办？利用 Tangent Map。
    *   选择与当前切线方向 $T(current)$ 最一致的邻居。
3.  **Sub-pixel Correction**: 收集沿途的每一个像素坐标 $(x, y)$，并加上 Offset Map 的修正值：
    *   $P_t = (x, y) + O(x, y)$
4.  **Result**: 得到一个有序的点序列 $P = [p_0, p_1, \dots, p_N]$。

### Step 2.2: Tangent Consistency Check
检查点序列的方向一致性。Tangent Map 存储的是 $(\cos 2\theta, \sin 2\theta)$，在还原为角度时有 $180^\circ$ 的模糊性。我们需要根据行进方向（$p_{t+1} - p_t$）来“解冻”这个角度，赋予它唯一的方向。

---

## 4. Phase 3: Curve Fitting (几何拟合)

这一步的目标是用最少的贝塞尔曲线段拟合点序列 $P$。

### Algorithm: Adaptive Least Squares Fitting (自适应最小二乘法)

这是一个递归过程（类似于 Schneider's Algorithm）。

**Input**: Ordered Point Chain $P$.
**Output**: List of Bezier Curves.

1.  **Base Case**: 尝试用**一条**三次贝塞尔曲线拟合整个 $P$。
    *   虽然贝塞尔曲线参数 $t$ 未知，可以使用 "Chord Length Parameterization" (弦长参数化) 来估算每个点 $p_i$ 对应的 $t_i$。
    *   固定起点 $B(0) = p_0$ 和终点 $B(1) = p_N$。
    *   根据切线方向 $T(p_0)$ 和 $T(p_N)$ 固定控制点 $P_1, P_2$ 的方向，只求解它们的长度（标量）。
    *   或者使用全参数最小二乘法求解最佳控制点。

2.  **Error Check**: 计算拟合曲线与原始点集 $P$ 的最大误差 (Max Deviation)。

3.  **Split**: 如果最大误差 > $\epsilon$ (容忍阈值)：
    *   在误差最大的点处将点集 $P$ 切分为两段：$P_{left}, P_{right}$。
    *   递归地对两段分别进行拟合。

4.  **Merge**: 最终将所有曲线段连接起来。

### Optimization: Tangent Constraint
利用神经网络预测的 Tangent Map $T$ 作为强约束。
*   在拟合时，强制贝塞尔曲线在端点处的切线方向等于 $T(p_{start})$ 和 $T(p_{end})$。
*   这能保证曲线在经过 Node（交叉点）时是平滑自然的，或者按照模型预测的方向突变（如折角）。

---

## 5. Implementation Roadmap (Python Prototype)

为了快速验证，我们先用 Python 实现一个原型的 `GraphReconstructor` 类。

### Class Structure
```python
class GraphReconstructor:
    def __init__(self, config):
        self.config = config # thresholds, epsilon, etc.

    def process(self, pred_maps):
        """
        Main pipeline
        """
        # 1. Preprocessing
        nodes = self._extract_nodes(pred_maps['junction'])
        skeleton = self._clean_skeleton(pred_maps['skeleton'], nodes)
        
        # 2. Graph Building
        segments = self._extract_segments(skeleton)
        graph = self._build_topology(nodes, segments)
        
        # 3. Path Tracing & Fitting
        vector_paths = []
        for edge in graph.edges:
            pixel_chain = self._trace_pixels(edge)
            bezier_path = self._fit_bezier(pixel_chain)
            vector_paths.append(bezier_path)
            
        return vector_paths
```

### Future Work (Rust)
一旦 Python 原型验证了逻辑的闭环（即 输入-> SVG -> 视觉对齐），我们就将这一逻辑移植到 `ink_trace_rs` 中，利用 Rayon 并行处理图像的每一个 Tile。
