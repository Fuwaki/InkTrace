//! Dense GT 光栅化模块
//!
//! 将贝塞尔曲线渲染为训练所需的各种 Dense 特征图:
//! - Skeleton Map: 骨架二值图 [H, W]
//! - Keypoints Map: 关键点图 [2, H, W]
//!   - Ch0: 拓扑节点 - 样条曲线的起点/终点（必须断开）
//!   - Ch1: 几何锚点 - 多段贝塞尔曲线之间的连接点（建议断开以改善拟合）
//! - Tangent Map: 双倍角切向场 [2, H, W]
//! - Width Map: 宽度图 [H, W]
//! - Offset Map: 亚像素偏移图 [2, H, W]
//! - Image: 带有笔画宽度的渲染图像 [H, W]

use crate::geometry::{double_angle_representation, CubicBezier, Point};

/// Dense GT Maps 结构
#[derive(Clone)]
pub struct DenseMaps {
    pub size: usize,
    /// 渲染的墨迹图像 [H, W]
    pub image: Vec<f32>,
    /// 骨架图 [H, W] (0/1)
    pub skeleton: Vec<f32>,
    /// 关键点图 [2, H, W]
    /// Ch0: 拓扑节点 - 样条曲线的起点/终点
    /// Ch1: 几何锚点 - 多段贝塞尔曲线之间的连接点
    pub keypoints: Vec<f32>,
    /// 切向场 [2, H, W] (cos2θ, sin2θ)
    pub tangent: Vec<f32>,
    /// 宽度图 [H, W]
    pub width: Vec<f32>,
    /// 亚像素偏移图 [2, H, W] (dx, dy)
    pub offset: Vec<f32>,
}

impl DenseMaps {
    /// 创建空的 Dense Maps
    pub fn new(size: usize) -> Self {
        let hw = size * size;
        Self {
            size,
            image: vec![0.0; hw],
            skeleton: vec![0.0; hw],
            keypoints: vec![0.0; 2 * hw], // [2, H, W]
            tangent: vec![0.0; 2 * hw],   // [2, H, W]
            width: vec![0.0; hw],
            offset: vec![0.0; 2 * hw],    // [2, H, W]
        }
    }

    /// 获取 1D 索引
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.size + x
    }

    /// 获取 2D 通道索引 (channel, y, x) -> flat index
    #[inline]
    fn idx2(&self, c: usize, x: usize, y: usize) -> usize {
        c * self.size * self.size + y * self.size + x
    }

    /// 设置骨架像素及其属性
    #[inline]
    pub fn set_skeleton_pixel(
        &mut self,
        x: usize,
        y: usize,
        tangent_cos2: f32,
        tangent_sin2: f32,
        width: f32,
        offset_x: f32,
        offset_y: f32,
    ) {
        let idx = self.idx(x, y);
        let tan_idx0 = self.idx2(0, x, y);
        let tan_idx1 = self.idx2(1, x, y);
        let off_idx0 = self.idx2(0, x, y);
        let off_idx1 = self.idx2(1, x, y);

        self.skeleton[idx] = 1.0;
        self.tangent[tan_idx0] = tangent_cos2;
        self.tangent[tan_idx1] = tangent_sin2;
        self.width[idx] = width;
        self.offset[off_idx0] = offset_x;
        self.offset[off_idx1] = offset_y;
    }

    /// 设置拓扑节点 (样条曲线的起点/终点) - Channel 0
    /// 优先级：高。若该点已被标记为几何锚点，将被覆盖（清除几何锚点标记）。
    #[inline]
    pub fn set_topological_keypoint(&mut self, x: usize, y: usize) {
        if x < self.size && y < self.size {
            let idx_topo = self.idx2(0, x, y);
            let idx_geom = self.idx2(1, x, y);
            self.keypoints[idx_topo] = 1.0;
            self.keypoints[idx_geom] = 0.0; // 强制清除几何锚点，保证互斥
        }
    }

    /// 设置几何锚点 (多段贝塞尔曲线之间的连接点) - Channel 1
    /// 优先级：低。若该点已被标记为拓扑节点，则忽略此标记。
    #[inline]
    pub fn set_geometric_keypoint(&mut self, x: usize, y: usize) {
        if x < self.size && y < self.size {
            let idx_topo = self.idx2(0, x, y);
            let idx_geom = self.idx2(1, x, y);

            // 只有当不是拓扑节点时，才标记为几何锚点
            if self.keypoints[idx_topo] == 0.0 {
                self.keypoints[idx_geom] = 1.0;
            }
        }
    }
}

/// 光栅化配置
#[derive(Clone, Debug)]
pub struct RasterConfig {
    /// 超采样倍数（用于图像渲染）
    pub supersample: f32,
    /// 骨架采样密度（每像素采样步数）
    pub skeleton_density: f32,
    /// 宽度归一化因子
    pub width_normalize: f32,
}

impl Default for RasterConfig {
    fn default() -> Self {
        Self {
            supersample: 2.0,
            skeleton_density: 2.0,
            width_normalize: 10.0,
        }
    }
}

/// 光栅化器
pub struct Rasterizer {
    config: RasterConfig,
}

impl Rasterizer {
    pub fn new() -> Self {
        Self {
            config: RasterConfig::default(),
        }
    }

    /// 渲染多条笔画到 Dense Maps
    pub fn render(&self, strokes: &[CubicBezier], size: usize) -> DenseMaps {
        let mut maps = DenseMaps::new(size);

        // 渲染每条笔画的骨架和属性
        for stroke in strokes {
            self.render_stroke_skeleton(stroke, &mut maps);
        }

        // 渲染图像（带宽度）
        self.render_image(strokes, &mut maps);

        // 标记关键点
        self.mark_keypoints(strokes, &mut maps);

        maps
    }

    /// 渲染单条笔画的骨架和属性
    fn render_stroke_skeleton(&self, stroke: &CubicBezier, maps: &mut DenseMaps) {
        let size = maps.size;

        // 估算曲线长度，决定采样步数
        let est_len = stroke.estimate_length();
        let steps = ((est_len * self.config.skeleton_density) as usize).max(2);

        for i in 0..steps {
            let t = i as f32 / (steps - 1) as f32;

            // 评估曲线点
            let pt = stroke.eval(t);
            let deriv = stroke.eval_derivative(t);
            let tangent = deriv.normalize();
            let width = stroke.eval_width(t);

            // 计算双倍角表示
            let (cos2t, sin2t) = double_angle_representation(&tangent);

            // 像素坐标
            let px = pt.x.floor() as isize;
            let py = pt.y.floor() as isize;

            if px >= 0 && px < size as isize && py >= 0 && py < size as isize {
                let ux = px as usize;
                let uy = py as usize;

                // 计算亚像素偏移：从像素中心到真实点
                let offset_x = pt.x - (px as f32 + 0.5);
                let offset_y = pt.y - (py as f32 + 0.5);

                maps.set_skeleton_pixel(
                    ux,
                    uy,
                    cos2t,
                    sin2t,
                    width / self.config.width_normalize,
                    offset_x,
                    offset_y,
                );
            }
        }
    }

    /// 标记关键点（拓扑节点 + 几何锚点）
    ///
    /// 拓扑节点 (Ch0): 样条曲线的真正端点（起点/终点）
    /// 几何锚点 (Ch1): 多段贝塞尔曲线之间的连接点，提示这里可以分段拟合
    ///
    /// 例如连续路径 [Bezier1, Bezier2, Bezier3]:
    /// - Bezier1.P0 → 拓扑节点（路径起点）
    /// - Bezier1.P3 = Bezier2.P0 → 几何锚点（中间连接点）
    /// - Bezier2.P3 = Bezier3.P0 → 几何锚点（中间连接点）
    /// - Bezier3.P3 → 拓扑节点（路径终点）
    fn mark_keypoints(&self, strokes: &[CubicBezier], maps: &mut DenseMaps) {
        if strokes.is_empty() {
            return;
        }

        let size = maps.size as isize;
        let continuity_threshold = 1.0; // 像素距离阈值，判断是否连续

        // 检测连续性：is_continuous[i] = true 表示 stroke[i] 和 stroke[i+1] 是连续的
        let mut is_continuous = vec![false; strokes.len().saturating_sub(1)];
        for i in 0..strokes.len().saturating_sub(1) {
            let dist = strokes[i].p3.distance(&strokes[i + 1].p0);
            is_continuous[i] = dist < continuity_threshold;
        }

        // 遍历所有笔画，根据连续性标记关键点
        for i in 0..strokes.len() {
            let stroke = &strokes[i];

            // 检查 P0：是路径起点还是中间连接点？
            let is_path_start = i == 0 || !is_continuous[i - 1];
            let x0 = stroke.p0.x.round() as isize;
            let y0 = stroke.p0.y.round() as isize;

            if x0 >= 0 && x0 < size && y0 >= 0 && y0 < size {
                if is_path_start {
                    // 路径起点 → 拓扑节点
                    maps.set_topological_keypoint(x0 as usize, y0 as usize);
                } else {
                    // 中间连接点 → 几何锚点
                    maps.set_geometric_keypoint(x0 as usize, y0 as usize);
                }
            }

            // 检查 P3：是路径终点还是中间连接点？
            let is_path_end = i == strokes.len() - 1 || !is_continuous[i];
            let x3 = stroke.p3.x.round() as isize;
            let y3 = stroke.p3.y.round() as isize;

            if x3 >= 0 && x3 < size && y3 >= 0 && y3 < size {
                if is_path_end {
                    // 路径终点 → 拓扑节点
                    maps.set_topological_keypoint(x3 as usize, y3 as usize);
                }
                // 注意：如果是中间连接点，会在下一个 stroke 的 P0 处标记为几何锚点
            }
        }

        // TODO: 检测交叉点（需要更复杂的算法）
        // 目前暂不实现，因为生成的数据很少有真正的交叉
    }

    /// 渲染带宽度的图像（2x 超采样后降采样）
    fn render_image(&self, strokes: &[CubicBezier], maps: &mut DenseMaps) {
        let size = maps.size;
        let scale = self.config.supersample;
        let canvas_size = (size as f32 * scale) as usize;

        let mut canvas = vec![0.0f32; canvas_size * canvas_size];

        // 在高分辨率画布上绘制
        for stroke in strokes {
            self.render_stroke_to_canvas(stroke, &mut canvas, canvas_size, scale);
        }

        // 降采样到目标分辨率（box filter）
        let scale_int = scale as usize;
        for y in 0..size {
            for x in 0..size {
                let src_x = x * scale_int;
                let src_y = y * scale_int;

                let mut sum = 0.0;
                for dy in 0..scale_int {
                    for dx in 0..scale_int {
                        let idx = (src_y + dy) * canvas_size + (src_x + dx);
                        sum += canvas[idx];
                    }
                }
                maps.image[y * size + x] = sum / (scale_int * scale_int) as f32;
            }
        }
    }

    /// 在超采样画布上绘制单条笔画
    fn render_stroke_to_canvas(
        &self,
        stroke: &CubicBezier,
        canvas: &mut [f32],
        canvas_size: usize,
        scale: f32,
    ) {
        let num_steps = 200;

        for i in 0..num_steps {
            let t = i as f32 / (num_steps - 1) as f32;
            let pt = stroke.eval(t);
            let w = stroke.eval_width(t);

            let radius = (w * scale / 2.0).max(0.5);
            let r2 = radius * radius;

            let center_x = pt.x * scale;
            let center_y = pt.y * scale;

            // 填充圆形区域
            let min_x = (center_x - radius).floor().max(0.0) as isize;
            let max_x = (center_x + radius).ceil().min(canvas_size as f32 - 1.0) as isize;
            let min_y = (center_y - radius).floor().max(0.0) as isize;
            let max_y = (center_y + radius).ceil().min(canvas_size as f32 - 1.0) as isize;

            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let dx = x as f32 - center_x;
                    let dy = y as f32 - center_y;
                    if dx * dx + dy * dy <= r2 {
                        let idx = y as usize * canvas_size + x as usize;
                        canvas[idx] = 1.0;
                    }
                }
            }
        }
    }
}

impl Default for Rasterizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rasterize_single_stroke() {
        let stroke = CubicBezier::new(
            Point::new(10.0, 32.0),
            Point::new(20.0, 20.0),
            Point::new(44.0, 44.0),
            Point::new(54.0, 32.0),
            3.0,
            2.5,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);

        // 检查骨架不为空
        let skeleton_sum: f32 = maps.skeleton.iter().sum();
        assert!(skeleton_sum > 0.0, "Skeleton should not be empty");

        // 检查图像不为空
        let image_sum: f32 = maps.image.iter().sum();
        assert!(image_sum > 0.0, "Image should not be empty");

        // 检查 keypoints 两个通道
        assert_eq!(maps.keypoints.len(), 2 * 64 * 64);

        // 检查拓扑节点标记了端点
        let topo_sum: f32 = maps.keypoints[..64 * 64].iter().sum();
        assert!(topo_sum >= 2.0, "Should have at least 2 topological keypoints");
    }
}
