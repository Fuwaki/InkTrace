//! 笔画生成模块
//!
//! 提供各种类型笔画的生成逻辑
//!
//! 注意：所有 RNG 使用 StdRng::from_entropy() 确保在 fork() 后能获得新的随机种子

use crate::geometry::{CubicBezier, Point};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::f32::consts::PI;

/// 笔画生成配置
#[derive(Clone, Debug)]
pub struct StrokeConfig {
    /// 画布大小
    pub canvas_size: f32,
    /// 笔画长度范围（相对于画布大小的比例）
    pub length_range: (f32, f32),
    /// 宽度范围
    pub width_range: (f32, f32),
    /// 控制点手柄长度系数范围（相对于弦长）
    pub handle_ratio_range: (f32, f32),
    /// 方向变化范围（弧度）
    pub direction_variation: f32,
}

impl Default for StrokeConfig {
    fn default() -> Self {
        Self {
            canvas_size: 64.0,
            length_range: (0.15, 0.4),
            width_range: (1.5, 4.5),
            handle_ratio_range: (0.25, 0.45),
            direction_variation: PI / 4.0,
        }
    }
}

impl StrokeConfig {
    pub fn for_canvas(size: usize) -> Self {
        Self {
            canvas_size: size as f32,
            ..Default::default()
        }
    }

    /// 单笔画配置（较长的笔画）
    pub fn single_stroke(size: usize) -> Self {
        Self {
            canvas_size: size as f32,
            length_range: (0.3, 0.8),
            width_range: (2.0, 5.0),
            ..Default::default()
        }
    }

    /// 独立多笔画配置（较短的笔画，避免过多重叠）
    pub fn independent_stroke(size: usize) -> Self {
        Self {
            canvas_size: size as f32,
            length_range: (0.12, 0.35),
            width_range: (1.5, 4.0),
            ..Default::default()
        }
    }

    /// 连续笔画段配置（适中长度）
    pub fn continuous_segment(size: usize) -> Self {
        Self {
            canvas_size: size as f32,
            length_range: (0.12, 0.35),
            width_range: (1.5, 4.0),
            ..Default::default()
        }
    }
}

/// 在画布内生成随机起始点
pub fn random_start_point<R: Rng>(rng: &mut R, size: f32) -> Point {
    let margin = size * 0.1;
    let range = size - 2.0 * margin;
    Point::new(
        rng.gen::<f32>() * range + margin,
        rng.gen::<f32>() * range + margin,
    )
}

/// 生成自然的贝塞尔曲线段
///
/// # Arguments
/// * `start_point` - 起始点
/// * `incoming_tangent` - 入射切向量（用于 G1 连续性）
/// * `config` - 生成配置
/// * `rng` - 随机数生成器
pub fn generate_natural_segment<R: Rng>(
    start_point: Point,
    incoming_tangent: Option<&Point>,
    config: &StrokeConfig,
    rng: &mut R,
) -> CubicBezier {
    let size = config.canvas_size;

    // 1. 确定大致方向（弦向量 P0 -> P3）
    let chord_dir = if let Some(tan) = incoming_tangent {
        // 延续前一段方向，但允许 ±60° 转弯
        tan.rotate(rng.gen_range(-PI / 3.0..PI / 3.0))
    } else {
        // 随机起始方向
        Point::from_angle(rng.gen_range(0.0..2.0 * PI))
    }
    .normalize();

    // 2. 确定弦长
    let len = size * rng.gen_range(config.length_range.0..config.length_range.1);
    let p3 = &start_point + &(&chord_dir * len);

    // 3. 确定控制点 P1（出发方向）
    let p1_dir = if let Some(tan) = incoming_tangent {
        // G1 连续：沿入射切向方向
        tan.clone()
    } else {
        // 首段：略微偏离弦向
        chord_dir.rotate(rng.gen_range(-config.direction_variation..config.direction_variation))
    }
    .normalize();

    let h1_len = len * rng.gen_range(config.handle_ratio_range.0..config.handle_ratio_range.1);
    let p1 = &start_point + &(&p1_dir * h1_len);

    // 4. 确定控制点 P2（着陆方向，从 P3 反向）
    let landing_dir = chord_dir
        .rotate(rng.gen_range(-config.direction_variation..config.direction_variation))
        .normalize();
    let h2_len = len * rng.gen_range(config.handle_ratio_range.0..config.handle_ratio_range.1);
    let p2 = &p3 - &(&landing_dir * h2_len);

    // 5. 宽度
    let w_start = rng.gen_range(config.width_range.0..config.width_range.1);
    let w_end = rng.gen_range(config.width_range.0..config.width_range.1);

    // 6. 限制所有点在画布内
    let mut bezier = CubicBezier::new(
        start_point.clamp(0.0, size),
        p1.clamp(0.0, size),
        p2.clamp(0.0, size),
        p3.clamp(0.0, size),
        w_start,
        w_end,
    );
    bezier.clamp_to_canvas(size);
    bezier
}

/// 创建新的随机数生成器（使用操作系统熵源）
///
/// 这确保即使在 fork() 之后，每个进程也能获得独立的随机序列
#[inline]
pub fn new_rng() -> StdRng {
    StdRng::from_entropy()
}

/// 生成单条独立笔画
pub fn generate_single_stroke(size: usize, rng: &mut StdRng) -> CubicBezier {
    let config = StrokeConfig::single_stroke(size);
    let start = random_start_point(rng, config.canvas_size);
    generate_natural_segment(start, None, &config, rng)
}

/// 生成多条独立笔画
pub fn generate_independent_strokes(
    size: usize,
    count: usize,
    rng: &mut StdRng,
) -> Vec<CubicBezier> {
    let config = StrokeConfig::independent_stroke(size);
    let mut strokes = Vec::with_capacity(count);

    for _ in 0..count {
        let start = random_start_point(rng, config.canvas_size);
        let stroke = generate_natural_segment(start, None, &config, rng);
        strokes.push(stroke);
    }

    // 按 y 坐标排序（稳定顺序，便于匹配）
    strokes.sort_by(|a, b| {
        a.p0.y
            .partial_cmp(&b.p0.y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    strokes
}

/// 生成连续多段笔画（G1 连续）
pub fn generate_continuous_path(
    size: usize,
    num_segments: usize,
    rng: &mut StdRng,
) -> Vec<CubicBezier> {
    let config = StrokeConfig::continuous_segment(size);
    let mut strokes = Vec::with_capacity(num_segments);

    // 随机起始点
    let mut current_pos = random_start_point(rng, config.canvas_size);
    let mut incoming_tangent: Option<Point> = None;

    for _ in 0..num_segments {
        let stroke = generate_natural_segment(
            current_pos.clone(),
            incoming_tangent.as_ref(),
            &config,
            rng,
        );

        // 更新状态：下一段从 P3 开始，入射切向为 P2->P3
        current_pos = stroke.p3.clone();
        incoming_tangent = Some(stroke.end_tangent());

        strokes.push(stroke);
    }

    strokes
}

/// 生成多条独立的连续路径
pub fn generate_multi_path<F>(
    size: usize,
    num_paths: usize,
    segments_per_path: F,
    rng: &mut StdRng,
) -> Vec<CubicBezier>
where
    F: Fn(&mut StdRng) -> usize,
{
    let mut all_strokes = Vec::new();

    for _ in 0..num_paths {
        let num_segs = segments_per_path(rng);
        let path = generate_continuous_path(size, num_segs, rng);
        all_strokes.extend(path);
    }

    all_strokes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_stroke_generation() {
        let mut rng = new_rng();
        let stroke = generate_single_stroke(64, &mut rng);

        // 检查所有点都在画布内
        assert!(stroke.p0.x >= 0.0 && stroke.p0.x <= 64.0);
        assert!(stroke.p0.y >= 0.0 && stroke.p0.y <= 64.0);
        assert!(stroke.p3.x >= 0.0 && stroke.p3.x <= 64.0);
        assert!(stroke.p3.y >= 0.0 && stroke.p3.y <= 64.0);
    }

    #[test]
    fn test_continuous_path() {
        let mut rng = new_rng();
        let path = generate_continuous_path(64, 3, &mut rng);

        assert_eq!(path.len(), 3);

        // 检查连续性：P3[i] == P0[i+1]
        for i in 0..path.len() - 1 {
            let d = path[i].p3.distance(&path[i + 1].p0);
            assert!(d < 1e-3, "Path not continuous at segment {}", i);
        }
    }

    #[test]
    fn test_rng_independence() {
        // 测试每次调用 new_rng() 都能获得不同的随机数
        let mut rng1 = new_rng();
        let mut rng2 = new_rng();

        let v1: f32 = rng1.gen();
        let v2: f32 = rng2.gen();

        // 两个独立 RNG 生成的第一个数应该不同（概率极高）
        assert_ne!(
            v1, v2,
            "Two independent RNGs should generate different values"
        );
    }
}
