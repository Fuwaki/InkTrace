//! 分层训练数据生成器
//!
//! 为不同训练阶段提供数据生成:
//! - Stage 0: 单笔画
//! - Stage 1-3: 多独立笔画（数量递增）
//! - Stage 4-6: 多段连续笔画（段数递增）
//! - Stage 7-9: 混合模式（多条多段路径）
//!
//! 注意：使用 StdRng::from_entropy() 确保每个并行任务获得独立的随机序列

use crate::geometry::CubicBezier;
use crate::rasterize::{DenseMaps, Rasterizer};
use crate::stroke::{
    generate_continuous_path, generate_independent_strokes, generate_multi_path,
    generate_single_stroke, new_rng,
};
use rand::prelude::*;
use rayon::prelude::*;

/// 训练阶段配置
#[derive(Clone, Debug)]
pub struct CurriculumConfig {
    /// 阶段编号
    pub stage: u32,
    /// 阶段名称
    pub name: String,
    /// 生成模式
    pub mode: GenerationMode,
}

/// 生成模式
#[derive(Clone, Debug)]
pub enum GenerationMode {
    /// 单条独立笔画
    SingleStroke,
    /// 多条独立笔画
    IndependentStrokes { min_count: usize, max_count: usize },
    /// 单条连续多段笔画
    ContinuousPath {
        min_segments: usize,
        max_segments: usize,
    },
    /// 多条连续路径（混合）
    MultiPath {
        min_paths: usize,
        max_paths: usize,
        min_segments: usize,
        max_segments: usize,
    },
}

impl CurriculumConfig {
    /// 获取预定义的训练阶段配置
    pub fn from_stage(stage: u32) -> Self {
        match stage {
            // 单笔画
            0 => Self {
                stage: 0,
                name: "single_stroke".into(),
                mode: GenerationMode::SingleStroke,
            },
            // 独立多笔画（递增）
            1 => Self {
                stage: 1,
                name: "independent_1_3".into(),
                mode: GenerationMode::IndependentStrokes {
                    min_count: 1,
                    max_count: 3,
                },
            },
            2 => Self {
                stage: 2,
                name: "independent_2_5".into(),
                mode: GenerationMode::IndependentStrokes {
                    min_count: 2,
                    max_count: 5,
                },
            },
            3 => Self {
                stage: 3,
                name: "independent_3_8".into(),
                mode: GenerationMode::IndependentStrokes {
                    min_count: 3,
                    max_count: 8,
                },
            },
            // 连续多段笔画（递增）
            4 => Self {
                stage: 4,
                name: "continuous_2_3".into(),
                mode: GenerationMode::ContinuousPath {
                    min_segments: 2,
                    max_segments: 3,
                },
            },
            5 => Self {
                stage: 5,
                name: "continuous_3_5".into(),
                mode: GenerationMode::ContinuousPath {
                    min_segments: 3,
                    max_segments: 5,
                },
            },
            6 => Self {
                stage: 6,
                name: "continuous_4_8".into(),
                mode: GenerationMode::ContinuousPath {
                    min_segments: 4,
                    max_segments: 8,
                },
            },
            // 混合模式（多条多段路径）
            7 => Self {
                stage: 7,
                name: "multi_path_2x3".into(),
                mode: GenerationMode::MultiPath {
                    min_paths: 1,
                    max_paths: 2,
                    min_segments: 2,
                    max_segments: 3,
                },
            },
            8 => Self {
                stage: 8,
                name: "multi_path_3x4".into(),
                mode: GenerationMode::MultiPath {
                    min_paths: 2,
                    max_paths: 3,
                    min_segments: 2,
                    max_segments: 4,
                },
            },
            9 => Self {
                stage: 9,
                name: "multi_path_4x5".into(),
                mode: GenerationMode::MultiPath {
                    min_paths: 2,
                    max_paths: 4,
                    min_segments: 2,
                    max_segments: 5,
                },
            },
            // 默认：最高阶段
            _ => Self::from_stage(9),
        }
    }

    /// 获取此阶段可能产生的最大笔画数
    pub fn max_strokes(&self) -> usize {
        match &self.mode {
            GenerationMode::SingleStroke => 1,
            GenerationMode::IndependentStrokes { max_count, .. } => *max_count,
            GenerationMode::ContinuousPath { max_segments, .. } => *max_segments,
            GenerationMode::MultiPath {
                max_paths,
                max_segments,
                ..
            } => max_paths * max_segments,
        }
    }
}

/// 单个样本的生成结果
pub struct GeneratedSample {
    pub strokes: Vec<CubicBezier>,
    pub maps: DenseMaps,
}

/// 数据生成器
pub struct DataGenerator {
    rasterizer: Rasterizer,
}

impl DataGenerator {
    pub fn new() -> Self {
        Self {
            rasterizer: Rasterizer::new(),
        }
    }

    /// 根据配置生成笔画
    pub fn generate_strokes(
        &self,
        config: &CurriculumConfig,
        size: usize,
        rng: &mut rand::rngs::StdRng,
    ) -> Vec<CubicBezier> {
        match &config.mode {
            GenerationMode::SingleStroke => {
                vec![generate_single_stroke(size, rng)]
            }
            GenerationMode::IndependentStrokes {
                min_count,
                max_count,
            } => {
                let count = rng.gen_range(*min_count..=*max_count);
                generate_independent_strokes(size, count, rng)
            }
            GenerationMode::ContinuousPath {
                min_segments,
                max_segments,
            } => {
                let num_segs = rng.gen_range(*min_segments..=*max_segments);
                generate_continuous_path(size, num_segs, rng)
            }
            GenerationMode::MultiPath {
                min_paths,
                max_paths,
                min_segments,
                max_segments,
            } => {
                let num_paths = rng.gen_range(*min_paths..=*max_paths);
                let min_s = *min_segments;
                let max_s = *max_segments;
                generate_multi_path(size, num_paths, |rng| rng.gen_range(min_s..=max_s), rng)
            }
        }
    }

    /// 生成单个样本（笔画 + Dense Maps）
    pub fn generate_sample(
        &self,
        config: &CurriculumConfig,
        size: usize,
        rng: &mut rand::rngs::StdRng,
    ) -> GeneratedSample {
        let strokes = self.generate_strokes(config, size, rng);
        let maps = self.rasterizer.render(&strokes, size);
        GeneratedSample { strokes, maps }
    }

    /// 并行生成一批样本
    ///
    /// 每个并行任务都会使用 StdRng::from_entropy() 创建新的 RNG，
    /// 确保即使在 fork() 之后也能获得独立的随机序列。
    pub fn generate_batch(
        &self,
        config: &CurriculumConfig,
        batch_size: usize,
        img_size: usize,
    ) -> Vec<GeneratedSample> {
        (0..batch_size)
            .into_par_iter()
            .map(|_| {
                // 使用 from_entropy() 而不是 thread_rng()
                // 确保每个并行任务获得独立的熵源
                let mut rng = new_rng();
                self.generate_sample(config, img_size, &mut rng)
            })
            .collect()
    }
}

impl Default for DataGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// 批量输出结构（用于 Python 接口）
pub struct BatchOutput {
    pub batch_size: usize,
    pub img_size: usize,
    pub images: Vec<f32>,    // [B, H, W]
    pub skeletons: Vec<f32>, // [B, H, W]
    pub keypoints: Vec<f32>, // [B, 2, H, W] - 改为 keypoints
    pub tangents: Vec<f32>,  // [B, 2, H, W]
    pub widths: Vec<f32>,    // [B, H, W]
    pub offsets: Vec<f32>,   // [B, 2, H, W]
}

impl BatchOutput {
    /// 从生成的样本列表组装批量输出
    pub fn from_samples(samples: Vec<GeneratedSample>, img_size: usize) -> Self {
        let batch_size = samples.len();
        let hw = img_size * img_size;

        let mut images = Vec::with_capacity(batch_size * hw);
        let mut skeletons = Vec::with_capacity(batch_size * hw);
        let mut keypoints = Vec::with_capacity(batch_size * 2 * hw);
        let mut tangents = Vec::with_capacity(batch_size * 2 * hw);
        let mut widths = Vec::with_capacity(batch_size * hw);
        let mut offsets = Vec::with_capacity(batch_size * 2 * hw);

        for sample in samples {
            let maps = sample.maps;
            images.extend(maps.image);
            skeletons.extend(maps.skeleton);
            keypoints.extend(maps.keypoints); // 使用 keypoints
            tangents.extend(maps.tangent);
            widths.extend(maps.width);
            offsets.extend(maps.offset);
        }

        Self {
            batch_size,
            img_size,
            images,
            skeletons,
            keypoints,
            tangents,
            widths,
            offsets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curriculum_stages() {
        let generator = DataGenerator::new();

        // 测试每个阶段
        for stage in 0..=9 {
            let config = CurriculumConfig::from_stage(stage);
            let mut rng = new_rng();
            let sample = generator.generate_sample(&config, 64, &mut rng);

            assert!(
                !sample.strokes.is_empty(),
                "Stage {} should generate strokes",
                stage
            );
            assert!(
                sample.strokes.len() <= config.max_strokes(),
                "Stage {} generated too many strokes",
                stage
            );

            // 检查 maps 不为空
            let skel_sum: f32 = sample.maps.skeleton.iter().sum();
            assert!(skel_sum > 0.0, "Stage {} skeleton is empty", stage);

            // 检查 keypoints 是 2 通道
            assert_eq!(
                sample.maps.keypoints.len(),
                2 * 64 * 64,
                "Stage {} keypoints should be 2 channels",
                stage
            );
        }
    }

    #[test]
    fn test_batch_generation() {
        let generator = DataGenerator::new();
        let config = CurriculumConfig::from_stage(2);

        let samples = generator.generate_batch(&config, 4, 64);
        assert_eq!(samples.len(), 4);

        let output = BatchOutput::from_samples(samples, 64);
        assert_eq!(output.batch_size, 4);
        assert_eq!(output.images.len(), 4 * 64 * 64);
        assert_eq!(output.keypoints.len(), 4 * 2 * 64 * 64); // 2 通道
    }

    #[test]
    fn test_parallel_randomness() {
        // 测试并行生成的样本是否具有不同的随机性
        let generator = DataGenerator::new();
        let config = CurriculumConfig::from_stage(0);

        let samples = generator.generate_batch(&config, 10, 64);

        // 检查前几个样本的起点坐标是否不同
        // 将 f32 转换为整数以便使用 HashSet
        let p0_coords: Vec<(i32, i32)> = samples
            .iter()
            .map(|s| {
                (
                    (s.strokes[0].p0.x * 1000.0) as i32,
                    (s.strokes[0].p0.y * 1000.0) as i32,
                )
            })
            .collect();

        // 应该有多个不同的坐标（概率极高）
        let unique_count = p0_coords
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            unique_count > 1,
            "Parallel samples should have different random values"
        );
    }
}
