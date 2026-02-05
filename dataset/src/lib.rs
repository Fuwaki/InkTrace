//! InkTrace Rust 数据生成库
//!
//! 为 Dense Prediction 训练提供高性能数据生成:
//! - 直接输出 Dense GT Maps (skeleton, keypoints, tangent, width, offset)
//! - 支持分层训练 (Curriculum Learning)
//! - 多线程并行生成
//!
//! Keypoints 输出格式 [B, 2, H, W]:
//! - Channel 0: 拓扑节点 - 样条曲线的起点/终点（必须断开）
//! - Channel 1: 几何锚点 - 多段贝塞尔曲线之间的连接点（建议断开以改善拟合）

mod generator;
mod geometry;
mod rasterize;
mod stroke;

#[cfg(test)]
mod tests;

use generator::{BatchOutput, CurriculumConfig, DataGenerator, GenerationMode};
use numpy::ndarray::{Array3, Array4};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// =============================================================================
// PyO3 接口
// =============================================================================

/// 生成一批 Dense Training 数据
///
/// Args:
///     batch_size: 批量大小
///     img_size: 图像尺寸 (64, 128, etc.)
///     stage: 训练阶段 (0-9)
///         - 0: 单笔画
///         - 1-3: 多独立笔画 (递增)
///         - 4-6: 多段连续笔画 (递增)
///         - 7-9: 混合模式 (多条多段路径)
///
/// Returns:
///     dict 包含:
///         - 'image': [B, H, W] 渲染图像
///         - 'skeleton': [B, H, W] 骨架图
///         - 'keypoints': [B, 2, H, W] 关键点图
///             - Ch0: 拓扑节点 - 样条曲线的起点/终点
///             - Ch1: 几何锚点 - 多段贝塞尔曲线之间的连接点
///         - 'tangent': [B, 2, H, W] 切向场 (cos2θ, sin2θ)
///         - 'width': [B, H, W] 宽度图 (归一化)
///         - 'offset': [B, 2, H, W] 亚像素偏移
#[pyfunction]
#[pyo3(signature = (batch_size, img_size, stage=0))]
fn generate_dense_batch<'py>(
    py: Python<'py>,
    batch_size: usize,
    img_size: usize,
    stage: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let config = CurriculumConfig::from_stage(stage);
    let generator = DataGenerator::new();

    // 并行生成
    let samples = generator.generate_batch(&config, batch_size, img_size);
    let output = BatchOutput::from_samples(samples, img_size);

    // 转换为 numpy 数组
    let images = Array3::from_shape_vec((batch_size, img_size, img_size), output.images)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let skeletons = Array3::from_shape_vec((batch_size, img_size, img_size), output.skeletons)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // keypoints 是 [B, 2, H, W]
    let keypoints = Array4::from_shape_vec((batch_size, 2, img_size, img_size), output.keypoints)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let tangents = Array4::from_shape_vec((batch_size, 2, img_size, img_size), output.tangents)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let widths = Array3::from_shape_vec((batch_size, img_size, img_size), output.widths)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let offsets = Array4::from_shape_vec((batch_size, 2, img_size, img_size), output.offsets)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // 创建返回字典
    let dict = PyDict::new(py);
    dict.set_item("image", images.into_pyarray(py))?;
    dict.set_item("skeleton", skeletons.into_pyarray(py))?;
    dict.set_item("keypoints", keypoints.into_pyarray(py))?;
    dict.set_item("tangent", tangents.into_pyarray(py))?;
    dict.set_item("width", widths.into_pyarray(py))?;
    dict.set_item("offset", offsets.into_pyarray(py))?;

    Ok(dict)
}

/// 获取指定阶段的配置信息
///
/// Args:
///     stage: 训练阶段 (0-9)
///
/// Returns:
///     dict 包含:
///         - 'stage': 阶段编号
///         - 'name': 阶段名称
///         - 'mode': 生成模式描述
///         - 'max_strokes': 最大笔画数
#[pyfunction]
fn get_stage_info(py: Python<'_>, stage: u32) -> PyResult<Bound<'_, PyDict>> {
    let config = CurriculumConfig::from_stage(stage);

    let mode_str = match &config.mode {
        GenerationMode::SingleStroke => "single".to_string(),
        GenerationMode::IndependentStrokes {
            min_count,
            max_count,
        } => {
            format!("independent({}-{})", min_count, max_count)
        }
        GenerationMode::ContinuousPath {
            min_segments,
            max_segments,
        } => format!("continuous({}-{})", min_segments, max_segments),
        GenerationMode::MultiPath {
            min_paths,
            max_paths,
            min_segments,
            max_segments,
        } => format!(
            "multi_path(paths:{}-{}, segs:{}-{})",
            min_paths, max_paths, min_segments, max_segments
        ),
    };

    let dict = PyDict::new(py);
    dict.set_item("stage", config.stage)?;
    dict.set_item("name", &config.name)?;
    dict.set_item("mode", mode_str)?;
    dict.set_item("max_strokes", config.max_strokes())?;

    Ok(dict)
}

/// 列出所有可用的训练阶段
#[pyfunction]
fn list_stages(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    let mut stages = Vec::new();
    for s in 0..=9 {
        stages.push(get_stage_info(py, s)?);
    }
    Ok(stages)
}

/// 配置 Rayon 线程池
///
/// 应在程序启动时调用一次。
/// 如果多次调用会返回错误（Rayon 的限制）。
#[pyfunction]
fn set_rayon_threads(num_threads: usize) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

// =============================================================================
// 模块注册
// =============================================================================

#[pymodule]
fn ink_trace_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Dense GT 生成 API
    m.add_function(wrap_pyfunction!(generate_dense_batch, m)?)?;
    m.add_function(wrap_pyfunction!(get_stage_info, m)?)?;
    m.add_function(wrap_pyfunction!(list_stages, m)?)?;

    // 工具函数
    m.add_function(wrap_pyfunction!(set_rayon_threads, m)?)?;

    Ok(())
}
