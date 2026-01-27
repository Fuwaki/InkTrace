use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4};
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::f32::consts::PI;
use numpy::ndarray::{Array2, Array3, Array4, ShapeError};

// -----------------------------------------------------------------------------
// Core Data Structures
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl std::ops::Add for Point {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub for Point {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Mul<f32> for Point {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl Point {
    fn norm(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn normalize(&self) -> Self {
        let n = self.norm();
        if n > 1e-6 {
            Self {
                x: self.x / n,
                y: self.y / n,
            }
        } else {
            Self { x: 0.0, y: 0.0 }
        }
    }

    fn rotate(&self, angle: f32) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Self {
            x: self.x * cos_a - self.y * sin_a,
            y: self.x * sin_a + self.y * cos_a,
        }
    }

    fn clamp(&self, min: f32, max: f32) -> Self {
        Self {
            x: self.x.max(min).min(max),
            y: self.y.max(min).min(max),
        }
    }
}

#[derive(Clone, Debug)]
struct CubicBezier {
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    w_start: f32,
    w_end: f32,
}

impl CubicBezier {
    fn eval(&self, t: f32) -> Point {
        let one_minus_t = 1.0 - t;
        let t2 = t * t;
        let one_minus_t2 = one_minus_t * one_minus_t;
        let one_minus_t3 = one_minus_t2 * one_minus_t;
        let t3 = t2 * t;

        self.p0.clone() * one_minus_t3
            + self.p1.clone() * (3.0 * one_minus_t2 * t)
            + self.p2.clone() * (3.0 * one_minus_t * t2)
            + self.p3.clone() * t3
    }

    fn eval_width(&self, t: f32) -> f32 {
        self.w_start + (self.w_end - self.w_start) * t
    }
}

// -----------------------------------------------------------------------------
// Rendering Logic
// -----------------------------------------------------------------------------

fn render_strokes_to_canvas(strokes: &[CubicBezier], size: usize) -> Vec<f32> {
    let scale = 2.0; // supersampling
    let canvas_size = (size as f32 * scale) as usize;
    let mut canvas = vec![0.0f32; canvas_size * canvas_size];

    for stroke in strokes {
        // Dynamic step count based on estimated length roughly
        // Or just fixed high number as in Python
        let num_steps = 200; 
        
        for i in 0..num_steps {
            let t = i as f32 / (num_steps - 1) as f32;
            let pt = stroke.eval(t);
            let w = stroke.eval_width(t);
            let radius = (w * scale / 2.0).max(0.5);
            let r2 = radius * radius;

            let center_x = pt.x * scale;
            let center_y = pt.y * scale;

            // Bounding box for the circle
            let min_x = (center_x - radius).floor().max(0.0) as isize;
            let max_x = (center_x + radius).ceil().min(canvas_size as f32 - 1.0) as isize;
            let min_y = (center_y - radius).floor().max(0.0) as isize;
            let max_y = (center_y + radius).ceil().min(canvas_size as f32 - 1.0) as isize;

            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let dx = x as f32 - center_x;
                    let dy = y as f32 - center_y;
                    if dx * dx + dy * dy <= r2 {
                        // idx = y * width + x
                        let idx = y as usize * canvas_size + x as usize;
                         canvas[idx] = 1.0;
                    }
                }
            }
        }
    }

    // Downsample (Average pooling / Resize)
    // simple box filter implementation for exactly 2x downsampling
    let mut final_canvas = vec![0.0f32; size * size];
    
    for y in 0..size {
        for x in 0..size {
            let src_x = x * 2;
            let src_y = y * 2;
            
            let mut sum = 0.0;
            sum += canvas[src_y * canvas_size + src_x];
            sum += canvas[src_y * canvas_size + (src_x + 1)];
            sum += canvas[(src_y + 1) * canvas_size + src_x];
            sum += canvas[(src_y + 1) * canvas_size + (src_x + 1)];
            
            final_canvas[y * size + x] = sum / 4.0;
        }
    }

    final_canvas
}

// -----------------------------------------------------------------------------
// Generation Logic Helpers
// -----------------------------------------------------------------------------

fn random_point(rng: &mut ThreadRng, size: f32) -> Point {
    Point::new(
        rng.gen::<f32>() * (size * 0.8) + (size * 0.1),
        rng.gen::<f32>() * (size * 0.8) + (size * 0.1),
    )
}

/// Generates a Cubic Bezier segment with "natural" handwriting characteristics.
fn generate_natural_segment(
    start_point: Point,
    incoming_tangent: Option<Point>,
    rng: &mut ThreadRng,
    len_range: std::ops::Range<f32>,
    _size: f32, // Canvas size for boundary check hints (unused strictly now)
) -> CubicBezier {
    // 1. Determine general direction (Chord P0 -> P3)
    let chord_dir = if let Some(tan) = incoming_tangent.clone() {
        // Continue roughly in the same direction but allow turning +/- 60 deg
        tan.rotate(rng.gen_range(-PI/3.0..PI/3.0))
    } else {
        // Random starting direction
        let angle = rng.gen_range(0.0..2.0 * PI);
        Point::new(angle.cos(), angle.sin())
    }.normalize();

    let len = rng.gen_range(len_range);
    let p3 = start_point.clone() + chord_dir.clone() * len;

    // 2. Determine Control Points P1 & P2
    // P1 Logic: Enforce G1 Continuity if incoming tangent exists
    let p1_dir = if let Some(tan) = incoming_tangent {
        tan 
    } else {
        // First stroke: small deviation
        chord_dir.rotate(rng.gen_range(-PI/4.0..PI/4.0)) 
    }.normalize();

    // Handle lengths (0.25 - 0.45 of chord length)
    let h1_len = len * rng.gen_range(0.25..0.45);
    let p1 = start_point.clone() + p1_dir.clone() * h1_len;

    // P2 Logic: Backwards from P3
    let landing_dir = chord_dir.rotate(rng.gen_range(-PI/4.0..PI/4.0)).normalize();
    let h2_len = len * rng.gen_range(0.25..0.45);
    let p2 = p3.clone() - landing_dir * h2_len;

    // Safety: Clamp all points to canvas bounds to ensure [0, 1] labels
    // This alters geometry slightly near edges but prevents invalid labels.
    CubicBezier {
        p0: start_point.clamp(0.0, _size),
        p1: p1.clamp(0.0, _size),
        p2: p2.clamp(0.0, _size),
        p3: p3.clamp(0.0, _size),
        w_start: rng.gen_range(2.0..5.0),
        w_end: rng.gen_range(2.0..5.0),
    }
}

// -----------------------------------------------------------------------------
// PyO3 Module Interface
// -----------------------------------------------------------------------------

/// Generates a batch of single stroke images.
///
/// Returns:
///     - images: [batch_size, img_size, img_size] (float32, 0.0-1.0)
///     - labels: [batch_size, 10] (float32, normalized)
///         [x0, y0, x1, y1, x2, y2, x3, y3, w_start, w_end]
#[pyfunction]
fn generate_single_stroke_batch(
    py: Python,
    batch_size: usize,
    img_size: usize,
) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray2<f32>>)> {
    
    // Parallel generation
    let results: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let size_f = img_size as f32;
            
            let p0 = random_point(&mut rng, size_f);
            let stroke = generate_natural_segment(
                p0, None, &mut rng, 
                (size_f * 0.3)..(size_f * 0.8),
                size_f
            );
            
            let img_flat = render_strokes_to_canvas(&[stroke.clone()], img_size);
            
            // Labels: [x0, y0, ..., x3, y3, w_s, w_e] normalized
            let mut label = Vec::with_capacity(10);
            label.push(stroke.p0.x / size_f); label.push(stroke.p0.y / size_f);
            label.push(stroke.p1.x / size_f); label.push(stroke.p1.y / size_f);
            label.push(stroke.p2.x / size_f); label.push(stroke.p2.y / size_f);
            label.push(stroke.p3.x / size_f); label.push(stroke.p3.y / size_f);
            label.push(stroke.w_start / 10.0);
            label.push(stroke.w_end / 10.0);

            (img_flat, label)
        })
        .collect();

    // Assemble batch
    let mut images = Vec::with_capacity(batch_size * img_size * img_size);
    let mut labels = Vec::with_capacity(batch_size * 10);

    for (img, label) in results {
        images.extend(img);
        labels.extend(label);
    }

    let img_array = Array3::from_shape_vec((batch_size, img_size, img_size), images)
        .map_err(|e: ShapeError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let label_array = Array2::from_shape_vec((batch_size, 10), labels)
        .map_err(|e: ShapeError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((
        img_array.into_pyarray_bound(py).unbind(),
        label_array.into_pyarray_bound(py).unbind()
    ))
}

/// Generates a batch of independent multi-stroke images (Set Prediction).
///
/// Returns:
///     - images: [batch_size, img_size, img_size]
///     - labels: [batch_size, max_strokes, 11]
///         [..., :8] = [x0, y0, x1, y1, x2, y2, x3, y3]
///         [..., 8]  = w_start
///         [..., 9]  = w_end
///         [..., 10] = 1.0 if valid, 0.0 if padding
#[pyfunction]
#[pyo3(signature = (batch_size, img_size, max_strokes, fixed_count=None))]
fn generate_independent_strokes_batch(
    py: Python,
    batch_size: usize,
    img_size: usize,
    max_strokes: usize,
    fixed_count: Option<usize>,
) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray3<f32>>)> {
    
    // Output label dimension: [batch, max_strokes, 11] (last is valid flag)

    let results: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
    .into_par_iter()
    .map(|_| {
        let mut rng = rand::thread_rng();
        let size_f = img_size as f32;
        
        let num_strokes = match fixed_count {
            Some(n) => n,
            None => rng.gen_range(1..=max_strokes),
        };

        // Grid-based-ish distribution: Divide canvas into cells or just random spread 
        // but avoid starting too close to others? 
        // Simple heuristic: Pure random P0, but controlled length/direction to avoid massive overlap
        
        let mut strokes = Vec::new();
        for _ in 0..num_strokes {
             let p0 = random_point(&mut rng, size_f);
             let stroke = generate_natural_segment(
                 p0, None, &mut rng, 
                 (size_f * 0.15)..(size_f * 0.4), // Keep independent strokes shorter
                 size_f
             );
             strokes.push(stroke);
        }
        
        // Sort by P0.y to stabilize (Hungarian matching easier)
        strokes.sort_by(|a, b| a.p0.y.partial_cmp(&b.p0.y).unwrap_or(std::cmp::Ordering::Equal));

        let img_flat = render_strokes_to_canvas(&strokes, img_size);

        // Prepare labels [max_strokes, 11]
        let mut label_flat = vec![0.0f32; max_strokes * 11];
        for (i, s) in strokes.iter().enumerate() {
            if i >= max_strokes { break; }
            let base = i * 11;
            label_flat[base + 0] = s.p0.x / size_f;
            label_flat[base + 1] = s.p0.y / size_f;
            label_flat[base + 2] = s.p1.x / size_f;
            label_flat[base + 3] = s.p1.y / size_f;
            label_flat[base + 4] = s.p2.x / size_f;
            label_flat[base + 5] = s.p2.y / size_f;
            label_flat[base + 6] = s.p3.x / size_f;
            label_flat[base + 7] = s.p3.y / size_f;
            label_flat[base + 8] = s.w_start / 10.0;
            label_flat[base + 9] = s.w_end / 10.0;
            label_flat[base + 10] = 1.0; // Valid
        }

        (img_flat, label_flat)
    })
    .collect();

    let mut images = Vec::with_capacity(batch_size * img_size * img_size);
    let mut labels = Vec::with_capacity(batch_size * max_strokes * 11);

    for (img, label) in results {
        images.extend(img);
        labels.extend(label);
    }

    let img_array = Array3::from_shape_vec((batch_size, img_size, img_size), images)
        .map_err(|e: ShapeError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let label_array = Array3::from_shape_vec((batch_size, max_strokes, 11), labels)
        .map_err(|e: ShapeError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((
        img_array.into_pyarray_bound(py).unbind(),
        label_array.into_pyarray_bound(py).unbind()
    ))
}

/// Generates a batch of continuous multi-segment strokes (Sequence Prediction).
/// All segments in a sequence are G1 continuous (smoothly connected).
///
/// Returns:
///     - images: [batch_size, img_size, img_size]
///     - labels: [batch_size, max_segments, 11]
///         [..., :8] = [x0, y0, x1, y1, x2, y2, x3, y3]
///         [..., 8]  = w_start
///         [..., 9]  = w_end
///         [..., 10] = 1.0 if valid, 0.0 if padding
#[pyfunction]
fn generate_continuous_strokes_batch(
    py: Python,
    batch_size: usize,
    img_size: usize,
    max_segments: usize,
) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray3<f32>>)> {
    
    // Output label dimension: [batch, max_strokes, 11]
    // But strokes are connected P3(i) == P0(i+1)

    let results: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let size_f = img_size as f32;
            
            let num_segments = rng.gen_range(1..=max_segments);
            let mut strokes = Vec::new();
            
            // Random start
            let mut current_pos = Point::new(
                rng.gen::<f32>() * size_f * 0.8 + size_f * 0.1, 
                rng.gen::<f32>() * size_f * 0.8 + size_f * 0.1
            );
            
            let mut incoming_tangent: Option<Point> = None;

            for _ in 0..num_segments {
                let stroke = generate_natural_segment(
                    current_pos.clone(), 
                    incoming_tangent, 
                    &mut rng, 
                    (size_f * 0.15)..(size_f * 0.4), // Shorter segments for continuous
                    size_f
                );
                
                // Next segment starts at P3
                current_pos = stroke.p3.clone();
                // Incoming tangent for next segment is normalized P2->P3
                incoming_tangent = Some((stroke.p3.clone() - stroke.p2.clone()).normalize());
                
                strokes.push(stroke);
            }

            let img_flat = render_strokes_to_canvas(&strokes, img_size);

             // Prepare labels
             let mut label_flat = vec![0.0f32; max_segments * 11];
             for (i, s) in strokes.iter().enumerate() {
                 if i >= max_segments { break; }
                 let base = i * 11;
                 label_flat[base + 0] = s.p0.x / size_f;
                 label_flat[base + 1] = s.p0.y / size_f;
                 label_flat[base + 2] = s.p1.x / size_f;
                 label_flat[base + 3] = s.p1.y / size_f;
                 label_flat[base + 4] = s.p2.x / size_f;
                 label_flat[base + 5] = s.p2.y / size_f;
                 label_flat[base + 6] = s.p3.x / size_f;
                 label_flat[base + 7] = s.p3.y / size_f;
                 label_flat[base + 8] = s.w_start / 10.0;
                 label_flat[base + 9] = s.w_end / 10.0;
                 label_flat[base + 10] = 1.0; 
             }
     
             (img_flat, label_flat)
        })
    .collect();

    let mut images = Vec::with_capacity(batch_size * img_size * img_size);
    let mut labels = Vec::with_capacity(batch_size * max_segments * 11);

    for (img, label) in results {
        images.extend(img);
        labels.extend(label);
    }

    let img_array = Array3::from_shape_vec((batch_size, img_size, img_size), images)
        .map_err(|e: ShapeError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let label_array = Array3::from_shape_vec((batch_size, max_segments, 11), labels)
        .map_err(|e: ShapeError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((
        img_array.into_pyarray_bound(py).unbind(),
        label_array.into_pyarray_bound(py).unbind()
    ))
}

/// Generates a batch of multiple independent continuous paths.
/// Each path consists of multiple smooth segments.
///
/// Returns:
///     - images: [batch_size, img_size, img_size]
///     - labels: [batch_size, max_paths, max_segments, 11]
///         [..., :8] = [x0, y0, x1, y1, x2, y2, x3, y3]
///         [..., 8]  = w_start
///         [..., 9]  = w_end
///         [..., 10] = 1.0 if valid, 0.0 if padding
#[pyfunction]
fn generate_multi_connected_strokes_batch(
    py: Python,
    batch_size: usize,
    img_size: usize,
    max_paths: usize,
    max_segments: usize,
) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray4<f32>>)> {
    
    // Output label dimension: [batch, max_paths, max_segments, 11]
    
    let results: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
    .into_par_iter()
    .map(|_| {
        let mut rng = rand::thread_rng();
        let size_f = img_size as f32;
        
        // Random number of paths
        let num_paths = rng.gen_range(1..=max_paths);
        
        let mut all_strokes_flat = Vec::new();
        // Labels for this sample: max_paths * max_segments * 11
        let mut label_flat = vec![0.0f32; max_paths * max_segments * 11];

        for path_idx in 0..num_paths {
            let num_segments = rng.gen_range(1..=max_segments);

            // Random start for this path
            let mut current_pos = Point::new(
                rng.gen::<f32>() * size_f * 0.8 + size_f * 0.1, 
                rng.gen::<f32>() * size_f * 0.8 + size_f * 0.1
            );
            
            let mut incoming_tangent: Option<Point> = None;

            for seg_idx in 0..num_segments {
                 let stroke = generate_natural_segment(
                     current_pos.clone(), 
                     incoming_tangent, 
                     &mut rng, 
                     (size_f * 0.15)..(size_f * 0.35),
                     size_f
                 );
                 
                 // Update state for next segment
                 current_pos = stroke.p3.clone();
                 incoming_tangent = Some((stroke.p3.clone() - stroke.p2.clone()).normalize());
                 
                 // Fill label
                 // Index: path_idx * (max_segments * 11) + seg_idx * 11
                 let base = path_idx * (max_segments * 11) + seg_idx * 11;
                 label_flat[base + 0] = stroke.p0.x / size_f;
                 label_flat[base + 1] = stroke.p0.y / size_f;
                 label_flat[base + 2] = stroke.p1.x / size_f;
                 label_flat[base + 3] = stroke.p1.y / size_f;
                 label_flat[base + 4] = stroke.p2.x / size_f;
                 label_flat[base + 5] = stroke.p2.y / size_f;
                 label_flat[base + 6] = stroke.p3.x / size_f;
                 label_flat[base + 7] = stroke.p3.y / size_f;
                 label_flat[base + 8] = stroke.w_start / 10.0;
                 label_flat[base + 9] = stroke.w_end / 10.0;
                 label_flat[base + 10] = 1.0; // Valid

                 all_strokes_flat.push(stroke);
            }
        }

        let img_flat = render_strokes_to_canvas(&all_strokes_flat, img_size);
        (img_flat, label_flat)
    })
    .collect();

    let mut images = Vec::with_capacity(batch_size * img_size * img_size);
    let mut labels = Vec::with_capacity(batch_size * max_paths * max_segments * 11);

    for (img, label) in results {
        images.extend(img);
        labels.extend(label);
    }

    let img_array = Array3::from_shape_vec((batch_size, img_size, img_size), images)
        .map_err(|e: ShapeError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let label_array = Array4::from_shape_vec((batch_size, max_paths, max_segments, 11), labels)
        .map_err(|e: ShapeError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((
        img_array.into_pyarray_bound(py).unbind(),
        label_array.into_pyarray_bound(py).unbind()
    ))
}

/// Configures the global thread pool for Rayon.
/// Should be called once at startup or inside worker initialization.
/// If called more than once, it will return an error (Rayon limitation), 
/// which we silence here to act idempotent-ish or just fail loudly if desired.
/// Here we just expose the simple wrapper.
#[pyfunction]
fn set_rayon_threads(num_threads: usize) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pymodule]
fn ink_trace_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_single_stroke_batch, m)?)?;
    m.add_function(wrap_pyfunction!(generate_independent_strokes_batch, m)?)?;
    m.add_function(wrap_pyfunction!(generate_continuous_strokes_batch, m)?)?;
    m.add_function(wrap_pyfunction!(generate_multi_connected_strokes_batch, m)?)?;
    m.add_function(wrap_pyfunction!(set_rayon_threads, m)?)?;
    Ok(())
}
