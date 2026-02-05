//! 几何基础模块
//! 
//! 提供 Point, CubicBezier 等基础几何类型和操作

/// 2D 点
#[derive(Clone, Debug, Default)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    #[inline]
    pub fn norm(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    #[inline]
    pub fn norm_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n > 1e-6 {
            Self {
                x: self.x / n,
                y: self.y / n,
            }
        } else {
            Self::zero()
        }
    }

    /// 旋转点（绕原点）
    #[inline]
    pub fn rotate(&self, angle: f32) -> Self {
        let (sin_a, cos_a) = angle.sin_cos();
        Self {
            x: self.x * cos_a - self.y * sin_a,
            y: self.x * sin_a + self.y * cos_a,
        }
    }

    /// 限制点在 [min, max] 范围内
    #[inline]
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Self {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
        }
    }

    /// 从角度创建单位向量
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        Self {
            x: angle.cos(),
            y: angle.sin(),
        }
    }

    /// 点乘
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// 两点距离
    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// 线性插值
    #[inline]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

// 运算符重载
impl std::ops::Add for Point {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Add for &Point {
    type Output = Point;
    #[inline]
    fn add(self, other: Self) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub for Point {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Sub for &Point {
    type Output = Point;
    #[inline]
    fn sub(self, other: Self) -> Point {
        Point {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Mul<f32> for Point {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl std::ops::Mul<f32> for &Point {
    type Output = Point;
    #[inline]
    fn mul(self, scalar: f32) -> Point {
        Point {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl std::ops::Neg for Point {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

/// 三次贝塞尔曲线
#[derive(Clone, Debug)]
pub struct CubicBezier {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
    pub w_start: f32,
    pub w_end: f32,
}

impl CubicBezier {
    pub fn new(p0: Point, p1: Point, p2: Point, p3: Point, w_start: f32, w_end: f32) -> Self {
        Self { p0, p1, p2, p3, w_start, w_end }
    }

    /// 评估曲线上 t 处的点 (t in [0, 1])
    #[inline]
    pub fn eval(&self, t: f32) -> Point {
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        Point {
            x: mt3 * self.p0.x + 3.0 * mt2 * t * self.p1.x 
               + 3.0 * mt * t2 * self.p2.x + t3 * self.p3.x,
            y: mt3 * self.p0.y + 3.0 * mt2 * t * self.p1.y 
               + 3.0 * mt * t2 * self.p2.y + t3 * self.p3.y,
        }
    }

    /// 评估曲线上 t 处的一阶导数（切向量，未归一化）
    #[inline]
    pub fn eval_derivative(&self, t: f32) -> Point {
        let t2 = t * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;

        // B'(t) = 3(1-t)²(P1-P0) + 6(1-t)t(P2-P1) + 3t²(P3-P2)
        let v10 = &self.p1 - &self.p0;
        let v21 = &self.p2 - &self.p1;
        let v32 = &self.p3 - &self.p2;

        Point {
            x: 3.0 * mt2 * v10.x + 6.0 * mt * t * v21.x + 3.0 * t2 * v32.x,
            y: 3.0 * mt2 * v10.y + 6.0 * mt * t * v21.y + 3.0 * t2 * v32.y,
        }
    }

    /// 评估曲线上 t 处的归一化切向量
    #[inline]
    pub fn eval_tangent(&self, t: f32) -> Point {
        self.eval_derivative(t).normalize()
    }

    /// 评估曲线上 t 处的宽度
    #[inline]
    pub fn eval_width(&self, t: f32) -> f32 {
        self.w_start + (self.w_end - self.w_start) * t
    }

    /// 估算曲线长度（弦长 + 控制多边形长度的平均）
    pub fn estimate_length(&self) -> f32 {
        let chord = self.p0.distance(&self.p3);
        let control_poly = self.p0.distance(&self.p1) 
            + self.p1.distance(&self.p2) 
            + self.p2.distance(&self.p3);
        (chord + control_poly) / 2.0
    }

    /// 获取结束点的切向量（用于 G1 连续性）
    #[inline]
    pub fn end_tangent(&self) -> Point {
        (&self.p3 - &self.p2).normalize()
    }

    /// 获取起始点的切向量
    #[inline]
    pub fn start_tangent(&self) -> Point {
        (&self.p1 - &self.p0).normalize()
    }

    /// 将所有控制点限制在画布范围内
    pub fn clamp_to_canvas(&mut self, size: f32) {
        self.p0 = self.p0.clamp(0.0, size);
        self.p1 = self.p1.clamp(0.0, size);
        self.p2 = self.p2.clamp(0.0, size);
        self.p3 = self.p3.clamp(0.0, size);
    }
}

/// 计算双倍角表示 (cos(2θ), sin(2θ))
/// 用于消除切向量的 180° 歧义性
#[inline]
pub fn double_angle_representation(tangent: &Point) -> (f32, f32) {
    let ux = tangent.x;
    let uy = tangent.y;
    // cos(2θ) = cos²θ - sin²θ = ux² - uy²
    // sin(2θ) = 2·cosθ·sinθ = 2·ux·uy
    (ux * ux - uy * uy, 2.0 * ux * uy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_operations() {
        let p1 = Point::new(1.0, 2.0);
        let p2 = Point::new(3.0, 4.0);
        
        let sum = p1.clone() + p2.clone();
        assert!((sum.x - 4.0).abs() < 1e-6);
        assert!((sum.y - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_bezier_eval() {
        // 直线段测试
        let bezier = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(0.33, 0.0),
            Point::new(0.66, 0.0),
            Point::new(1.0, 0.0),
            1.0, 1.0
        );
        
        let mid = bezier.eval(0.5);
        assert!((mid.x - 0.5).abs() < 0.1);
        assert!((mid.y - 0.0).abs() < 0.1);
    }
}
