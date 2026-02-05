//! 综合测试模块
//!
//! 验证生成的 Ground Truth 数据的正确性和一致性

#[cfg(test)]
mod dense_maps_tests {
    use crate::generator::{CurriculumConfig, DataGenerator};
    use crate::geometry::{CubicBezier, Point};
    use crate::rasterize::{DenseMaps, Rasterizer};
    use crate::stroke::new_rng;

    // =========================================================================
    // DenseMaps 形状和数值范围测试
    // =========================================================================

    #[test]
    fn test_dense_maps_shapes() {
        let size = 64;
        let maps = DenseMaps::new(size);

        assert_eq!(maps.image.len(), size * size, "image should be [H, W]");
        assert_eq!(maps.skeleton.len(), size * size, "skeleton should be [H, W]");
        assert_eq!(
            maps.keypoints.len(),
            2 * size * size,
            "keypoints should be [2, H, W]"
        );
        assert_eq!(
            maps.tangent.len(),
            2 * size * size,
            "tangent should be [2, H, W]"
        );
        assert_eq!(maps.width.len(), size * size, "width should be [H, W]");
        assert_eq!(
            maps.offset.len(),
            2 * size * size,
            "offset should be [2, H, W]"
        );
    }

    #[test]
    fn test_skeleton_binary() {
        let stroke = CubicBezier::new(
            Point::new(10.0, 32.0),
            Point::new(25.0, 20.0),
            Point::new(39.0, 44.0),
            Point::new(54.0, 32.0),
            3.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);

        // skeleton 应该只有 0 和 1
        for &val in &maps.skeleton {
            assert!(
                val == 0.0 || val == 1.0,
                "Skeleton should be binary, got {}",
                val
            );
        }
    }

    #[test]
    fn test_image_range() {
        let stroke = CubicBezier::new(
            Point::new(10.0, 32.0),
            Point::new(25.0, 20.0),
            Point::new(39.0, 44.0),
            Point::new(54.0, 32.0),
            3.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);

        // image 应该在 [0, 1] 范围内
        for &val in &maps.image {
            assert!(
                val >= 0.0 && val <= 1.0,
                "Image should be in [0, 1], got {}",
                val
            );
        }
    }

    #[test]
    fn test_keypoints_binary() {
        let stroke = CubicBezier::new(
            Point::new(10.0, 32.0),
            Point::new(25.0, 20.0),
            Point::new(39.0, 44.0),
            Point::new(54.0, 32.0),
            3.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);

        // keypoints 应该只有 0 和 1
        for &val in &maps.keypoints {
            assert!(
                val == 0.0 || val == 1.0,
                "Keypoints should be binary, got {}",
                val
            );
        }
    }

    #[test]
    fn test_tangent_unit_magnitude() {
        let stroke = CubicBezier::new(
            Point::new(10.0, 32.0),
            Point::new(25.0, 20.0),
            Point::new(39.0, 44.0),
            Point::new(54.0, 32.0),
            3.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);
        let size = 64;

        // 检查骨架上的切向量模长应该为 1
        for y in 0..size {
            for x in 0..size {
                let idx = y * size + x;
                if maps.skeleton[idx] > 0.5 {
                    let cos2t = maps.tangent[0 * size * size + idx];
                    let sin2t = maps.tangent[1 * size * size + idx];
                    let mag = (cos2t * cos2t + sin2t * sin2t).sqrt();

                    assert!(
                        (mag - 1.0).abs() < 0.01,
                        "Tangent magnitude should be ~1, got {} at ({}, {})",
                        mag,
                        x,
                        y
                    );
                }
            }
        }
    }

    #[test]
    fn test_width_positive() {
        let stroke = CubicBezier::new(
            Point::new(10.0, 32.0),
            Point::new(25.0, 20.0),
            Point::new(39.0, 44.0),
            Point::new(54.0, 32.0),
            3.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);
        let size = 64;

        // 骨架上的宽度应该为正
        for y in 0..size {
            for x in 0..size {
                let idx = y * size + x;
                if maps.skeleton[idx] > 0.5 {
                    assert!(
                        maps.width[idx] > 0.0,
                        "Width should be positive on skeleton, got {} at ({}, {})",
                        maps.width[idx],
                        x,
                        y
                    );
                }
            }
        }
    }

    #[test]
    fn test_offset_range() {
        let stroke = CubicBezier::new(
            Point::new(10.0, 32.0),
            Point::new(25.0, 20.0),
            Point::new(39.0, 44.0),
            Point::new(54.0, 32.0),
            3.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);
        let size = 64;

        // offset 应该在 [-0.5, 0.5] 范围内（亚像素偏移）
        for y in 0..size {
            for x in 0..size {
                let idx = y * size + x;
                if maps.skeleton[idx] > 0.5 {
                    let off_x = maps.offset[0 * size * size + idx];
                    let off_y = maps.offset[1 * size * size + idx];

                    assert!(
                        off_x >= -0.5 && off_x <= 0.5,
                        "Offset X should be in [-0.5, 0.5], got {} at ({}, {})",
                        off_x,
                        x,
                        y
                    );
                    assert!(
                        off_y >= -0.5 && off_y <= 0.5,
                        "Offset Y should be in [-0.5, 0.5], got {} at ({}, {})",
                        off_y,
                        x,
                        y
                    );
                }
            }
        }
    }

    // =========================================================================
    // Keypoints 逻辑正确性测试
    // =========================================================================

    #[test]
    fn test_single_stroke_keypoints() {
        // 单笔画：应该有 2 个拓扑节点，0 个几何锚点
        let stroke = CubicBezier::new(
            Point::new(10.0, 10.0),
            Point::new(20.0, 30.0),
            Point::new(40.0, 30.0),
            Point::new(50.0, 50.0),
            3.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);
        let size = 64;
        let hw = size * size;

        // 统计 keypoints
        let topo_count: f32 = maps.keypoints[..hw].iter().sum();
        let geom_count: f32 = maps.keypoints[hw..].iter().sum();

        // 单笔画应该有 1-2 个拓扑节点（可能重叠）
        assert!(
            topo_count >= 1.0 && topo_count <= 2.0,
            "Single stroke should have 1-2 topological keypoints, got {}",
            topo_count
        );

        // 单笔画应该有 0 个几何锚点
        assert_eq!(
            geom_count, 0.0,
            "Single stroke should have 0 geometric keypoints, got {}",
            geom_count
        );
    }

    #[test]
    fn test_continuous_path_keypoints() {
        // 连续路径：3 段贝塞尔曲线
        // 应该有 2 个拓扑节点（起点和终点），2 个几何锚点（中间连接点）
        let strokes = vec![
            CubicBezier::new(
                Point::new(10.0, 10.0),
                Point::new(15.0, 20.0),
                Point::new(20.0, 20.0),
                Point::new(25.0, 10.0), // P3
                2.0,
                2.0,
            ),
            CubicBezier::new(
                Point::new(25.0, 10.0), // P0 = 上一个的 P3
                Point::new(30.0, 5.0),
                Point::new(35.0, 5.0),
                Point::new(40.0, 15.0), // P3
                2.0,
                2.0,
            ),
            CubicBezier::new(
                Point::new(40.0, 15.0), // P0 = 上一个的 P3
                Point::new(45.0, 25.0),
                Point::new(50.0, 25.0),
                Point::new(55.0, 20.0), // P3 - 终点
                2.0,
                2.0,
            ),
        ];

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&strokes, 64);
        let size = 64;
        let hw = size * size;

        let topo_count: f32 = maps.keypoints[..hw].iter().sum();
        let geom_count: f32 = maps.keypoints[hw..].iter().sum();

        // 连续路径应该有 2 个拓扑节点
        assert!(
            topo_count >= 1.0 && topo_count <= 2.0,
            "Continuous path should have 1-2 topological keypoints, got {}",
            topo_count
        );

        // 连续路径应该有 2 个几何锚点
        assert!(
            geom_count >= 1.0 && geom_count <= 2.0,
            "Continuous path (3 segments) should have 1-2 geometric keypoints, got {}",
            geom_count
        );
    }

    #[test]
    fn test_independent_strokes_keypoints() {
        // 多条独立笔画：3 条独立的笔画
        // 应该有 6 个拓扑节点（每条 2 个），0 个几何锚点
        let strokes = vec![
            CubicBezier::new(
                Point::new(5.0, 10.0),
                Point::new(10.0, 20.0),
                Point::new(15.0, 20.0),
                Point::new(20.0, 10.0),
                2.0,
                2.0,
            ),
            CubicBezier::new(
                Point::new(25.0, 30.0), // 不连续，新起点
                Point::new(30.0, 40.0),
                Point::new(35.0, 40.0),
                Point::new(40.0, 30.0),
                2.0,
                2.0,
            ),
            CubicBezier::new(
                Point::new(45.0, 50.0), // 不连续，新起点
                Point::new(50.0, 55.0),
                Point::new(55.0, 55.0),
                Point::new(60.0, 50.0),
                2.0,
                2.0,
            ),
        ];

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&strokes, 64);
        let size = 64;
        let hw = size * size;

        let topo_count: f32 = maps.keypoints[..hw].iter().sum();
        let geom_count: f32 = maps.keypoints[hw..].iter().sum();

        // 3 条独立笔画应该有 4-6 个拓扑节点（可能有些端点重叠在同一像素）
        assert!(
            topo_count >= 4.0 && topo_count <= 6.0,
            "3 independent strokes should have 4-6 topological keypoints, got {}",
            topo_count
        );

        // 独立笔画应该有 0 个几何锚点
        assert_eq!(
            geom_count, 0.0,
            "Independent strokes should have 0 geometric keypoints, got {}",
            geom_count
        );
    }

    #[test]
    fn test_keypoints_on_skeleton() {
        // keypoints 应该位于骨架上（或至少在骨架附近）
        let stroke = CubicBezier::new(
            Point::new(10.0, 32.0),
            Point::new(25.0, 20.0),
            Point::new(39.0, 44.0),
            Point::new(54.0, 32.0),
            3.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);
        let size = 64;

        // 检查每个拓扑节点是否在骨架上或附近
        for y in 0..size {
            for x in 0..size {
                let idx = y * size + x;
                if maps.keypoints[idx] > 0.5 {
                    // 检查此点或相邻点是否在骨架上
                    let mut on_or_near_skeleton = maps.skeleton[idx] > 0.5;

                    // 检查 8 邻域
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 {
                                let nidx = ny as usize * size + nx as usize;
                                if maps.skeleton[nidx] > 0.5 {
                                    on_or_near_skeleton = true;
                                }
                            }
                        }
                    }

                    assert!(
                        on_or_near_skeleton,
                        "Topological keypoint at ({}, {}) should be on or near skeleton",
                        x,
                        y
                    );
                }
            }
        }
    }

    // =========================================================================
    // Curriculum 阶段测试
    // =========================================================================

    #[test]
    fn test_stage0_single_stroke() {
        let generator = DataGenerator::new();
        let config = CurriculumConfig::from_stage(0);
        let mut rng = new_rng();

        for _ in 0..10 {
            let sample = generator.generate_sample(&config, 64, &mut rng);

            // Stage 0 应该只有 1 条笔画
            assert_eq!(
                sample.strokes.len(),
                1,
                "Stage 0 should generate exactly 1 stroke"
            );

            let hw = 64 * 64;
            let topo_count: f32 = sample.maps.keypoints[..hw].iter().sum();
            let geom_count: f32 = sample.maps.keypoints[hw..].iter().sum();

            // 单笔画应该有 1-2 个拓扑节点
            assert!(
                topo_count >= 1.0 && topo_count <= 2.0,
                "Stage 0 should have 1-2 topological keypoints, got {}",
                topo_count
            );

            // 单笔画应该没有几何锚点
            assert_eq!(
                geom_count, 0.0,
                "Stage 0 should have 0 geometric keypoints, got {}",
                geom_count
            );
        }
    }

    #[test]
    fn test_stage4_continuous_path() {
        let generator = DataGenerator::new();
        let config = CurriculumConfig::from_stage(4); // continuous 2-3 segments
        let mut rng = new_rng();

        let mut total_geom = 0.0f32;
        let num_samples = 20;

        for _ in 0..num_samples {
            let sample = generator.generate_sample(&config, 64, &mut rng);

            // Stage 4 应该有 2-3 段
            assert!(
                sample.strokes.len() >= 2 && sample.strokes.len() <= 3,
                "Stage 4 should generate 2-3 strokes, got {}",
                sample.strokes.len()
            );

            // 检查连续性
            for i in 0..sample.strokes.len() - 1 {
                let dist = sample.strokes[i].p3.distance(&sample.strokes[i + 1].p0);
                assert!(
                    dist < 1.0,
                    "Stage 4 strokes should be continuous, gap {} at segment {}",
                    dist,
                    i
                );
            }

            let hw = 64 * 64;
            let topo_count: f32 = sample.maps.keypoints[..hw].iter().sum();
            let geom_count: f32 = sample.maps.keypoints[hw..].iter().sum();

            // 连续路径应该有 2 个拓扑节点
            assert!(
                topo_count >= 1.0 && topo_count <= 2.0,
                "Stage 4 should have 1-2 topological keypoints, got {}",
                topo_count
            );

            total_geom += geom_count;
        }

        // 平均几何锚点数应该 > 0（连续路径至少有 1 个连接点）
        let avg_geom = total_geom / num_samples as f32;
        assert!(
            avg_geom >= 1.0,
            "Stage 4 average geometric keypoints should be >= 1, got {}",
            avg_geom
        );
    }

    #[test]
    fn test_stage2_independent_strokes() {
        let generator = DataGenerator::new();
        let config = CurriculumConfig::from_stage(2); // independent 2-5
        let mut rng = new_rng();

        for _ in 0..10 {
            let sample = generator.generate_sample(&config, 64, &mut rng);

            // Stage 2 应该有 2-5 条独立笔画
            assert!(
                sample.strokes.len() >= 2 && sample.strokes.len() <= 5,
                "Stage 2 should generate 2-5 strokes, got {}",
                sample.strokes.len()
            );

            let hw = 64 * 64;
            let topo_count: f32 = sample.maps.keypoints[..hw].iter().sum();
            let geom_count: f32 = sample.maps.keypoints[hw..].iter().sum();

            // 独立笔画：每条 2 个拓扑节点
            let min_topo = sample.strokes.len() as f32; // 至少每条 1 个（可能重叠）
            let max_topo = (sample.strokes.len() * 2) as f32;
            assert!(
                topo_count >= min_topo && topo_count <= max_topo,
                "Stage 2 with {} strokes should have {}-{} topological keypoints, got {}",
                sample.strokes.len(),
                min_topo,
                max_topo,
                topo_count
            );

            // 独立笔画应该没有几何锚点
            assert_eq!(
                geom_count, 0.0,
                "Stage 2 should have 0 geometric keypoints, got {}",
                geom_count
            );
        }
    }

    // =========================================================================
    // 边界条件测试
    // =========================================================================

    #[test]
    fn test_stroke_at_boundary() {
        // 笔画延伸到画布边界
        let stroke = CubicBezier::new(
            Point::new(0.0, 32.0),
            Point::new(20.0, 0.0),
            Point::new(44.0, 63.0),
            Point::new(63.0, 32.0),
            3.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);

        // 应该不会 panic，且数据有效
        let skeleton_sum: f32 = maps.skeleton.iter().sum();
        assert!(skeleton_sum > 0.0, "Skeleton should not be empty");

        // 所有值都应该在有效范围内
        for &val in &maps.skeleton {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_very_short_stroke() {
        // 非常短的笔画
        let stroke = CubicBezier::new(
            Point::new(32.0, 32.0),
            Point::new(32.5, 32.5),
            Point::new(33.0, 33.0),
            Point::new(33.5, 33.5),
            2.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);

        // 应该仍然有骨架
        let skeleton_sum: f32 = maps.skeleton.iter().sum();
        assert!(skeleton_sum > 0.0, "Even short stroke should have skeleton");
    }

    #[test]
    fn test_different_image_sizes() {
        let stroke = CubicBezier::new(
            Point::new(5.0, 16.0),
            Point::new(12.0, 8.0),
            Point::new(20.0, 24.0),
            Point::new(27.0, 16.0),
            2.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();

        for size in [32, 64, 128] {
            // 缩放笔画到不同尺寸
            let scale = size as f32 / 32.0;
            let scaled_stroke = CubicBezier::new(
                Point::new(stroke.p0.x * scale, stroke.p0.y * scale),
                Point::new(stroke.p1.x * scale, stroke.p1.y * scale),
                Point::new(stroke.p2.x * scale, stroke.p2.y * scale),
                Point::new(stroke.p3.x * scale, stroke.p3.y * scale),
                stroke.w_start * scale,
                stroke.w_end * scale,
            );

            let maps = rasterizer.render(&[scaled_stroke], size);

            assert_eq!(maps.image.len(), size * size);
            assert_eq!(maps.skeleton.len(), size * size);
            assert_eq!(maps.keypoints.len(), 2 * size * size);
            assert_eq!(maps.tangent.len(), 2 * size * size);
            assert_eq!(maps.width.len(), size * size);
            assert_eq!(maps.offset.len(), 2 * size * size);

            let skeleton_sum: f32 = maps.skeleton.iter().sum();
            assert!(
                skeleton_sum > 0.0,
                "Skeleton should not be empty for size {}",
                size
            );
        }
    }

    // =========================================================================
    // 双倍角编码正确性测试
    // =========================================================================

    #[test]
    fn test_horizontal_stroke_tangent() {
        // 水平笔画：切向量应该是 (1, 0)，双倍角应该是 (cos(0), sin(0)) = (1, 0)
        let stroke = CubicBezier::new(
            Point::new(10.0, 32.0),
            Point::new(25.0, 32.0),
            Point::new(40.0, 32.0),
            Point::new(55.0, 32.0),
            2.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);
        let size = 64;

        // 找到骨架上的点并检查切向
        let mut found_points = 0;
        for y in 30..35 {
            for x in 15..50 {
                let idx = y * size + x;
                if maps.skeleton[idx] > 0.5 {
                    let cos2t = maps.tangent[0 * size * size + idx];
                    let sin2t = maps.tangent[1 * size * size + idx];

                    // 水平线的 θ ≈ 0，所以 cos(2θ) ≈ 1, sin(2θ) ≈ 0
                    assert!(
                        cos2t > 0.9,
                        "Horizontal stroke should have cos2θ ≈ 1, got {} at ({}, {})",
                        cos2t,
                        x,
                        y
                    );
                    assert!(
                        sin2t.abs() < 0.3,
                        "Horizontal stroke should have sin2θ ≈ 0, got {} at ({}, {})",
                        sin2t,
                        x,
                        y
                    );
                    found_points += 1;
                }
            }
        }
        assert!(found_points > 0, "Should find points on horizontal stroke");
    }

    #[test]
    fn test_vertical_stroke_tangent() {
        // 垂直笔画：切向量应该是 (0, 1)，θ = π/2，双倍角应该是 (cos(π), sin(π)) = (-1, 0)
        let stroke = CubicBezier::new(
            Point::new(32.0, 10.0),
            Point::new(32.0, 25.0),
            Point::new(32.0, 40.0),
            Point::new(32.0, 55.0),
            2.0,
            2.0,
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);
        let size = 64;

        let mut found_points = 0;
        for y in 15..50 {
            for x in 30..35 {
                let idx = y * size + x;
                if maps.skeleton[idx] > 0.5 {
                    let cos2t = maps.tangent[0 * size * size + idx];
                    let sin2t = maps.tangent[1 * size * size + idx];

                    // 垂直线的 θ ≈ π/2，所以 cos(2θ) ≈ -1, sin(2θ) ≈ 0
                    assert!(
                        cos2t < -0.9,
                        "Vertical stroke should have cos2θ ≈ -1, got {} at ({}, {})",
                        cos2t,
                        x,
                        y
                    );
                    assert!(
                        sin2t.abs() < 0.3,
                        "Vertical stroke should have sin2θ ≈ 0, got {} at ({}, {})",
                        sin2t,
                        x,
                        y
                    );
                    found_points += 1;
                }
            }
        }
        assert!(found_points > 0, "Should find points on vertical stroke");
    }

    // =========================================================================
    // 宽度连续性测试
    // =========================================================================

    #[test]
    fn test_width_consistency() {
        // 笔画从宽到窄
        let stroke = CubicBezier::new(
            Point::new(10.0, 32.0),
            Point::new(25.0, 32.0),
            Point::new(40.0, 32.0),
            Point::new(55.0, 32.0),
            5.0, // 起点宽
            1.0, // 终点窄
        );

        let rasterizer = Rasterizer::new();
        let maps = rasterizer.render(&[stroke], 64);
        let size = 64;

        // 在骨架上找到左侧和右侧的点
        let mut left_width = 0.0;
        let mut right_width = 0.0;

        for x in 12..18 {
            let idx = 32 * size + x;
            if maps.skeleton[idx] > 0.5 {
                left_width = maps.width[idx];
                break;
            }
        }

        for x in (48..54).rev() {
            let idx = 32 * size + x;
            if maps.skeleton[idx] > 0.5 {
                right_width = maps.width[idx];
                break;
            }
        }

        // 左侧应该比右侧宽
        assert!(
            left_width > right_width,
            "Width should decrease from {} to {}",
            left_width,
            right_width
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use crate::generator::{BatchOutput, CurriculumConfig, DataGenerator};
    use crate::stroke::new_rng;

    #[test]
    fn test_batch_output_consistency() {
        let generator = DataGenerator::new();
        let config = CurriculumConfig::from_stage(5);

        let samples = generator.generate_batch(&config, 8, 64);
        let output = BatchOutput::from_samples(samples, 64);

        // 验证批量输出的形状
        let batch_size = 8;
        let hw = 64 * 64;

        assert_eq!(output.images.len(), batch_size * hw);
        assert_eq!(output.skeletons.len(), batch_size * hw);
        assert_eq!(output.keypoints.len(), batch_size * 2 * hw);
        assert_eq!(output.tangents.len(), batch_size * 2 * hw);
        assert_eq!(output.widths.len(), batch_size * hw);
        assert_eq!(output.offsets.len(), batch_size * 2 * hw);
    }

    #[test]
    fn test_all_stages_produce_valid_output() {
        let generator = DataGenerator::new();

        for stage in 0..=9 {
            let config = CurriculumConfig::from_stage(stage);
            let mut rng = new_rng();
            let sample = generator.generate_sample(&config, 64, &mut rng);

            // 基本验证
            assert!(!sample.strokes.is_empty(), "Stage {} produced no strokes", stage);

            let skeleton_sum: f32 = sample.maps.skeleton.iter().sum();
            assert!(
                skeleton_sum > 0.0,
                "Stage {} skeleton is empty",
                stage
            );

            let image_sum: f32 = sample.maps.image.iter().sum();
            assert!(image_sum > 0.0, "Stage {} image is empty", stage);

            // keypoints 通道验证
            let hw = 64 * 64;
            let topo_count: f32 = sample.maps.keypoints[..hw].iter().sum();
            assert!(
                topo_count >= 1.0,
                "Stage {} should have at least 1 topological keypoint",
                stage
            );
        }
    }
}
