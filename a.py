import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_bezier_point(t, p0, p1, p2, p3):
    """标准的贝塞尔曲线计算公式"""
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )


def visualize_transform():
    # 1. 我们的魔法矩阵 (之前验证过的逆矩阵)
    # M_inv = A^-1
    transform_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-5 / 6, 3.0, -1.5, 1 / 3],
            [1 / 3, -1.5, 3.0, -5 / 6],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # 2. 设置 Ground Truth (上帝视角)
    # 假设这是真实的贝塞尔控制点 (P0, P1, P2, P3)
    # 注意：P1 和 P2 故意设置得很远，模拟"控制点溢出"的情况
    gt_controls = np.array(
        [
            [10, 10],  # P0 (Start)
            [15, 40],  # P1 (Control 1 - 飞得很远)
            [45, 40],  # P2 (Control 2 - 飞得很远)
            [50, 10],  # P3 (End)
        ]
    )

    # 3. 模拟"神经网络看到的东西" (采样实点)
    # 计算贝塞尔曲线在 t=[0, 1/3, 2/3, 1] 处的坐标
    t_samples = [0, 1 / 3, 2 / 3, 1]
    on_curve_points = np.array([get_bezier_point(t, *gt_controls) for t in t_samples])

    # 4. 执行转换 (模拟模型推理后的后处理)
    # [4, 4] @ [4, 2] -> [4, 2]
    # recovered_controls 就是我们需要输出给 SVG 的东西
    recovered_controls = transform_matrix @ on_curve_points

    # --- 开始绘图 ---
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # A. 画出完美的贝塞尔曲线 (用还原后的点画)
    t_smooth = np.linspace(0, 1, 100)
    curve_smooth = np.array(
        [get_bezier_point(t, *recovered_controls) for t in t_smooth]
    )
    plt.plot(
        curve_smooth[:, 0],
        curve_smooth[:, 1],
        "k-",
        linewidth=2,
        label="Reconstructed Bezier Curve",
        alpha=0.6,
    )

    # B. 画出"虚点" (控制点 P0-P3) - 也就是 GT 和 还原结果
    # 画控制手柄 (P0-P1, P2-P3)
    plt.plot(
        recovered_controls[:, 0],
        recovered_controls[:, 1],
        "b--",
        linewidth=1,
        alpha=0.5,
        label="Control Polygon (Off-Curve)",
    )
    plt.scatter(
        recovered_controls[:, 0],
        recovered_controls[:, 1],
        color="blue",
        marker="x",
        s=100,
        label="Recovered Control Points (P0..P3)",
        zorder=5,
    )

    # C. 画出"实点" (On-Curve Points) - 也就是神经网络预测的目标
    plt.scatter(
        on_curve_points[:, 0],
        on_curve_points[:, 1],
        color="red",
        s=100,
        label="Neural Net Prediction (On-Curve)",
        zorder=6,
    )

    # 标注文字
    labels_on = ["Start (P0)", "t=1/3", "t=2/3", "End (P3)"]
    for i, txt in enumerate(labels_on):
        plt.text(
            on_curve_points[i, 0],
            on_curve_points[i, 1] - 2,
            txt,
            color="red",
            ha="center",
            fontweight="bold",
        )

    labels_off = ["", "P1 (Off-Curve)", "P2 (Off-Curve)", ""]
    for i, txt in enumerate(labels_off):
        if txt:
            plt.text(
                recovered_controls[i, 0],
                recovered_controls[i, 1] + 1,
                txt,
                color="blue",
                ha="center",
            )

    # 设置图表
    plt.title("Visual Proof: On-Curve to Bezier Transformation", fontsize=14)
    plt.legend(loc="lower center")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.axis("equal")  # 保证比例一致，不拉伸

    # 打印数值验证
    print("Ground Truth Controls:\n", gt_controls)
    print("\nRecovered Controls:\n", recovered_controls)
    print("\nError (Max Diff):", np.max(np.abs(gt_controls - recovered_controls)))

    plt.show()


if __name__ == "__main__":
    visualize_transform()
