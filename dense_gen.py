import numpy as np


def generate_dense_maps(strokes, img_size=64):
    """
    Generate Dense GT Maps from vector stroke parameters.

    Args:
        strokes: np.array of shape [N, 11] or [N, 10]
                 Format: [x0, y0, x1, y1, x2, y2, x3, y3, w_start, w_end, (valid)]
                 Coordinates should be normalized 0-1 (will be scaled to img_size).
        img_size: int, size of the output maps.

    Returns:
        maps: dict containing:
            - 'skeleton': [H, W] (0/1)
            - 'junction': [H, W] (0/1) - All special points (endpoints + connections)
            - 'tangent':  [2, H, W] (cos2t, sin2t) - Double Angle Representation
            - 'width':    [H, W]
            - 'offset':   [2, H, W] (dx, dy)
    
    Note:
        Junction map marks ALL P0 and P3 points of all strokes.
        For continuous strokes, connection points (P3[i] == P0[i+1]) will also be marked.
        The post-processing algorithm distinguishes endpoints vs connections by
        analyzing the skeleton topology (neighbor count).
    """
    H, W = img_size, img_size

    # Initialize maps
    skeleton = np.zeros((H, W), dtype=np.float32)
    junction = np.zeros((H, W), dtype=np.float32)
    tangent = np.zeros((2, H, W), dtype=np.float32)
    width = np.zeros((H, W), dtype=np.float32)
    offset = np.zeros((2, H, W), dtype=np.float32)

    for s in strokes:
        # Check validity if 11th dim exists
        if s.shape[0] > 10 and s[10] < 0.5:
            continue

        # Unpack control points (0-1 normalized -> pixel coordinates)
        p0 = s[0:2] * img_size
        p1 = s[2:4] * img_size
        p2 = s[4:6] * img_size
        p3 = s[6:8] * img_size
        # Width is normalized by 10.0 in Rust, restore to pixel units
        w_start = s[8] * 10.0
        w_end = s[9] * 10.0

        # 1. Junctions (mark ALL endpoints P0 and P3)
        for pt in [p0, p3]:
            px, py = int(np.round(pt[0])), int(np.round(pt[1]))
            if 0 <= px < W and 0 <= py < H:
                junction[py, px] = 1.0

        # 2. Skeleton & Attributes (Sampling)
        chord = np.linalg.norm(p3 - p0)
        cont_net = (
            np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
        )
        est_len = (chord + cont_net) / 2.0

        # Step size: 0.5 pixel to ensure connectivity
        if est_len < 1.0:
            steps = 2
        else:
            steps = int(est_len * 2) + 1

        t_vals = np.linspace(0, 1, steps)

        # Vectorized Bezier evaluation
        tm = 1.0 - t_vals

        # Coefficients [steps, 1]
        c0 = (tm**3)[:, None]
        c1 = (3 * (tm**2) * t_vals)[:, None]
        c2 = (3 * tm * (t_vals**2))[:, None]
        c3 = (t_vals**3)[:, None]

        # Points [steps, 2]
        pts = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3

        # Derivative Coefficients
        d0 = (3 * (tm**2))[:, None]
        d1 = (6 * tm * t_vals)[:, None]
        d2 = (3 * (t_vals**2))[:, None]

        # Derivatives [steps, 2]
        v10 = p1 - p0
        v21 = p2 - p1
        v32 = p3 - p2

        derivs = d0 * v10 + d1 * v21 + d2 * v32

        # Normalize tangents
        norms = np.linalg.norm(derivs, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        tangents = derivs / norms

        # Convert to Double Angle representation (cos2t, sin2t)
        ux = tangents[:, 0]
        uy = tangents[:, 1]

        cos2t = ux**2 - uy**2
        sin2t = 2 * ux * uy

        tangents_double = np.stack([cos2t, sin2t], axis=1)

        # Widths
        widths = w_start + (w_end - w_start) * t_vals

        # Rasterize
        for i in range(steps):
            bx, by = pts[i]
            ix = int(np.floor(bx))
            iy = int(np.floor(by))

            if 0 <= ix < W and 0 <= iy < H:
                skeleton[iy, ix] = 1.0
                tangent[0, iy, ix] = tangents_double[i, 0]
                tangent[1, iy, ix] = tangents_double[i, 1]
                width[iy, ix] = widths[i]

                # Offset: vector from pixel center to true point
                dex = bx - (ix + 0.5)
                dey = by - (iy + 0.5)
                offset[0, iy, ix] = dex
                offset[1, iy, ix] = dey

    return {
        "skeleton": skeleton,
        "junction": junction,
        "tangent": tangent,
        "width": width,
        "offset": offset,
    }


def batch_generate_dense_maps(labels_batch, img_size=64):
    """
    Process a batch of labels.
    labels_batch: [B, N, 11]
    """
    B = labels_batch.shape[0]
    maps_list = []

    for i in range(B):
        m = generate_dense_maps(labels_batch[i], img_size)
        maps_list.append(m)

    # Stack into tensors
    res = {
        "skeleton": np.stack([m["skeleton"] for m in maps_list])[:, None, ...],
        "junction": np.stack([m["junction"] for m in maps_list])[:, None, ...],
        "tangent": np.stack([m["tangent"] for m in maps_list]),
        "width": np.stack([m["width"] for m in maps_list])[:, None, ...],
        "offset": np.stack([m["offset"] for m in maps_list]),
    }
    return res
