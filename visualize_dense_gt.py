import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
import cv2

# Import local dense_gen
try:
    from dense_gen import batch_generate_dense_maps
except ImportError:
    # If running from root, simple import
    sys.path.append(os.getcwd())
    from dense_gen import batch_generate_dense_maps

def test_visualization():
    # 1. Generate Input Data
    use_dummy = False
    
    if not use_dummy:
        try:
            import ink_trace_rs
            print("Using ink_trace_rs for data generation.")
            B = 4
            img_size = 64
            max_strokes = 5
            imgs, labels = ink_trace_rs.generate_independent_strokes_batch(B, img_size, max_strokes)
            imgs = np.array(imgs)
            labels = np.array(labels)
        except ImportError:
            use_dummy = True

    if use_dummy:
        print("Using DUMMY vector data.")
        B = 4
        img_size = 64
        imgs = np.zeros((B, img_size, img_size), dtype=np.float32)
        # Random independent strokes
        labels = np.zeros((B, 5, 11), dtype=np.float32)
        for i in range(B):
            # One generic curve
            # p0=(0.2, 0.2), p3=(0.8, 0.8)
            labels[i, 0, :10] = [0.2, 0.2, 0.4, 0.2, 0.6, 0.8, 0.8, 0.8, 0.2, 0.5]
            labels[i, 0, 10] = 1.0
            
            # Another crossing
            labels[i, 1, :10] = [0.2, 0.8, 0.4, 0.6, 0.6, 0.4, 0.8, 0.2, 0.3, 0.3]
            labels[i, 1, 10] = 1.0

    # 2. Generate Dense Maps
    print("Generating dense maps...")
    dense_targets = batch_generate_dense_maps(labels, img_size=img_size)
    
    # 3. Visualize
    # Keys: skeleton, junction, tangent, width, offset
    
    rows = B
    cols = 7 # Input, Skeleton, Junction, TangentX, TangentY, Width, OffsetNorm
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if rows == 1: axes = axes[None, :]
    
    for i in range(B):
        # Input
        axes[i, 0].imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')
        
        # Skeleton
        skel = dense_targets['skeleton'][i, 0]
        axes[i, 1].imshow(skel, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Skeleton")
        axes[i, 1].axis('off')
        
        # Junction
        junc = dense_targets['junction'][i, 0]
        # Dilate for visibility in plot
        import cv2
        junc_vis = cv2.dilate(junc, np.ones((3,3)))
        axes[i, 2].imshow(junc_vis, cmap='magma', vmin=0, vmax=1)
        axes[i, 2].set_title("Junction")
        axes[i, 2].axis('off')
        
        # Tangent: Show Vector Field using HSV
        # Map Angle to Hue
        tan_x = dense_targets['tangent'][i, 0]
        tan_y = dense_targets['tangent'][i, 1]
        
        # Calculate angle in degrees [0, 360]
        angle = np.degrees(np.arctan2(tan_y, tan_x))
        angle[angle < 0] += 360
        
        # HSV construction
        h = angle  # 0-360
        s = np.ones_like(angle) * 255 # Saturation max
        v = np.ones_like(angle) * 255 # Value max
        
        # Mask out non-skeleton areas clearly
        mask = dense_targets['skeleton'][i, 0] > 0
        v[~mask] = 0
        
        # Convert to RGB for display
        hsv_img = np.stack([h, s, v], axis=-1).astype(np.float32)
        # OpenCV expects H: 0-180, S: 0-255, V: 0-255 for uint8
        # Or H: 0-360, S: 0-1, V: 0-1 for float32. Let's use float32 [0,1]
        hsv_img_norm = np.stack([h/360.0, np.ones_like(h), np.where(mask, 1.0, 0.0)], axis=-1)
        
        from matplotlib.colors import hsv_to_rgb
        tan_vis = hsv_to_rgb(hsv_img_norm)
        
        axes[i, 3].imshow(tan_vis) 

        axes[i, 3].set_title("Tangent")
        axes[i, 3].axis('off')

        # Width
        w_map = dense_targets['width'][i, 0]
        axes[i, 4].imshow(w_map, cmap='viridis') # Scale auto
        axes[i, 4].set_title("Width")
        axes[i, 4].axis('off')
        
        # Offset Norm (show magnitude)
        off = dense_targets['offset'][i] # [2, H, W]
        off_norm = np.sqrt(off[0]**2 + off[1]**2)
        axes[i, 5].imshow(off_norm * skel, cmap='plasma')
        axes[i, 5].set_title("Offset")
        axes[i, 5].axis('off')

        # Overlay Sketch (Skeleton on Input)
        base = np.stack([imgs[i]]*3, axis=-1)
        # Red skeleton
        base[skel > 0] = [1, 0, 0]
        axes[i, 6].imshow(base)
        axes[i, 6].set_title("Overlay")
        axes[i, 6].axis('off')
        
    plt.tight_layout()
    out_path = "vis_dense_gt.png"
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    test_visualization()
