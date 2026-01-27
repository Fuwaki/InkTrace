import ink_trace_rs
import matplotlib.pyplot as plt
import time
import numpy as np


def test_speed():
    batch_size = 64
    img_size = 64

    print("Testing Single Stroke Generation...")
    start = time.time()
    imgs, labels = ink_trace_rs.generate_single_stroke_batch(batch_size, img_size)
    end = time.time()
    print(f"Generated {batch_size} images in {end - start:.4f}s. Shape: {imgs.shape}")

    # Visualize first one
    plt.figure()
    plt.imshow(imgs[0], cmap="gray")
    plt.title("Single Stroke Rust")
    plt.savefig("test_rust_single.png")

    print("\nTesting Independent Strokes Generation...")
    start = time.time()
    # Batch=64, Max=5 strokes
    imgs, labels = ink_trace_rs.generate_independent_strokes_batch(
        batch_size, img_size, 5
    )
    end = time.time()
    print(
        f"Generated {batch_size} images in {end - start:.4f}s. Shape: {imgs.shape}, Labels: {labels.shape}"
    )

    plt.figure()
    plt.imshow(imgs[0], cmap="gray")
    plt.title("Independent Strokes Rust")
    plt.savefig("test_rust_indep.png")

    print("\nTesting Continuous Strokes Generation...")
    start = time.time()
    # Batch=64, Max=5 segments
    imgs, labels = ink_trace_rs.generate_continuous_strokes_batch(
        batch_size, img_size, 5
    )
    end = time.time()
    print(f"Generated {batch_size} images in {end - start:.4f}s. Shape: {imgs.shape}")

    plt.figure()
    plt.imshow(imgs[0], cmap="gray")
    plt.title("Continuous Strokes Rust")
    plt.savefig("test_rust_cont.png")

    print("\nTesting Multi-Connected Strokes Generation...")
    start = time.time()
    # Batch=64, MaxPaths=3, MaxSegments=5
    imgs, labels = ink_trace_rs.generate_multi_connected_strokes_batch(
        batch_size, img_size, 3, 5
    )
    end = time.time()
    print(
        f"Generated {batch_size} images in {end - start:.4f}s. Shape: {imgs.shape}, Labels: {labels.shape}"
    )

    plt.figure()
    plt.imshow(imgs[0], cmap="gray")
    plt.title("Multi-Connected Strokes Rust")
    plt.savefig("test_rust_multi.png")


if __name__ == "__main__":
    try:
        test_speed()
        print("\nSuccess! Rust module is working.")
    except ImportError:
        print("Error: ink_trace_rs not found. Did you run 'maturin develop'?")
