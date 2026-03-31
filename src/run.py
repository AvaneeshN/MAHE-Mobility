#!/usr/bin/env python
# src/run.py — CLI entry point for the BEV pipeline
#
# Usage:
#   python src/run.py                          # run on nuScenes (uses configs/configs.yaml)
#   python src/run.py --input data/raw/img.jpg  # run on a single custom image
#   python src/run.py --input data/raw/video.mp4 --mode video
#   python src/run.py --input data/raw/img.jpg  --mode homography

import argparse
import os
import cv2
import numpy as np

from src.pipeline import BEVPipeline, load_config, visualize
from src.utils.visualize import plot_bev_panel


def run_on_nuscenes(cfg, n_samples=10):
    """Run the full pipeline on nuScenes mini dataset."""
    from src.data.nuscenes_loader import NuScenesLoader

    loader = NuScenesLoader(
        dataroot=cfg["nuscenes"]["dataroot"],
        version=cfg["nuscenes"]["version"]
    )
    pipeline = BEVPipeline(cfg)

    os.makedirs("outputs", exist_ok=True)

    for i in range(min(n_samples, len(loader))):
        print(f"\n--- Sample {i+1}/{min(n_samples, len(loader))} ---")
        sample = loader.get_sample(i)
        image = sample["image"]
        K     = sample["intrinsic"]
        R     = sample["rotation"]
        t     = sample["translation"]

        grid, depth, detections, obj_pos = pipeline.process_frame(
            image, K, camera_height=1.51, R=R, t=t
        )

        print(f"  Detections: {len(detections)}  |  "
              f"Occupied cells: {(grid.grid > 0.3).sum()}")

        plot_bev_panel(
            image, depth, grid, detections, obj_pos,
            save_path=f"outputs/bev_sample_{i:02d}.png",
            title_suffix=f"(sample {i})"
        )


def run_on_image(image_path, cfg):
    """Run on a single image with default camera params from config."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    cam = cfg["camera"]
    K = np.array([
        [cam["fx"],       0, cam["cx"]],
        [      0, cam["fy"], cam["cy"]],
        [      0,       0,         1]
    ], dtype=np.float64)

    pipeline = BEVPipeline(cfg)
    grid, depth, detections, obj_pos = pipeline.process_frame(
        image, K, camera_height=cam["height"]
    )

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", os.path.splitext(os.path.basename(image_path))[0] + "_bev.png")
    plot_bev_panel(image, depth, grid, detections, obj_pos, save_path=out_path)
    print(f"\nDone. Output saved to {out_path}")


def run_on_video(video_path, cfg):
    """Run on a video file, saving one BEV per frame."""
    cam = cfg["camera"]
    K = np.array([
        [cam["fx"],       0, cam["cx"]],
        [      0, cam["fy"], cam["cy"]],
        [      0,       0,         1]
    ], dtype=np.float64)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    pipeline = BEVPipeline(cfg)
    os.makedirs("outputs/frames", exist_ok=True)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grid, depth, detections, obj_pos = pipeline.process_frame(
            frame, K, camera_height=cam["height"]
        )
        out_path = f"outputs/frames/bev_frame_{frame_idx:04d}.png"
        plot_bev_panel(frame, depth, grid, detections, obj_pos,
                       save_path=out_path)
        print(f"Frame {frame_idx}: {len(detections)} detections")
        frame_idx += 1

    cap.release()
    print(f"\nProcessed {frame_idx} frames → outputs/frames/")


def run_homography_mode(image_path, cfg):
    """Fast mode: warp road surface to BEV using planar homography."""
    from src.geometry.homography import InversePerspectiveMapping

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    cam = cfg["camera"]
    K = np.array([
        [cam["fx"],       0, cam["cx"]],
        [      0, cam["fy"], cam["cy"]],
        [      0,       0,         1]
    ], dtype=np.float64)

    ipm = InversePerspectiveMapping(K, camera_height=cam["height"])
    bev, H = ipm.warp(image, bev_width=400, bev_height=400)

    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = f"outputs/{base}_homography_bev.png"
    cv2.imwrite(out_path, bev)
    print(f"Homography BEV saved → {out_path}")

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
    ax2.set_title("Homography BEV (road surface only)")
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(f"outputs/{base}_homography_comparison.png", dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="BEV-Mapper: convert camera images to Bird's-Eye-View occupancy maps"
    )
    parser.add_argument("--input",   type=str, default=None,
                        help="Path to image or video. Omit to run on nuScenes.")
    parser.add_argument("--mode",    type=str, default="pipeline",
                        choices=["pipeline", "video", "homography"],
                        help="Processing mode")
    parser.add_argument("--config",  type=str, default="configs/configs.yaml",
                        help="Path to config YAML")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of nuScenes samples to process")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.input is None:
        print("No --input given. Running on nuScenes dataset.")
        run_on_nuscenes(cfg, n_samples=args.samples)
    elif args.mode == "homography":
        run_homography_mode(args.input, cfg)
    elif args.mode == "video":
        run_on_video(args.input, cfg)
    else:
        run_on_image(args.input, cfg)


if __name__ == "__main__":
    main()