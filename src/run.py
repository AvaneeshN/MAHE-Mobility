#!/usr/bin/env python
# src/run.py — Final Fixed Version

import argparse
import os
import cv2
import numpy as np

from src.pipeline import BEVPipeline, load_config, visualize


def run_on_nuscenes(cfg, n_samples=10):
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

        grid, depth, detections, obj_pos = pipeline.process_frame(
            image, K, camera_height=1.51
        )

        print(f"  Detections: {len(detections)}  |  Occupied cells: {(grid.grid > 0.25).sum()}")

        visualize(
            image, depth, grid, detections, obj_pos,
            save_path=f"outputs/bev_sample_{i:02d}.png",
            title_suffix=f"(sample {i})"
        )


def main():
    parser = argparse.ArgumentParser(description="BEV-Mapper")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--mode", type=str, default="pipeline", choices=["pipeline", "video", "homography"])
    parser.add_argument("--config", type=str, default="configs/configs.yaml")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.input is None:
        print("Running on nuScenes dataset.")
        run_on_nuscenes(cfg, n_samples=args.samples)
    else:
        print("Custom input mode not enabled in this version.")


if __name__ == "__main__":
    main()