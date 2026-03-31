# src/pipeline.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.depth.estimator import DepthEstimator
from src.geometry.camera import Camera
from src.perception.detector import ObjectDetector
from src.mapping.point_cloud import depth_to_point_cloud
from src.mapping.occupancy_grid import OccupancyGrid, filter_ground
from src.data.nuscenes_loader import NuScenesLoader


def load_config(path="configs/configs.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_single_frame(image, camera, depth_estimator, detector, cfg):
    """
    Full BEV pipeline for one image.
    Returns (bev_grid, depth_map, detections)
    """

    # 1. Depth estimation
    depth = depth_estimator.predict(image)

    # 2. Object detection
    detections = detector.detect(image)

    # 3. Point cloud from depth
    points_cam = depth_to_point_cloud(depth, camera)

    # 4. Transform to world frame (vectorized)
    points_world = camera.camera_to_world(points_cam)

    # 5. Filter to ground level using real camera height
    cam_height = cfg["camera"]["height"]
    thresh = cfg["grid"]["height_thresh"]
    ground_pts = filter_ground(points_world, camera_height=cam_height, thresh=thresh)

    # 6. Build occupancy grid
    grid_cfg = cfg["grid"]
    grid = OccupancyGrid(size=grid_cfg["size"], resolution=grid_cfg["resolution"])
    grid.fill(ground_pts)
    grid.smooth(kernel_size=5)

    # 7. Overlay detected object positions onto grid
    for det in detections:
        x1, y1, x2, y2, label, conf = det
        if label not in ["car", "truck", "bus", "motorcycle", "person"]:
            continue

        # Bottom-centre of bounding box = object's ground contact point
        u = int((x1 + x2) / 2)
        v = int(y2)
        u = np.clip(u, 0, depth.shape[1] - 1)
        v = np.clip(v, 0, depth.shape[0] - 1)

        Z = depth[v, u]
        if Z <= 0 or Z > cfg["depth"]["max_depth"]:
            continue

        pt3d = camera.unproject_pixel(u, v, Z)
        pt_world = camera.camera_to_world(pt3d)
        x, _, z = pt_world

        col, row = grid.world_to_grid(
            np.array([x]), np.array([z])
        )
        col, row = int(col[0]), int(row[0])

        # Paint a 5x5 blob for each detected object
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                r, c = row + dy, col + dx
                if 0 <= r < grid.N and 0 <= c < grid.N:
                    grid.grid[r, c] = 1.0

    return grid, depth, detections


def visualize(image, depth, grid, detections, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Camera image with detections
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    for det in detections:
        x1, y1, x2, y2, label, conf = det
        axes[0].add_patch(plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, edgecolor='lime', linewidth=2
        ))
        axes[0].text(x1, y1-5, f"{label} {conf:.2f}",
                     color='lime', fontsize=7, weight='bold')
    axes[0].set_title("Camera View + Detections")
    axes[0].axis("off")

    # Panel 2: Depth map
    axes[1].imshow(depth, cmap='plasma')
    axes[1].set_title("Depth Map (metres)")
    axes[1].axis("off")

    # Panel 3: BEV occupancy grid
    axes[2].imshow(grid.grid, cmap='inferno', origin='lower')
    axes[2].set_title("BEV Occupancy Grid (Top-Down)")
    axes[2].set_xlabel("X (cells)")
    axes[2].set_ylabel("Forward (cells)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def main():
    cfg = load_config()

    # Load nuScenes dataset
    loader = NuScenesLoader(
        dataroot=cfg["nuscenes"]["dataroot"],
        version=cfg["nuscenes"]["version"]
    )

    # Load models
    depth_estimator = DepthEstimator(
        model_type=cfg["models"]["depth"],
        scale_factor=cfg["depth"]["scale_factor"],
        min_depth=cfg["depth"]["min_depth"],
        max_depth=cfg["depth"]["max_depth"],
    )
    detector = ObjectDetector(model_name=cfg["models"]["yolo"])

    # Run on first 5 samples
    for i in range(min(5, len(loader))):
        print(f"\n--- Sample {i+1} ---")
        sample = loader.get_sample(i)

        image = sample["image"]
        K = sample["intrinsic"]
        R = sample["rotation"]
        t = sample["translation"]

        # Build camera with REAL nuScenes calibration
        camera = Camera(
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2],
            R=R, t=t
        )

        grid, depth, detections = run_single_frame(
            image, camera, depth_estimator, detector, cfg
        )

        print(f"Detections: {len(detections)}")
        print(f"Grid occupied cells: {(grid.grid > 0.5).sum()}")

        visualize(
            image, depth, grid, detections,
            save_path=f"outputs/bev_sample_{i}.png"
        )


if __name__ == "__main__":
    main()