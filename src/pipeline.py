# src/pipeline.py - Final Version with All Improvements

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import yaml

from src.depth.estimator import DepthEstimator
from src.geometry.camera import Camera
from src.perception.detector import ObjectDetector
from src.mapping.point_cloud import depth_to_point_cloud
from src.mapping.occupancy_grid import OccupancyGrid
from src.data.nuscenes_loader import NuScenesLoader


def load_config(path="configs/configs.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class BEVPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.depth_estimator = DepthEstimator(
            model_type=cfg["models"]["depth"],
            scale_factor=cfg["depth"]["scale_factor"],
            min_depth=cfg["depth"]["min_depth"],
            max_depth=cfg["depth"]["max_depth"],
        )
        self.detector = ObjectDetector(model_name=cfg["models"]["yolo"])

        # Temporal fusion: keep last N frames for denser map
        self.frame_history = deque(maxlen=4)

    def process_frame(self, image, K, camera_height=1.51):
        camera = Camera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
        camera.K = K

        # Depth + Detection
        depth = self.depth_estimator.predict(image)
        detections = self.detector.detect(image)

        # Point cloud in camera frame
        points_cam = depth_to_point_cloud(depth, camera)

        # Better filtering using camera height (Improvement 2)
        # Remove points that are too high or too low relative to camera
        mask = (points_cam[:, 2] > 0.5) & (points_cam[:, 2] < 50.0) & \
               (np.abs(points_cam[:, 0]) < 25.0) & \
               (np.abs(points_cam[:, 1]) < camera_height + 2.0)   # Y = height
        points_cam = points_cam[mask]

        # Build grid
        grid_cfg = self.cfg["grid"]
        grid = OccupancyGrid(size=grid_cfg["size"], resolution=grid_cfg["resolution"])
        grid.fill(points_cam)
        grid.smooth(kernel_size=7)

        # Overlay detections with labels
        object_positions = []
        for det in detections:
            x1, y1, x2, y2, label, conf = det
            if label not in ["car", "truck", "bus", "motorcycle", "person"]:
                continue

            u = int((x1 + x2) / 2)
            v = int(y2)
            u = np.clip(u, 0, depth.shape[1] - 1)
            v = np.clip(v, 0, depth.shape[0] - 1)

            Z = float(depth[v, u])
            if Z <= 0.5 or Z > 50.0:
                continue

            fx = K[0, 0]
            cx = K[0, 2]
            X = (u - cx) * Z / fx

            col, row = grid.world_to_grid(np.array([X]), np.array([Z]))
            col, row = int(col[0]), int(row[0])

            # Larger blob for better visibility
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    r, c = row + dy, col + dx
                    if 0 <= r < grid.N and 0 <= c < grid.N:
                        grid.grid[r, c] = 1.0

            object_positions.append((col, row, label, conf))

        # Temporal fusion (Improvement 1)
        self.frame_history.append((grid.grid.copy(), object_positions))
        if len(self.frame_history) > 1:
            fused_grid = np.maximum.reduce([g for g, _ in self.frame_history])
            grid.grid = fused_grid

        return grid, depth, detections, object_positions


def visualize(image, depth, grid, detections, object_positions, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Camera View
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    for det in detections:
        x1, y1, x2, y2, label, conf = det
        axes[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                          fill=False, edgecolor='lime', linewidth=2.5))
        axes[0].text(x1, y1-8, f"{label} {conf:.2f}",
                     color='lime', fontsize=9, weight='bold')
    axes[0].set_title("Camera View + Detections")
    axes[0].axis("off")

    # Depth Map
    axes[1].imshow(depth, cmap='plasma')
    axes[1].set_title("Depth Map (metres)")
    axes[1].axis("off")

    # BEV Grid with labels
    axes[2].imshow(grid.grid, cmap='inferno', origin='lower')
    axes[2].set_title("BEV Occupancy Grid (with labels)")
    axes[2].set_xlabel("X lateral (cells)")
    axes[2].set_ylabel("Z forward (cells)")

    # Draw labels on BEV
    for col, row, label, conf in object_positions:
        axes[2].text(col, row, f"{label}\n{conf:.2f}", color='white',
                     fontsize=10, ha='center', va='center', weight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def main():
    cfg = load_config()

    loader = NuScenesLoader(
        dataroot=cfg["nuscenes"]["dataroot"],
        version=cfg["nuscenes"]["version"]
    )

    pipeline = BEVPipeline(cfg)

    for i in range(min(10, len(loader))):
        print(f"\n--- Processing Sample {i+1} ---")
        sample = loader.get_sample(i)

        image = sample["image"]
        K = sample["intrinsic"]

        grid, depth, detections, obj_positions = pipeline.process_frame(image, K, camera_height=1.51)

        print(f"Detections: {len(detections)} | Occupied cells: {(grid.grid > 0.4).sum()}")

        visualize(
            image, depth, grid, detections, obj_positions,
            save_path=f"outputs/bev_sample_{i:02d}.png"
        )


if __name__ == "__main__":
    main()