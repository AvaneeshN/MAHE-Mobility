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

    def process_frame(self, image, K, camera_height=1.51, R=None, t=None):
        """
        Process one frame through the full BEV pipeline.

        Args:
            image:          BGR image (H x W x 3)
            K:              3x3 intrinsic matrix (from nuScenes calibration)
            camera_height:  height of camera above ground in metres (default 1.51 for nuScenes)
            R:              3x3 rotation matrix camera→world (optional, from nuScenes)
            t:              3-vector translation camera→world (optional, from nuScenes)

        Returns:
            grid, depth, detections, object_positions
        """
        # Build Camera object with extrinsics if provided
        if R is not None and t is not None:
            camera = Camera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2], R=R, t=t)
        else:
            camera = Camera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
        camera.K = K

        # --- FIX: use ground-plane-calibrated depth instead of raw MiDaS ---
        depth, scale = self.depth_estimator.predict_with_scale(
            image, K, camera_height=camera_height
        )

        # Debug: print depth statistics so we can verify calibration is working
        print(f"  Depth stats → min:{depth.min():.1f}m  median:{np.median(depth):.1f}m  "
              f"max:{depth.max():.1f}m  scale_s={scale:.3f}")

        detections = self.detector.detect(image)

        # Point cloud in camera frame (N x 3: X_cam, Y_cam, Z_cam)
        points_cam = depth_to_point_cloud(depth, camera)

        # --- FIX: ground filtering in CAMERA frame ---
        # In the nuScenes camera frame, Y axis points DOWN.
        # Ground plane is at Y ≈ camera_height (below the camera).
        # Keep points where Y is within ±thresh of the ground level.
        ground_y = camera_height
        thresh = self.cfg["grid"].get("height_thresh", 0.5)

        # Also basic Z range filter (don't want sky or things behind camera)
        z_mask = (points_cam[:, 2] > 1.0) & (points_cam[:, 2] < self.cfg["depth"]["max_depth"])
        y_mask = np.abs(points_cam[:, 1] - ground_y) < thresh
        x_mask = np.abs(points_cam[:, 0]) < (self.cfg["grid"]["size"] / 2.0)
        points_cam = points_cam[z_mask & y_mask & x_mask]

        # If R and t provided, transform to world frame for the grid
        # Otherwise stay in camera frame (Z forward = forward in BEV, X = lateral)
        if R is not None and t is not None:
            points_world = camera.camera_to_world(points_cam)
            # In world frame, use X and Y for lateral/forward (nuScenes convention)
            # Remap so BEV grid uses (x_world, y_world) as (lateral, forward)
            grid_x = points_world[:, 0]
            grid_z = points_world[:, 1]
            points_for_grid = np.stack([grid_x, np.zeros_like(grid_x), grid_z], axis=-1)
        else:
            # Camera frame: X=lateral, Z=forward — works directly
            points_for_grid = points_cam

        # Build occupancy grid
        grid_cfg = self.cfg["grid"]
        grid = OccupancyGrid(size=grid_cfg["size"], resolution=grid_cfg["resolution"])
        grid.fill(points_for_grid)
        grid.smooth(kernel_size=5)

        # --- Overlay YOLO detections as labelled blobs ---
        object_positions = []
        for det in detections:
            x1, y1, x2, y2, label, conf = det
            if label not in ["car", "truck", "bus", "motorcycle", "person"]:
                continue

            # Use bottom-centre of bounding box (feet / base of object)
            u = int((x1 + x2) / 2)
            v = int(y2)
            u = np.clip(u, 0, depth.shape[1] - 1)
            v = np.clip(v, 0, depth.shape[0] - 1)

            Z = float(depth[v, u])
            if Z <= 1.0 or Z > self.cfg["depth"]["max_depth"]:
                continue

            fx, cx = K[0, 0], K[0, 2]
            X = (u - cx) * Z / fx

            col, row = grid.world_to_grid(np.array([X]), np.array([Z]))
            col, row = int(col[0]), int(row[0])

            if not (0 <= row < grid.N and 0 <= col < grid.N):
                continue

            # Paint a blob — size scales with object type
            blob = 6 if label in ["car", "truck", "bus"] else 4
            for dx in range(-blob, blob + 1):
                for dy in range(-blob, blob + 1):
                    r, c = row + dy, col + dx
                    if 0 <= r < grid.N and 0 <= c < grid.N:
                        grid.grid[r, c] = 1.0

            object_positions.append((col, row, label, conf))

        # Temporal fusion — max across last N frames
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
        K     = sample["intrinsic"]
        R     = sample["rotation"]
        t     = sample["translation"]

        grid, depth, detections, obj_positions = pipeline.process_frame(
            image, K, camera_height=1.51, R=R, t=t
        )

        print(f"Detections: {len(detections)} | Occupied cells: {(grid.grid > 0.4).sum()}")

        visualize(
            image, depth, grid, detections, obj_positions,
            save_path=f"outputs/bev_sample_{i:02d}.png"
        )


if __name__ == "__main__":
    main()