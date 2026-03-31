# src/pipeline.py — Final Competition Version

import cv2
import numpy as np
from collections import deque
import yaml
import os
import matplotlib.pyplot as plt

from src.depth.estimator import DepthEstimator
from src.geometry.camera import Camera
from src.perception.detector import ObjectDetector
from src.mapping.point_cloud import depth_to_point_cloud
from src.mapping.occupancy_grid import OccupancyGrid
from src.data.nuscenes_loader import NuScenesLoader
from src.utils.metrics import OccupancyMetrics, load_lidar_from_nuscenes


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
        self.frame_history = deque(maxlen=4)

    def process_frame(self, image, K, camera_height=1.51):
        # 1. Depth
        depth, _ = self.depth_estimator.predict_with_scale(image, K, camera_height=camera_height)

        # 2. Detections
        detections = self.detector.detect(image)

        # 3. Point cloud
        camera = Camera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
        points_cam = depth_to_point_cloud(depth, camera)

        # 4. Filter
        max_d = self.cfg["depth"]["max_depth"]
        grid_half = self.cfg["grid"]["size"] / 2.0
        mask = (
            (points_cam[:, 2] > 0.8) &
            (points_cam[:, 2] < max_d) &
            (np.abs(points_cam[:, 0]) < grid_half)
        )
        points_cam = points_cam[mask]

        # 5. Grid
        grid = OccupancyGrid(size=self.cfg["grid"]["size"], resolution=self.cfg["grid"]["resolution"])
        grid.fill_from_points(points_cam, camera_height=camera_height)
        grid.smooth(7, 1.5)

        # 6. Detection footprints
        object_positions = []
        for det in detections:
            x1, y1, x2, y2, label, conf = det
            if label not in ["car", "truck", "bus", "person"]:
                continue

            u = int((x1 + x2) / 2)
            v = int(y2 - 8) if y2 > 20 else int(y2)
            u = np.clip(u, 0, depth.shape[1]-1)
            v = np.clip(v, 0, depth.shape[0]-1)

            Z = float(depth[v, u])
            if Z < 1.0 or Z > 40.0:
                continue

            X = (u - K[0,2]) * Z / K[0,0]
            col_arr, row_arr = grid.world_to_grid(np.array([X]), np.array([Z]))
            col, row = int(col_arr[0]), int(row_arr[0])

            grid.add_detection_footprint(col, row, label, min(conf, 0.95))
            object_positions.append((col, row, label, conf))

        # 7. Temporal fusion
        self.frame_history.append(grid.grid.copy())
        if len(self.frame_history) >= 2:
            weights = np.linspace(0.6, 1.0, len(self.frame_history))
            weights /= weights.sum()
            grid.grid = np.average(np.stack(list(self.frame_history)), axis=0, weights=weights)

        # 8. Post-process
        grid.post_process()
        grid.smooth(5, 1.3)

        return grid, depth, detections, object_positions


# Visualization
def visualize(image, depth, grid, detections, object_positions,
              gt_grid=None, metrics=None, save_path=None, title_suffix=""):
    n_panels = 4 if gt_grid is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 7))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#0d0d0d")

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    for det in detections:
        x1, y1, x2, y2, label, conf = det
        color = "lime" if label in ["car", "truck", "bus"] else "cyan"
        axes[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2))
        axes[0].text(x1, max(y1-6, 0), f"{label} {conf:.2f}", color=color, fontsize=8,
                     bbox=dict(facecolor="black", alpha=0.5))
    axes[0].set_title(f"Camera View + Detections {title_suffix}", color="white")
    axes[0].axis("off")

    vmin = float(np.percentile(depth, 2))
    vmax = float(np.percentile(depth, 98))
    im2 = axes[1].imshow(depth, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Depth Map (metres) {title_suffix}", color="white")
    axes[1].axis("off")
    cbar = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("depth (m)", color="white")

    axes[2].imshow(grid.grid, cmap="inferno", origin="lower", vmin=0, vmax=1)
    axes[2].plot(grid.N//2, 8, "w^", markersize=14)
    for col, row, label, conf in object_positions:
        axes[2].plot(col, row, "wo", markersize=6, markeredgewidth=1.5)
        axes[2].text(col+4, row+4, f"{label}\n{conf:.2f}", color="white", fontsize=7)
    if metrics:
        txt = f"IoU: {metrics['iou']:.3f}\nDW-Err: {metrics['dw_error']:.4f}"
        axes[2].text(15, grid.N-20, txt, color="lime", fontsize=9, bbox=dict(facecolor="black", alpha=0.7))
    axes[2].set_title(f"Predicted BEV {title_suffix}", color="white")
    _style_bev_axes(axes[2], grid)

    if gt_grid is not None:
        axes[3].imshow(gt_grid, cmap="Blues", origin="lower", vmin=0, vmax=1)
        axes[3].set_title("LiDAR Ground Truth", color="white")
        _style_bev_axes(axes[3], grid)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)


def _style_bev_axes(ax, grid):
    res, size = grid.res, grid.size
    step = int(5.0 / res)
    xticks = np.arange(0, grid.N + 1, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x*res - size/2:.0f}m" for x in xticks], color="white", fontsize=7)
    yticks = np.arange(0, grid.N + 1, step)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y*res:.0f}m" for y in yticks], color="white", fontsize=7)
    ax.set_xlabel("X lateral", color="white")
    ax.set_ylabel("Z forward (m)", color="white")
    ax.tick_params(colors="white")


def main():
    cfg = load_config()
    loader = NuScenesLoader(dataroot=cfg["nuscenes"]["dataroot"], version=cfg["nuscenes"]["version"])
    pipeline = BEVPipeline(cfg)

    metrics_engine = OccupancyMetrics(grid_size=cfg["grid"]["size"], resolution=cfg["grid"]["resolution"], occ_thresh=0.25)

    all_ious, all_dw = [], []
    os.makedirs("outputs", exist_ok=True)

    n = min(15, len(loader))
    for i in range(n):
        print(f"\n=== Sample {i+1}/{n} ===")
        sample = loader.get_sample(i)
        image = sample["image"]
        K = sample["intrinsic"]

        grid, depth, detections, obj_pos = pipeline.process_frame(image, K)

        try:
            lidar_pts = load_lidar_from_nuscenes(loader.nusc, sample["sample_token"])
            gt_grid = metrics_engine.lidar_to_gt_grid(lidar_pts)
            result = metrics_engine.evaluate(grid.grid, lidar_pts)
            all_ious.append(result["iou"])
            all_dw.append(result["dw_error"])
            print(f"IoU: {result['iou']:.4f} | DW-Error: {result['dw_error']:.4f}")
        except Exception as e:
            print(f"LiDAR skipped: {e}")
            gt_grid = None
            result = None

        visualize(image, depth, grid, detections, obj_pos, gt_grid, result,
                  save_path=f"outputs/bev_sample_{i:02d}.png", title_suffix=f"(sample {i})")

    if all_ious:
        print(f"\n=== FINAL ===\nMean IoU: {np.mean(all_ious):.4f} | Mean DW-Error: {np.mean(all_dw):.4f}")


if __name__ == "__main__":
    main()