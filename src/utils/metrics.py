# src/utils/metrics.py
#
# Competition evaluation metrics for BEV occupancy maps.
#
# Two required metrics:
#   1. Occupancy IoU  — how well our grid matches LiDAR ground truth
#   2. Distance-weighted Error — errors near the ego vehicle penalised more
#
# Usage:
#   from src.utils.metrics import OccupancyMetrics
#   metrics = OccupancyMetrics(grid_size=20.0, resolution=0.1)
#   results = metrics.evaluate(pred_grid, lidar_points)
#   print(results)

import numpy as np


class OccupancyMetrics:
    """
    Evaluate a predicted BEV occupancy grid against LiDAR ground truth.

    Args:
        grid_size:   real-world coverage in metres (square grid)
        resolution:  metres per cell (0.1 = 10cm)
        occ_thresh:  probability threshold to call a cell "occupied" (default 0.3)
        dist_sigma:  sigma for distance weighting (Gaussian, in metres)
    """

    def __init__(self, grid_size=20.0, resolution=0.1,
                 occ_thresh=0.3, dist_sigma=5.0):
        self.size = grid_size
        self.res  = resolution
        self.N    = int(grid_size / resolution)
        self.occ_thresh = occ_thresh
        self.dist_sigma = dist_sigma

        # Pre-compute distance-weight matrix
        # Each cell gets a weight = exp(-d² / 2σ²) where d = distance from ego (metres)
        self._dist_weights = self._make_distance_weights()

    def _make_distance_weights(self):
        """
        Build an (N x N) matrix where each cell's value is its distance weight.
        Ego vehicle is at the bottom centre of the grid (row=0, col=N//2).
        Cells closer to the vehicle get HIGHER weight (more penalty for errors there).
        """
        rows = np.arange(self.N)  # row 0 = closest to ego
        cols = np.arange(self.N)
        col_grid, row_grid = np.meshgrid(cols, rows)

        # Convert to metres
        x_m = (col_grid - self.N / 2) * self.res   # lateral distance
        z_m = row_grid * self.res                    # forward distance

        d = np.sqrt(x_m**2 + z_m**2)

        # Gaussian weight: closer = higher weight
        weights = np.exp(-d**2 / (2 * self.dist_sigma**2))
        # Normalise so weights sum to 1
        weights /= weights.sum()
        return weights.astype(np.float32)

    def lidar_to_gt_grid(self, lidar_points, camera_height=1.51, ground_thresh=0.3):
        """
        Convert nuScenes LiDAR point cloud to a binary ground-truth occupancy grid.

        Args:
            lidar_points:   np.array (N, 3) — 3D points in EGO frame
                            (nuScenes ego frame: X=forward, Y=left, Z=up)
            camera_height:  not used directly, but kept for signature compat
            ground_thresh:  height above ground (Z < thresh → ground plane)

        Returns:
            gt_grid:   (N_cells x N_cells) binary array, 1=occupied, 0=free
        """
        gt_grid = np.zeros((self.N, self.N), dtype=np.float32)

        if lidar_points is None or len(lidar_points) == 0:
            return gt_grid

        # nuScenes ego frame: X=forward, Y=left, Z=up
        x_fwd = lidar_points[:, 0]   # forward
        y_lat = lidar_points[:, 1]   # lateral (left positive)
        z_up  = lidar_points[:, 2]   # height

        # Keep only near-ground points
        mask = (z_up > -0.5) & (z_up < ground_thresh)
        x_fwd = x_fwd[mask]
        y_lat = y_lat[mask]

        # Map forward → row (ego at row=0, forward → increasing row)
        # Map lateral → col (centre at col=N/2, left=+Y → smaller col)
        row = (x_fwd / self.res).astype(int)
        col = ((-y_lat + self.size / 2) / self.res).astype(int)  # flip Y

        valid = (row >= 0) & (row < self.N) & (col >= 0) & (col < self.N)
        row, col = row[valid], col[valid]

        np.add.at(gt_grid, (row, col), 1.0)
        gt_grid = (gt_grid > 0).astype(np.float32)

        return gt_grid

    def occupancy_iou(self, pred_grid, gt_grid):
        """
        Compute binary Intersection-over-Union between prediction and ground truth.

        Args:
            pred_grid:  (N x N) float array, occupancy probability [0, 1]
            gt_grid:    (N x N) binary float array (0 or 1)

        Returns:
            iou:        float in [0, 1]. Higher is better. 1.0 = perfect match.
        """
        pred_binary = (pred_grid >= self.occ_thresh).astype(bool)
        gt_binary   = (gt_grid   >= 0.5).astype(bool)

        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union        = np.logical_or(pred_binary, gt_binary).sum()

        if union == 0:
            return 1.0  # both empty = perfect

        return float(intersection) / float(union)

    def distance_weighted_error(self, pred_grid, gt_grid):
        """
        Compute distance-weighted binary cross-entropy error.

        Cells closer to the ego vehicle are penalised more heavily for errors.
        Uses the pre-computed Gaussian distance weight matrix.

        Args:
            pred_grid:  (N x N) float, occupancy probability [0, 1]
            gt_grid:    (N x N) binary float (0 or 1)

        Returns:
            dw_error:   float >= 0. Lower is better. 0.0 = perfect.
        """
        pred = np.clip(pred_grid.astype(np.float32), 1e-6, 1 - 1e-6)
        gt   = gt_grid.astype(np.float32)

        # Binary cross-entropy per cell
        bce = -(gt * np.log(pred) + (1 - gt) * np.log(1 - pred))

        # Apply distance weights
        dw_error = float(np.sum(self._dist_weights * bce))
        return dw_error

    def evaluate(self, pred_grid, lidar_points, camera_height=1.51):
        """
        Full evaluation: compute all competition metrics given a predicted grid
        and raw LiDAR points from nuScenes.

        Args:
            pred_grid:       (N x N) float occupancy grid from our pipeline
            lidar_points:    (M, 3) LiDAR points in ego frame from nuScenes
            camera_height:   camera height above ground

        Returns:
            dict with keys: 'iou', 'dw_error', 'occupied_cells', 'gt_cells'
        """
        gt_grid = self.lidar_to_gt_grid(lidar_points, camera_height)
        iou     = self.occupancy_iou(pred_grid, gt_grid)
        dw_err  = self.distance_weighted_error(pred_grid, gt_grid)

        return {
            "iou":            iou,
            "dw_error":       dw_err,
            "occupied_cells": int((pred_grid >= self.occ_thresh).sum()),
            "gt_cells":       int((gt_grid >= 0.5).sum()),
        }

    def evaluate_sequence(self, pred_grids, lidar_points_list, camera_height=1.51):
        """
        Evaluate over a sequence of frames and return mean metrics.

        Args:
            pred_grids:          list of (N x N) predicted grids
            lidar_points_list:   list of (M, 3) LiDAR point clouds

        Returns:
            dict with mean 'iou' and mean 'dw_error' over all frames
        """
        ious, errs = [], []
        for pred, lidar in zip(pred_grids, lidar_points_list):
            result = self.evaluate(pred, lidar, camera_height)
            ious.append(result["iou"])
            errs.append(result["dw_error"])

        return {
            "mean_iou":      float(np.mean(ious)),
            "mean_dw_error": float(np.mean(errs)),
            "frame_ious":    ious,
            "frame_errors":  errs,
        }


def load_lidar_from_nuscenes(nusc, sample_token, ego_frame=True):
    """
    Load LiDAR points for a nuScenes sample.

    Args:
        nusc:         NuScenes instance
        sample_token: sample token string
        ego_frame:    if True, transform points to ego vehicle frame

    Returns:
        points: np.array (N, 3) — X=forward, Y=left, Z=up in ego frame
    """
    import os
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion

    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)

    # Load raw point cloud
    pc_path = os.path.join(nusc.dataroot, lidar_data["filename"])
    pc = LidarPointCloud.from_file(pc_path)

    if ego_frame:
        # Transform from sensor frame → ego frame
        calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        rot = Quaternion(calib["rotation"]).rotation_matrix
        trans = np.array(calib["translation"])
        pc.rotate(rot)
        pc.translate(trans)

    # Return X, Y, Z (first 3 dims)
    return pc.points[:3, :].T   # shape (N, 3)