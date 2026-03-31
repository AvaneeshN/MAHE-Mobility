# src/mapping/occupancy_grid.py

import numpy as np
import cv2


class OccupancyGrid:
    def __init__(self, size=40.0, resolution=0.1):
        """
        size: real-world coverage in metres (square)
        resolution: metres per cell (0.1 = 10cm per cell)
        """
        self.size = size
        self.res = resolution
        self.N = int(size / resolution)
        self.grid = np.zeros((self.N, self.N), dtype=np.float32)

    def world_to_grid(self, x, z):
        """Convert real-world (x, z) to grid (col, row). Vectorized."""
        col = ((x + self.size / 2) / self.res).astype(int)
        row = (z / self.res).astype(int)
        return col, row

    def fill(self, points):
        """
        Fill grid using point cloud. Fully vectorized — no for-loop.
        points: np.array of shape [N, 3] (x, y, z in world frame)
        """
        if len(points) == 0:
            return self.grid

        x = points[:, 0]
        z = points[:, 2]

        col = ((x + self.size / 2) / self.res).astype(int)
        row = (z / self.res).astype(int)

        # Keep only in-bounds indices
        mask = (row >= 0) & (row < self.N) & (col >= 0) & (col < self.N)
        row, col = row[mask], col[mask]

        # Use np.add.at for accumulation (handles duplicate indices)
        np.add.at(self.grid, (row, col), 0.2)
        self.grid = np.clip(self.grid, 0.0, 1.0)

        return self.grid

    def smooth(self, kernel_size=5):
        """Apply Gaussian blur to reduce noise."""
        self.grid = cv2.GaussianBlur(self.grid, (kernel_size, kernel_size), 0)
        return self.grid

    def reset(self):
        """Clear the grid (use between frames)."""
        self.grid = np.zeros((self.N, self.N), dtype=np.float32)


def filter_ground(points, camera_height=1.51, thresh=0.5):
    """
    Keep only near-ground points.

    camera_height: how high the camera is above ground (metres)
    thresh: tolerance band above/below ground plane

    In camera frame, Y axis points down. Ground is at Y ~ camera_height.
    We keep points where Y is close to camera_height (i.e., near the floor).
    """
    ground_y = camera_height
    mask = np.abs(points[:, 1] - ground_y) < thresh
    return points[mask]