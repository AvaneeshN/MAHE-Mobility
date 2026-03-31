import numpy as np
import cv2

class OccupancyGrid:
    def __init__(self, size=40.0, resolution=0.08):
        self.size = size
        self.res = resolution
        self.N = int(size / resolution)
        self.grid = np.zeros((self.N, self.N), dtype=np.float32)

    def world_to_grid(self, x, z):
        col = ((x + self.size / 2) / self.res).astype(int)
        row = (z / self.res).astype(int)
        return col, row

    def fill_from_points(self, points_cam, camera_height=1.51):
        if len(points_cam) == 0:
            return self.grid
        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2]

        ground_y = camera_height
        height_diff = np.abs(Y - ground_y)
        ground_mask = height_diff < 0.65

        if not np.any(ground_mask):
            return self.grid

        X = X[ground_mask]
        Z = Z[ground_mask]
        height_diff = height_diff[ground_mask]

        height_weight = np.exp(-height_diff / 0.45)
        depth_weight = np.clip(20.0 / (Z + 5.0), 0.25, 1.0)
        weights = height_weight * depth_weight * 0.18

        col = ((X + self.size / 2) / self.res).astype(int)
        row = (Z / self.res).astype(int)

        mask = (row >= 0) & (row < self.N) & (col >= 0) & (col < self.N)
        np.add.at(self.grid, (row[mask], col[mask]), weights[mask])
        self.grid = np.clip(self.grid, 0.0, 1.0)
        return self.grid

    def add_detection_footprint(self, col, row, label, conf):
        if not (0 <= row < self.N and 0 <= col < self.N):
            return
        radius = 8 if label in ["truck", "bus"] else 5
        max_value = 0.75 + conf * 0.25
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                r, c = row + dy, col + dx
                if 0 <= r < self.N and 0 <= c < self.N:
                    dist = np.hypot(dx, dy)
                    falloff = np.exp(-dist**2 / (radius**2 * 1.4))
                    self.grid[r, c] = max(self.grid[r, c], max_value * falloff)

    def post_process(self):
        kernel = np.ones((3, 3), np.uint8)
        self.grid = cv2.morphologyEx(self.grid, cv2.MORPH_CLOSE, kernel)
        boost = np.linspace(1.30, 1.0, self.N)[:, None]
        self.grid = np.clip(self.grid * boost, 0.0, 1.0)

    def smooth(self, kernel_size=5, sigma=1.3):
        self.grid = cv2.GaussianBlur(self.grid, (kernel_size, kernel_size), sigma)
        self.grid = np.clip(self.grid, 0.0, 1.0)
        return self.grid