# src/mapping/occupancy_grid.py

import numpy as np


class OccupancyGrid:
    def __init__(self, size=20.0, resolution=0.1):
        """
        size: meters (20x20 area)
        resolution: meters per cell
        """

        self.size = size
        self.res = resolution

        self.N = int(size / resolution)

        # Grid initialized to zero
        self.grid = np.zeros((self.N, self.N))

    def world_to_grid(self, x, z):
        """
        Convert world coords → grid indices
        """

        col = int((x + self.size / 2) / self.res)
        row = int(z / self.res)

        return col, row

    def fill(self, points):
        """
        Fill grid using point cloud
        """

        for pt in points:
            x, y, z = pt

            col, row = self.world_to_grid(x, z)

            if 0 <= row < self.N and 0 <= col < self.N:
                self.grid[row, col] += 1

        # Normalize
        self.grid = np.clip(self.grid, 0, 1)

        return self.grid


def filter_ground(points, height_thresh=2.0):
    """
    Keep only near-ground points
    """

    return points[np.abs(points[:, 1]) < height_thresh]