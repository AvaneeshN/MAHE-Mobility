import numpy as np


class OccupancyGrid:
    def __init__(self, size=40.0, resolution=0.1):  # 🔥 increased size
        self.size = size
        self.res = resolution
        self.N = int(size / resolution)
        self.grid = np.zeros((self.N, self.N))

    def world_to_grid(self, x, z):
        col = int((x + self.size / 2) / self.res)
        row = int(z / self.res)
        return col, row

    def fill(self, points):
        for pt in points:
            x, y, z = pt

            col, row = self.world_to_grid(x, z)

            if 0 <= row < self.N and 0 <= col < self.N:
                self.grid[row, col] = min(self.grid[row, col] + 0.2, 1.0)

        return self.grid


def filter_ground(points, min_height=-2.0, max_height=2.0):
    return points[
        (points[:, 1] > min_height) &
        (points[:, 1] < max_height)
    ]