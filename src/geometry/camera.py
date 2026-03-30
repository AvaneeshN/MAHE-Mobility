# src/geometry/camera.py

import numpy as np


class Camera:
    def __init__(self, fx, fy, cx, cy, R=None, t=None):
        """
        fx, fy: focal lengths
        cx, cy: principal point
        R: rotation matrix (3x3)
        t: translation vector (3,)
        """

        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)

        self.K_inv = np.linalg.inv(self.K)

        # Default: camera aligned with world
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)

    def pixel_to_ray(self, u, v):
        """
        Convert pixel to 3D direction vector
        """

        pixel = np.array([u, v, 1.0])
        ray = self.K_inv @ pixel

        # Normalize
        ray = ray / np.linalg.norm(ray)

        return ray

    def unproject_pixel(self, u, v, depth):
        """
        Convert pixel + depth → 3D point (camera frame)
        """

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z])

    def camera_to_world(self, point_cam):
        """
        Convert camera coordinates → world coordinates
        """

        point_world = self.R.T @ (point_cam - self.t)

        return point_world