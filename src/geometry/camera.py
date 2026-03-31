# src/geometry/camera.py

import numpy as np


class Camera:
    def __init__(self, fx, fy, cx, cy, R=None, t=None):
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

        self.K_inv = np.linalg.inv(self.K)
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)

    def pixel_to_ray(self, u, v):
        """Single pixel → 3D direction ray."""
        pixel = np.array([u, v, 1.0])
        ray = self.K_inv @ pixel
        return ray / np.linalg.norm(ray)

    def unproject_pixel(self, u, v, depth):
        """Single pixel + depth → 3D point in camera frame."""
        X = (u - self.K[0, 2]) * depth / self.K[0, 0]
        Y = (v - self.K[1, 2]) * depth / self.K[1, 1]
        return np.array([X, Y, depth])

    def camera_to_world(self, points_cam):
        """
        Camera frame → world frame.
        Accepts BOTH single point [3] and batch [N, 3].
        """
        if points_cam.ndim == 1:
            # Single point
            return self.R.T @ (points_cam - self.t)
        else:
            # Batch: [N, 3] — vectorized, fast
            return (self.R.T @ (points_cam - self.t).T).T

    @classmethod
    def from_nuscenes(cls, cam_intrinsic, cam_rotation, cam_translation):
        """
        Build Camera directly from nuScenes calibration data.

        cam_intrinsic:   3x3 list (from nuScenes)
        cam_rotation:    quaternion [w, x, y, z] (from nuScenes)
        cam_translation: [x, y, z] (from nuScenes)
        """
        K = np.array(cam_intrinsic)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Convert quaternion → rotation matrix
        R = quaternion_to_rotation(cam_rotation)
        t = np.array(cam_translation)

        return cls(fx=fx, fy=fy, cx=cx, cy=cy, R=R, t=t)


def quaternion_to_rotation(q):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    nuScenes stores quaternions as [w, x, y, z].
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R