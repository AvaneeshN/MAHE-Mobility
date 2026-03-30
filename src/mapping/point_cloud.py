# src/mapping/point_cloud.py

import numpy as np


def depth_to_point_cloud(depth_map, camera):
    """
    Convert full depth map → point cloud

    depth_map: (H, W)
    camera: Camera object
    """

    H, W = depth_map.shape

    fx = camera.K[0, 0]
    fy = camera.K[1, 1]
    cx = camera.K[0, 2]
    cy = camera.K[1, 2]

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth_map

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Stack into (H, W, 3)
    points = np.stack([X, Y, Z], axis=-1)

    # Flatten → (N, 3)
    points = points.reshape(-1, 3)

    # Remove invalid depth
    points = points[points[:, 2] > 0]

    return points