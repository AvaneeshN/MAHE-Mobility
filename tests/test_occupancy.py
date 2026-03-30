import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.depth.estimator import DepthEstimator
from src.geometry.camera import Camera
from src.mapping.point_cloud import depth_to_point_cloud
from src.mapping.occupancy_grid import OccupancyGrid, filter_ground


# Load image
image = cv2.imread("data/raw/sample.jpg")

# Depth estimation (NOW METRIC-SCALED)
depth_estimator = DepthEstimator()
depth = depth_estimator.predict(image)

# Camera
cam = Camera(fx=800, fy=800, cx=640, cy=360)

# Convert to point cloud
points = depth_to_point_cloud(depth, cam)

# 🔥 Filter depth range
points = points[(points[:, 2] > 1) & (points[:, 2] < 30)]

# 🔥 Ground filtering
ground_points = filter_ground(points)

# Build occupancy grid
grid = OccupancyGrid(size=20, resolution=0.1)
bev = grid.fill(ground_points)

# 🔥 Smooth BEV
bev = cv2.GaussianBlur(bev, (5, 5), 0)

# Visualization
plt.figure(figsize=(6, 6))
plt.imshow(bev, cmap='inferno', origin='lower')
plt.title("Final BEV Occupancy Grid")
plt.xlabel("X (cells)")
plt.ylabel("Forward (cells)")

plt.savefig("outputs/bev_final.png", dpi=150)
plt.show()

print("Final BEV saved to outputs/bev_final.png")