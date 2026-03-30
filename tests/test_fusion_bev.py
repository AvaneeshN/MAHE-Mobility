import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.depth.estimator import DepthEstimator
from src.geometry.camera import Camera
from src.mapping.point_cloud import depth_to_point_cloud
from src.mapping.occupancy_grid import OccupancyGrid, filter_ground
from src.perception.detector import ObjectDetector


image = cv2.imread("data/raw/sample.jpg")

depth_estimator = DepthEstimator()
detector = ObjectDetector()

cam = Camera(fx=800, fy=800, cx=640, cy=360)

# ===== DEPTH =====
depth = depth_estimator.predict(image)

# ===== ROAD =====
points = depth_to_point_cloud(depth, cam)
points = points[(points[:, 2] > 1) & (points[:, 2] < 40)]

ground_points = filter_ground(points)

grid = OccupancyGrid(size=40, resolution=0.1)
bev = grid.fill(ground_points)

bev = cv2.GaussianBlur(bev, (5, 5), 0)

# ===== DETECTIONS =====
detections = detector.detect(image)

xs, ys = [], []

for det in detections:
    x1, y1, x2, y2, label, conf = det

    if label not in ["car", "truck", "bus", "motorcycle"]:
        continue

    u = int((x1 + x2) / 2)
    v = int(y2)

    u = np.clip(u, 0, depth.shape[1] - 1)
    v = np.clip(v, 0, depth.shape[0] - 1)

    Z = depth[v, u]

    # 🔥 DEPTH BOOST
    Z = Z ** 1.3

    if Z <= 0 or Z > 50:
        continue

    point_3d = cam.unproject_pixel(u, v, Z)
    x, y, z = point_3d

    x = x * 1.5

    if abs(x) > 40 or z > 40:
        continue

    col, row = grid.world_to_grid(x, z)

    if 0 <= row < grid.N and 0 <= col < grid.N:
        xs.append(col)
        ys.append(row)

# ===== VISUALIZATION =====
plt.figure(figsize=(6, 6))

plt.imshow(bev, cmap='gray', origin='lower')

plt.scatter(xs, ys, c='red', s=50, label='Vehicles')

plt.title("Final BEV (Road + Objects)")
plt.legend()

plt.savefig("outputs/bev_fusion_final.png")
plt.show()