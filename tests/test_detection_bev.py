import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.depth.estimator import DepthEstimator
from src.geometry.camera import Camera
from src.perception.detector import ObjectDetector
from src.mapping.occupancy_grid import OccupancyGrid


image = cv2.imread("data/raw/sample.jpg")

depth_estimator = DepthEstimator()
detector = ObjectDetector()

cam = Camera(fx=800, fy=800, cx=640, cy=360)

depth = depth_estimator.predict(image)

detections = detector.detect(image)

grid = OccupancyGrid(size=40, resolution=0.1)

valid_points = 0

for det in detections:
    x1, y1, x2, y2, label, conf = det

    if label not in ["car", "truck", "bus", "motorcycle", "person"]:
        continue

    u = int((x1 + x2) / 2)
    v = int(y2)

    u = np.clip(u, 0, depth.shape[1] - 1)
    v = np.clip(v, 0, depth.shape[0] - 1)

    Z = depth[v, u]

    # 🔥 NON-LINEAR DEPTH BOOST
    Z = Z ** 1.3

    if Z <= 0 or Z > 50:
        continue

    point_3d = cam.unproject_pixel(u, v, Z)
    x, y, z = point_3d

    # 🔥 SPREAD X AXIS
    x = x * 1.5

    if abs(x) > 40 or z > 40:
        continue

    col, row = grid.world_to_grid(x, z)

    # 🔥 BIGGER BLOBS
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            r = row + dy
            c = col + dx
            if 0 <= r < grid.N and 0 <= c < grid.N:
                grid.grid[r, c] = 1.0

    valid_points += 1

print("Valid BEV points:", valid_points)

ys, xs = np.where(grid.grid > 0)

plt.figure(figsize=(6, 6))
plt.imshow(grid.grid, cmap='inferno', origin='lower')
plt.scatter(xs, ys, c='cyan', s=10)

plt.title("Object-level BEV (Final Improved)")
plt.savefig("outputs/bev_objects_final.png")
plt.show()