import cv2
from src.depth.estimator import DepthEstimator
from src.geometry.camera import Camera
from src.mapping.point_cloud import depth_to_point_cloud

# Load image
image = cv2.imread("data/raw/sample.jpg")

# Depth
depth_estimator = DepthEstimator()
depth = depth_estimator.predict(image)

# Camera
cam = Camera(fx=800, fy=800, cx=640, cy=360)

# Point cloud
points = depth_to_point_cloud(depth, cam)

print("Point cloud shape:", points.shape)
print("Sample points:", points[:5])