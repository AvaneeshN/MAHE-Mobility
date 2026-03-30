import cv2
from src.depth.estimator import DepthEstimator

# Load image
image = cv2.imread("data/raw/sample.jpg")  # put any image here

estimator = DepthEstimator()

depth = estimator.predict(image)
depth_vis = estimator.normalize_depth(depth)

cv2.imwrite("outputs/depth_map.jpg", depth_vis)

print("Depth map saved to outputs/depth_map.jpg")