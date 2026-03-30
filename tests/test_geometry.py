import numpy as np
from src.geometry.camera import Camera

# Create camera
cam = Camera(
    fx=800,
    fy=800,
    cx=640,
    cy=360
)

# Test pixel
u, v = 640, 360  # center pixel
depth = 10.0     # 10 meters

# Unproject
point_3d = cam.unproject_pixel(u, v, depth)

print("3D Point:", point_3d)