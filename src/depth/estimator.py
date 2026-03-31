# src/depth/estimator.py

import torch
import cv2
import numpy as np


class DepthEstimator:
    def __init__(self, model_type="MiDaS_small", scale_factor=20.0,
                 min_depth=1.0, max_depth=50.0, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scale_factor = scale_factor
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "MiDaS_small":
            self.transform = transforms.small_transform
        else:
            self.transform = transforms.dpt_transform

    def predict(self, image_bgr):
        """Returns metric depth map in metres. Shape: [H, W]"""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(image_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        # MiDaS gives inverse depth (disparity-like) — invert to get depth
        depth = self.scale_factor / (depth + 1e-6)
        depth = np.clip(depth, self.min_depth, self.max_depth)

        return depth

    def normalize_depth(self, depth):
        """
        Normalize depth map to 0-255 uint8 for saving/display.
        This is what test_depth.py was trying to call.
        """
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        return depth_norm.astype(np.uint8)

    def visualize(self, depth):
        """Returns a colourmap-applied BGR image for nice visualization."""
        depth_norm = self.normalize_depth(depth)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)