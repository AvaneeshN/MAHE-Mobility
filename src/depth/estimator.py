

import torch
import cv2
import numpy as np


class DepthEstimator:
    def __init__(self, model_type="MiDaS_small", device=None):
        """
        Initialize MiDaS depth model

        model_type:
            - MiDaS_small (fast, lightweight)
            - DPT_Large (more accurate, slower)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "MiDaS_small":
            self.transform = transforms.small_transform
        else:
            self.transform = transforms.dpt_transform

    def predict(self, image_bgr):
        """
        Input:
            image_bgr: OpenCV image (H, W, 3)

        Output:
            depth_map: (H, W) numpy array
        """

        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Apply transform
        input_batch = self.transform(image_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        return depth_map

    def normalize_depth(self, depth_map):
        """
        Normalize for visualization
        """
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        return depth_norm.astype(np.uint8)