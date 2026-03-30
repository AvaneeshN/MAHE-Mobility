import torch
import cv2
import numpy as np


class DepthEstimator:
    def __init__(self, model_type="MiDaS_small", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "MiDaS_small":
            self.transform = transforms.small_transform
        else:
            self.transform = transforms.dpt_transform

    def predict(self, image_bgr):
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

        depth = prediction.cpu().numpy()

        # 🔥 IMPROVED DEPTH SCALING
        depth = 1.0 / (depth + 1e-6)
        depth = depth * 20

        depth = np.clip(depth, 1.0, 50.0)

        return depth

    def visualize(self, depth):
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)