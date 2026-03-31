# src/depth/estimator.py  — v3: Affine-invariant calibration
#
# WHY THE PREVIOUS FIX STILL PRODUCED BINARY DEPTH:
#   MiDaS outputs disparity in an *affine-invariant* space, meaning the
#   output is only defined up to an unknown scale AND shift:
#       disparity_midas = s * (1/Z_true) + t
#   Simply inverting (1/raw) doesn't help when the shift 't' is large —
#   the result is still dominated by the offset, not the true depth structure.
#   On urban scenes with a large truck close-up, MiDaS saturates the near
#   range, making the sky the only "far" region → binary blue/yellow.
#
# FIX — Two-anchor affine calibration using camera geometry:
#   1. Get the raw MiDaS disparity map (don't invert yet).
#   2. Sample two ground-plane "anchor" rows at different image heights.
#      For each row v, the true depth is:  Z = camera_height * fy / (v - cy)
#      This gives us two known (disparity, true_depth) pairs.
#   3. Solve for affine coefficients:  disparity = s/Z + t
#      → s = (d1 - d2) / (1/Z1 - 1/Z2),  t = d1 - s/Z1
#   4. Convert full map:  Z = s / (disparity - t),  clipped to [min, max].
#   This properly handles the scale+shift ambiguity.

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

    def _raw_disparity(self, image_bgr):
        """Run MiDaS and return raw disparity map (large = near). Shape [H, W]."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(image_rgb).to(self.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=image_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return pred.cpu().numpy().astype(np.float32)

    def predict(self, image_bgr):
        """
        Returns a relative depth map stretched to [min_depth, max_depth].
        Useful for visualization even without camera calibration.
        Near = small value, Far = large value.
        """
        raw = self._raw_disparity(image_bgr)

        # Robust percentile stretch of the raw disparity
        lo = float(np.percentile(raw, 5))
        hi = float(np.percentile(raw, 95))
        if hi - lo < 1e-4:
            return np.full(raw.shape, (self.min_depth + self.max_depth) / 2,
                           dtype=np.float32)

        # Normalise disparity to [0,1], then invert so near=small, far=large
        disp_norm = (raw - lo) / (hi - lo)
        disp_norm = np.clip(disp_norm, 0, 1)
        # Invert: high disparity (near) → low depth value
        depth = self.min_depth + (1.0 - disp_norm) * (self.max_depth - self.min_depth)
        return depth.astype(np.float32)

    def predict_with_scale(self, image_bgr, K, camera_height=1.51):
        """
        Predict metric depth using two-anchor affine calibration.

        Uses two ground-plane rows at known image heights to solve for
        the scale AND shift of MiDaS's affine-invariant disparity output.

        Returns: (depth_metric, scale_s)
        """
        raw = self._raw_disparity(image_bgr)
        H, W = raw.shape
        fy = float(K[1, 1])
        cy = float(K[1, 2])

        # --- Choose two anchor rows in the ground-plane region ---
        # Avoid the very bottom edge (may be hood/bonnet) and horizon
        # Use rows at 65% and 85% of image height — reliably ground in nuScenes
        v1 = int(0.65 * H)
        v2 = int(0.85 * H)

        # True depth at each anchor row from camera-height geometry:
        #   Z = camera_height * fy / (v - cy)
        Z1 = camera_height * fy / max(v1 - cy, 1.0)
        Z2 = camera_height * fy / max(v2 - cy, 1.0)
        Z1 = float(np.clip(Z1, self.min_depth, self.max_depth))
        Z2 = float(np.clip(Z2, self.min_depth, self.max_depth))

        # Sample median disparity in a centre strip at each anchor row
        u_lo = int(0.40 * W)
        u_hi = int(0.60 * W)
        strip_h = 8   # average over 8 rows for robustness

        d1 = float(np.median(raw[max(0, v1-strip_h):v1+strip_h, u_lo:u_hi]))
        d2 = float(np.median(raw[max(0, v2-strip_h):v2+strip_h, u_lo:u_hi]))

        # Solve affine system:  d = s*(1/Z) + t
        inv_Z1, inv_Z2 = 1.0 / Z1, 1.0 / Z2

        denom = inv_Z1 - inv_Z2
        if abs(denom) < 1e-6 or abs(d1 - d2) < 1e-4:
            # Fallback: simple scale-only (t=0)
            s = d1 * Z1
            t = 0.0
        else:
            s = (d1 - d2) / denom
            t = d1 - s * inv_Z1

        # Convert full disparity map to metric depth:  Z = s / (d - t)
        denom_map = raw - t
        # Avoid divide-by-zero and sign flips
        denom_map = np.where(np.abs(denom_map) < 1e-4,
                             np.sign(denom_map + 1e-8) * 1e-4,
                             denom_map)
        depth = s / denom_map

        # If s is negative (MiDaS sometimes inverts sign), flip
        if s < 0:
            depth = -depth

        depth = np.clip(depth, self.min_depth, self.max_depth).astype(np.float32)

        # Final sanity check: if median depth is implausibly small, fall back
        med = float(np.median(depth))
        if med < self.min_depth or med > self.max_depth * 0.9:
            depth = self.predict(image_bgr)
            s = 1.0

        return depth, float(s)

    def normalize_depth(self, depth):
        """Normalize depth map to 0-255 uint8 for saving/display."""
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        return depth_norm.astype(np.uint8)

    def visualize(self, depth):
        """Returns a colourmap-applied BGR image for visualization."""
        depth_norm = self.normalize_depth(depth)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)