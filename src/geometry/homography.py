# src/geometry/homography.py
#
# Homography-based Image-to-BEV (Inverse Perspective Mapping)
#
# This is the "fast mode" alternative to the full depth pipeline.
# It warps the road surface directly from perspective view to top-down view
# using a planar homography derived from the camera intrinsics and height.
#
# Limitation: only valid for flat ground-plane points. Tall objects
# (cars, buildings) will be distorted / "stretched" in BEV.
#
# Usage:
#   ipm = InversePerspectiveMapping(K, camera_height=1.51, pitch_deg=0.0)
#   bev = ipm.warp(image, bev_width=400, bev_height=400, scale=10.0)

import cv2
import numpy as np


class InversePerspectiveMapping:
    """
    Inverse Perspective Mapping (IPM) using a planar ground homography.

    Assumes a flat ground plane and known camera intrinsics + height.
    The resulting BEV image shows the ground surface in a top-down view
    where 1 pixel = (1/scale) metres.
    """

    def __init__(self, K, camera_height=1.51, pitch_deg=0.0, roll_deg=0.0):
        """
        Args:
            K:              3x3 camera intrinsic matrix
            camera_height:  height of camera above ground (metres)
            pitch_deg:      camera pitch (positive = looking down, degrees)
            roll_deg:       camera roll (degrees, usually 0)
        """
        self.K = np.array(K, dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.camera_height = camera_height
        self.pitch = np.radians(pitch_deg)
        self.roll  = np.radians(roll_deg)

        # Build rotation matrix: camera frame → (level) camera frame
        # Pitch rotation around X axis
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cr, sr = np.cos(self.roll),  np.sin(self.roll)

        Rx = np.array([[1,  0,   0],
                       [0, cp, -sp],
                       [0, sp,  cp]])
        Rz = np.array([[ cr, -sr, 0],
                       [ sr,  cr, 0],
                       [  0,   0, 1]])
        self.R = Rx @ Rz

    def compute_homography(self, image_shape, bev_width, bev_height, scale=10.0):
        """
        Compute the 3x3 homography mapping image pixels → BEV pixels.

        Args:
            image_shape: (H, W) of the input image
            bev_width:   output BEV image width in pixels
            bev_height:  output BEV image height in pixels
            scale:       pixels per metre in the BEV output

        Returns:
            H_img2bev: 3x3 homography (for cv2.warpPerspective)
            H_bev2img: inverse
        """
        H_img, W_img = image_shape[:2]

        # --- Source points: bottom strip of the image (ground region) ---
        # Choose 4 points along the bottom of the image in a trapezoid
        # that corresponds to the visible road surface.
        margin_x = W_img * 0.1
        src_y_near = H_img * 0.75   # near edge of road (bottom)
        src_y_far  = H_img * 0.55   # far edge of road (horizon)

        src_pts = np.float32([
            [margin_x,         src_y_near],  # bottom-left
            [W_img - margin_x, src_y_near],  # bottom-right
            [W_img * 0.55,     src_y_far],   # top-right (far)
            [W_img * 0.45,     src_y_far],   # top-left (far)
        ])

        # --- Destination points: rectangle in BEV space ---
        # near_row = bottom of BEV (close to ego vehicle)
        # far_row  = top of BEV (far from ego vehicle)
        near_row = bev_height - 10
        far_row  = bev_height // 3
        left_col = bev_width * 0.2
        right_col = bev_width * 0.8

        dst_pts = np.float32([
            [left_col,  near_row],
            [right_col, near_row],
            [right_col, far_row],
            [left_col,  far_row],
        ])

        H_img2bev = cv2.getPerspectiveTransform(src_pts, dst_pts)
        H_bev2img = cv2.getPerspectiveTransform(dst_pts, src_pts)

        return H_img2bev, H_bev2img

    def warp(self, image, bev_width=400, bev_height=400, scale=10.0):
        """
        Warp the input image to a top-down BEV view.

        Args:
            image:      BGR input image
            bev_width:  output width (pixels)
            bev_height: output height (pixels)
            scale:      pixels per metre

        Returns:
            bev_image: warped top-down BGR image
            H:         3x3 homography used
        """
        H_mat, _ = self.compute_homography(
            image.shape, bev_width, bev_height, scale
        )
        bev = cv2.warpPerspective(image, H_mat, (bev_width, bev_height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        return bev, H_mat

    @classmethod
    def from_nuscenes(cls, K, camera_height=1.51):
        """Convenience constructor using nuScenes default camera tilt (~0°)."""
        return cls(K, camera_height=camera_height, pitch_deg=0.0)


def four_point_homography(src_pts, dst_pts):
    """
    Compute homography from 4 point correspondences.
    Thin wrapper around cv2.getPerspectiveTransform for clarity.

    src_pts, dst_pts: np.float32 arrays of shape (4, 2)
    """
    return cv2.getPerspectiveTransform(
        src_pts.astype(np.float32),
        dst_pts.astype(np.float32)
    )