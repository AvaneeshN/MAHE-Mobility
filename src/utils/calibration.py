# src/utils/calibration.py
#
# Camera calibration utilities.
# Supports checkerboard calibration AND nuScenes intrinsic extraction.

import cv2
import numpy as np
import glob
import os
import yaml


def calibrate_from_checkerboard(image_dir, pattern=(8, 6),
                                  square_size_m=0.025, save_path=None):
    """
    Run OpenCV checkerboard calibration on a folder of images.

    Args:
        image_dir:      directory containing calibration images (jpg/png)
        pattern:        (cols, rows) of INNER corners, e.g. (8, 6)
        square_size_m:  physical size of each square in metres
        save_path:      if given, save calibration YAML here

    Returns:
        K:       3x3 intrinsic matrix (float64)
        dist:    distortion coefficients
        rms:     reprojection RMS error (lower is better, <1.0 is good)
    """
    obj_p = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    obj_p *= square_size_m

    obj_points = []   # 3D points in real world
    img_points = []   # 2D points in image plane

    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                    glob.glob(os.path.join(image_dir, "*.png")))

    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    img_shape = None
    found_count = 0

    for fpath in images:
        img = cv2.imread(fpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (W, H)

        ret, corners = cv2.findChessboardCorners(gray, pattern, None)
        if ret:
            obj_points.append(obj_p)
            corners_sub = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            img_points.append(corners_sub)
            found_count += 1

    if found_count < 5:
        raise RuntimeError(
            f"Only {found_count} checkerboard patterns found — need at least 5. "
            "Check image quality and pattern size."
        )

    print(f"Found checkerboard in {found_count}/{len(images)} images.")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )
    print(f"Calibration RMS: {rms:.4f} px")

    if save_path:
        data = {
            "camera_matrix": K.tolist(),
            "distortion": dist.tolist(),
            "rms": float(rms),
            "pattern": list(pattern),
            "square_size_m": square_size_m,
        }
        with open(save_path, "w") as f:
            yaml.dump(data, f)
        print(f"Calibration saved → {save_path}")

    return K, dist, rms


def load_calibration(yaml_path):
    """Load K and dist from a saved calibration YAML."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["distortion"], dtype=np.float64)
    return K, dist


def undistort_image(image, K, dist):
    """Undistort an image using calibration parameters."""
    h, w = image.shape[:2]
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, K, dist, None, K_new)
    x, y, w, h = roi
    return undistorted[y:y+h, x:x+w], K_new


def nuscenes_intrinsics_to_K(cam_intrinsic):
    """
    Convert nuScenes camera_intrinsic (3x3 list) to numpy K matrix.

    nuScenes calibrated_sensor.camera_intrinsic is already a 3x3
    list of lists: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    return np.array(cam_intrinsic, dtype=np.float64)


def print_K(K, name="Camera"):
    """Pretty-print an intrinsic matrix."""
    print(f"\n{name} intrinsics K:")
    print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}")
    print(f"  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"  K =\n{K}")