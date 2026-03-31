# src/data/nuscenes_loader.py

import os
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes


class NuScenesLoader:
    """
    Loads images, depth hints, and calibration data from nuScenes v1.0-mini.
    Drop-in data source for the BEV pipeline.
    """

    # nuScenes has 6 cameras — we focus on front camera for now
    CAMERA = "CAM_FRONT"

    def __init__(self, dataroot="data/nuscenes", version="v1.0-mini"):
        print(f"Loading nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.samples = self.nusc.sample
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def get_sample(self, index):
        """
        Returns a dict with everything needed for one BEV frame:
        {
            'image':      BGR image (H x W x 3),
            'intrinsic':  3x3 K matrix,
            'rotation':   3x3 R matrix (camera → world),
            'translation': [x, y, z] (camera position in world),
            'sample_token': str
        }
        """
        sample = self.samples[index]

        # Get front camera data
        cam_token = sample["data"][self.CAMERA]
        cam_data = self.nusc.get("sample_data", cam_token)

        # Load image
        img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
        image = cv2.imread(img_path)

        # Get calibration
        calib_token = cam_data["calibrated_sensor_token"]
        calib = self.nusc.get("calibrated_sensor", calib_token)

        intrinsic = np.array(calib["camera_intrinsic"])  # 3x3

        # Quaternion → rotation matrix
        q = calib["rotation"]   # [w, x, y, z]
        R = self._quat_to_rot(q)

        translation = np.array(calib["translation"])  # [x, y, z]

        return {
            "image":       image,
            "intrinsic":   intrinsic,
            "rotation":    R,
            "translation": translation,
            "sample_token": sample["token"],
        }

    def get_sequential_samples(self, start_index=0, n_frames=10):
        """
        Get N consecutive frames from a scene for temporal fusion.
        Returns list of sample dicts.
        """
        frames = []
        sample = self.samples[start_index]

        for _ in range(n_frames):
            idx = self.nusc.sample.index(sample)
            frames.append(self.get_sample(idx))

            # Move to next sample in scene
            if sample["next"] == "":
                break
            sample = self.nusc.get("sample", sample["next"])

        return frames

    def _quat_to_rot(self, q):
        """Quaternion [w, x, y, z] → 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2+z**2),   2*(x*y - z*w),   2*(x*z + y*w)],
            [  2*(x*y + z*w), 1 - 2*(x**2+z**2),   2*(y*z - x*w)],
            [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x**2+y**2)]
        ])