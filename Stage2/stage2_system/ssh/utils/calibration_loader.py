"""Calibration loader and epipolar rectification engine for binocular cameras."""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


class StereoCalibration:
    """Manages binocular calibration parameters and precomputes epipolar rectification maps."""

    def __init__(
        self,
        calibration_path: str | Path = "calibration/stereo_calibration_fixed.yml",
        target_size: tuple[int, int] = (1920, 1080),  # (width, height)
    ) -> None:
        self.calibration_path = Path(calibration_path)
        self.target_width, self.target_height = target_size
        self.K1: np.ndarray | None = None
        self.D1: np.ndarray | None = None
        self.K2: np.ndarray | None = None
        self.D2: np.ndarray | None = None
        self.R: np.ndarray | None = None
        self.T: np.ndarray | None = None  # in mm
        self.baseline_m: float = 0.0
        self.focal_length_px: float = 0.0

        self.map_left_x: np.ndarray | None = None
        self.map_left_y: np.ndarray | None = None
        self.map_right_x: np.ndarray | None = None
        self.map_right_y: np.ndarray | None = None
        self.P1: np.ndarray | None = None
        self.P2: np.ndarray | None = None
        self.Q: np.ndarray | None = None

        self._load_and_init()

    def _load_and_init(self) -> None:
        if not self.calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found at: {self.calibration_path}")

        storage = cv2.FileStorage(str(self.calibration_path), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise IOError(f"Could not open OpenCV calibration file: {self.calibration_path}")

        matrices = {}
        for key in ("K1", "D1", "K2", "D2", "R", "T"):
            node = storage.getNode(key)
            if node.empty():
                storage.release()
                raise ValueError(f"Calibration file missing required matrix node: {key}")
            matrices[key] = node.mat().astype(np.float64)
        storage.release()

        self.K1 = matrices["K1"]
        self.D1 = matrices["D1"]
        self.K2 = matrices["K2"]
        self.D2 = matrices["D2"]
        self.R = matrices["R"]
        self.T = matrices["T"].reshape(3, 1)

        # Baseline: T translation vector is in millimetres in the YAML file
        norm_t_mm = float(np.linalg.norm(self.T))
        self.baseline_m = norm_t_mm / 1000.0
        if self.baseline_m <= 0:
            raise ValueError(f"Calculated non-positive baseline from T: {self.T}")

        # Compute scaling factors if capture resolution differs from native 1080p calibration
        calib_w, calib_h = 1920, 1080
        sx = self.target_width / calib_w
        sy = self.target_height / calib_h
        if abs(sx - sy) > 0.01:
            print(
                f"[Calibration] Warning: non-isotropic aspect ratio scaling: "
                f"sx={sx:.3f}, sy={sy:.3f}. Epipolar lines may be distorted!"
            )

        k1_scaled = self.K1.copy()
        k1_scaled[0, 0] *= sx
        k1_scaled[0, 2] *= sx
        k1_scaled[1, 1] *= sy
        k1_scaled[1, 2] *= sy

        k2_scaled = self.K2.copy()
        k2_scaled[0, 0] *= sx
        k2_scaled[0, 2] *= sx
        k2_scaled[1, 1] *= sy
        k2_scaled[1, 2] *= sy

        image_size = (self.target_width, self.target_height)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            k1_scaled,
            self.D1,
            k2_scaled,
            self.D2,
            image_size,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            newImageSize=image_size,
        )

        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            k1_scaled, self.D1, R1, P1, image_size, cv2.CV_32FC1
        )
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            k2_scaled, self.D2, R2, P2, image_size, cv2.CV_32FC1
        )

        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        self.focal_length_px = float(P1[0, 0])

        print(
            f"[Calibration] Loaded successfully. "
            f"Baseline: {self.baseline_m * 1000.0:.2f} mm | "
            f"Rectified fx: {self.focal_length_px:.1f} px | "
            f"Resolution: {self.target_width}x{self.target_height}"
        )
        self._cached_maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {
            (self.target_width, self.target_height): (
                self.map_left_x,
                self.map_left_y,
                self.map_right_x,
                self.map_right_y,
                self.focal_length_px,
            )
        }

    def get_maps_for_size(
        self, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute and cache rectification maps for any incoming stream resolution."""
        if (width, height) in self._cached_maps:
            return self._cached_maps[(width, height)]

        calib_w, calib_h = 1920, 1080
        sx = width / calib_w
        sy = height / calib_h

        k1_scaled = self.K1.copy()
        k1_scaled[0, 0] *= sx
        k1_scaled[0, 2] *= sx
        k1_scaled[1, 1] *= sy
        k1_scaled[1, 2] *= sy

        k2_scaled = self.K2.copy()
        k2_scaled[0, 0] *= sx
        k2_scaled[0, 2] *= sx
        k2_scaled[1, 1] *= sy
        k2_scaled[1, 2] *= sy

        image_size = (width, height)
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            k1_scaled,
            self.D1,
            k2_scaled,
            self.D2,
            image_size,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            newImageSize=image_size,
        )

        map_l_x, map_l_y = cv2.initUndistortRectifyMap(
            k1_scaled, self.D1, R1, P1, image_size, cv2.CV_32FC1
        )
        map_r_x, map_r_y = cv2.initUndistortRectifyMap(
            k2_scaled, self.D2, R2, P2, image_size, cv2.CV_32FC1
        )
        focal_px = float(P1[0, 0])
        self._cached_maps[(width, height)] = (map_l_x, map_l_y, map_r_x, map_r_y, focal_px)
        return self._cached_maps[(width, height)]

    def get_focal_length(self, width: int | None = None, height: int | None = None) -> float:
        """Return the rectified focal length in pixels for the given frame dimensions."""
        if width is None or height is None or (width == self.target_width and height == self.target_height):
            return self.focal_length_px
        _, _, _, _, focal_px = self.get_maps_for_size(width, height)
        return focal_px

    def rectify_pair(
        self,
        left_raw: np.ndarray,
        right_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply undistortion and epipolar rectification maps to left and right frames."""
        h, w = left_raw.shape[:2]
        if w == self.target_width and h == self.target_height:
            map_lx, map_ly = self.map_left_x, self.map_left_y
            map_rx, map_ry = self.map_right_x, self.map_right_y
        else:
            map_lx, map_ly, map_rx, map_ry, _ = self.get_maps_for_size(w, h)

        left_rect = cv2.remap(left_raw, map_lx, map_ly, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, map_rx, map_ry, cv2.INTER_LINEAR)
        return left_rect, right_rect

