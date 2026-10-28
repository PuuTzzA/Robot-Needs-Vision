"""Traditional OpenCV Semi-Global Block Matching (StereoSGBM) Estimator."""

from __future__ import annotations
import cv2
import numpy as np

from .base import BaseStereoEstimator


class TraditionalSGBMEstimator(BaseStereoEstimator):
    """Traditional stereo matching using OpenCV's StereoSGBM algorithm with optional WLS filter.
    
    This serves as the classical baseline for Stage 1. It demonstrates the
    inherent limitation of local intensity/census correlation on untextured
    planar surfaces (white tables, cardboard boxes), yielding black void pixels (holes).
    """

    def __init__(
        self,
        min_disparity: int = 0,
        num_disparities: int = 16 * 10,  # 160 px search range covers close indoor objects down to ~0.7m
        block_size: int = 5,
        uniqueness_ratio: int = 5,  # Slightly lower ratio prevents excessive void rejection on low-texture walls
        speckle_window_size: int = 50,
        speckle_range: int = 16,
        disp_12_max_diff: int = 1,
        use_wls_filter: bool = True,
        wls_lambda: float = 8000.0,
        wls_sigma: float = 1.5,
        swap_cameras: bool = False,
    ) -> None:
        self._min_disp = min_disparity
        self._num_disp = num_disparities
        self._block_size = block_size
        self._uniqueness = uniqueness_ratio
        self._speckle_window = speckle_window_size
        self._speckle_range = speckle_range
        self._disp12_max_diff = disp_12_max_diff
        self._use_wls = use_wls_filter
        self._wls_lambda = wls_lambda
        self._wls_sigma = wls_sigma
        self.swap_cameras = swap_cameras

        self._matcher_left: cv2.StereoSGBM | None = None
        self._matcher_right: cv2.StereoMatcher | None = None
        self._wls_filter = None
        self._has_ximgproc = False

    @property
    def name(self) -> str:
        return "Traditional StereoSGBM"

    def initialize(self) -> None:
        # P1 and P2 control disparity smoothness across scanlines
        channels = 1
        p1 = 8 * channels * (self._block_size ** 2)
        p2 = 32 * channels * (self._block_size ** 2)

        self._matcher_left = cv2.StereoSGBM_create(
            minDisparity=self._min_disp,
            numDisparities=self._num_disp,
            blockSize=self._block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=self._disp12_max_diff,
            uniquenessRatio=self._uniqueness,
            speckleWindowSize=self._speckle_window,
            speckleRange=self._speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        if self._use_wls and hasattr(cv2, "ximgproc"):
            try:
                self._matcher_right = cv2.ximgproc.createRightMatcher(self._matcher_left)
                self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(self._matcher_left)
                self._wls_filter.setLambda(self._wls_lambda)
                self._wls_filter.setSigmaColor(self._wls_sigma)
                self._has_ximgproc = True
            except Exception as e:
                print(f"[TraditionalSGBM] WLS filter init failed ({e}); using raw SGBM.")
                self._has_ximgproc = False

    def estimate_disparity(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
    ) -> np.ndarray:
        if self._matcher_left is None:
            self.initialize()

        gray_left = (
            cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            if left_rectified.ndim == 3
            else left_rectified
        )
        gray_right = (
            cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
            if right_rectified.ndim == 3
            else right_rectified
        )

        if self.swap_cameras:
            gray_left, gray_right = gray_right, gray_left

        if self._has_ximgproc and self._matcher_right is not None and self._wls_filter is not None:
            disp_l = self._matcher_left.compute(gray_left, gray_right)
            disp_r = self._matcher_right.compute(gray_right, gray_left)
            filtered_disp = self._wls_filter.filter(disp_l, gray_left, disparity_map_right=disp_r)
            disparity = filtered_disp.astype(np.float32) / 16.0
        else:
            disp_raw = self._matcher_left.compute(gray_left, gray_right)
            disparity = disp_raw.astype(np.float32) / 16.0

        # Mark negative disparities and invalid matches as NaN
        invalid_mask = disparity <= self._min_disp
        disparity[invalid_mask] = np.nan
        return disparity

