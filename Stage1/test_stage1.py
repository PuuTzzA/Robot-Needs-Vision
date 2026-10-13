"""Automated unit and integration test suite for Stage 1 Stereo Vision."""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

# Ensure Stage1 is in python path
stage1_dir = Path(__file__).resolve().parent
if str(stage1_dir) not in sys.path:
    sys.path.insert(0, str(stage1_dir))

from utils import (
    StereoCalibration,
    disparity_to_depth,
    colorize_disparity,
    colorize_depth,
    attach_depth_legend,
    calculate_compactness,
    sample_roi_depth,
)
from estimators import TraditionalSGBMEstimator, DeepCREStereoEstimator


def test_calibration() -> None:
    print("\n--- [Test 1] Testing Stereo Calibration & Rectification ---")
    calib_path = stage1_dir / "calibration" / "stereo_calibration_fixed.yml"
    assert calib_path.exists(), f"Calibration file missing at {calib_path}"

    calib = StereoCalibration(calibration_path=calib_path, target_size=(1920, 1080))
    assert calib.K1 is not None and calib.K1.shape == (3, 3)
    assert calib.K2 is not None and calib.K2.shape == (3, 3)
    assert calib.baseline_m > 0.05 and calib.baseline_m < 0.07, f"Unexpected baseline: {calib.baseline_m}m"
    assert calib.focal_length_px > 3000, f"Unexpected focal length: {calib.focal_length_px}px"
    assert calib.map_left_x is not None and calib.map_left_y is not None
    assert calib.map_right_x is not None and calib.map_right_y is not None

    print(f"✓ Baseline validated: {calib.baseline_m * 1000.0:.2f} mm")
    print(f"✓ Rectified fx validated: {calib.focal_length_px:.1f} px")
    print(f"✓ Rectification maps generated successfully: {calib.map_left_x.shape}")


def test_traditional_sgbm() -> None:
    print("\n--- [Test 2] Testing Traditional StereoSGBM Estimator ---")
    import cv2
    l_path = stage1_dir / "test_data" / "cones" / "im2.png"
    r_path = stage1_dir / "test_data" / "cones" / "im6.png"
    assert l_path.exists() and r_path.exists()

    left = cv2.imread(str(l_path))
    right = cv2.imread(str(r_path))

    estimator = TraditionalSGBMEstimator()
    estimator.initialize()
    assert estimator.name == "Traditional StereoSGBM"

    disparity = estimator.estimate_disparity(left, right)
    assert disparity.shape == left.shape[:2]
    assert disparity.dtype == np.float32

    compactness = calculate_compactness(disparity)
    print(f"✓ SGBM disparity computed. Shape: {disparity.shape}")
    print(f"✓ SGBM Compactness: {compactness:.1f}% (demonstrates classical voids/holes)")


def test_depth_conversion_and_legend() -> None:
    print("\n--- [Test 3] Testing Metric Depth Conversion & Color Legend ---")
    h, w = 480, 640
    dummy_disp = np.full((h, w), 50.0, dtype=np.float32)
    # Inject some holes/voids
    dummy_disp[:50, :50] = np.nan

    fx = 3700.0
    baseline_m = 0.05952
    depth = disparity_to_depth(dummy_disp, fx, baseline_m, min_depth=0.2, max_depth=10.0)

    expected_z = (fx * baseline_m) / 50.0  # 4.404 m
    valid_depths = depth[np.isfinite(depth)]
    assert np.allclose(valid_depths, expected_z, atol=1e-3), f"Depth mismatch: {valid_depths[0]} vs {expected_z}"

    compactness = calculate_compactness(depth)
    assert compactness < 100.0 and compactness > 90.0

    med_z, std_z, valid_ratio = sample_roi_depth(depth, 320, 240, box_radius=15)
    assert np.isclose(med_z, expected_z, atol=1e-3)
    assert std_z == 0.0

    # Colorize and attach legend
    depth_vis = colorize_depth(depth, min_depth=0.4, max_depth=5.0)
    assert depth_vis.shape == (h, w, 3)
    with_legend = attach_depth_legend(depth_vis, min_depth=0.4, max_depth=5.0, legend_width=70)
    assert with_legend.shape == (h, w + 70, 3)
    print(f"✓ Depth calculation verified: Z = {expected_z:.3f} m")
    print(f"✓ Color legend appended successfully: final shape {with_legend.shape}")


def test_deep_crestereo() -> None:
    print("\n--- [Test 4] Testing Deep CREStereo (CUDA/ONNX) ---")
    import cv2
    model_path = stage1_dir / "models" / "crestereo_embedded.onnx"
    assert model_path.exists(), f"Model file missing at {model_path}"

    l_path = stage1_dir / "test_data" / "cones" / "im2.png"
    r_path = stage1_dir / "test_data" / "cones" / "im6.png"
    left = cv2.imread(str(l_path))
    right = cv2.imread(str(r_path))

    estimator = DeepCREStereoEstimator(model_path=model_path)
    estimator.initialize()
    print(f"✓ Initialized estimator: {estimator.name}")

    disparity = estimator.estimate_disparity(left, right)
    assert disparity.shape == left.shape[:2]
    assert disparity.dtype == np.float32

    # Measure compactness inside the active region
    valid_mask = np.isfinite(disparity) & (disparity > 0)
    assert np.any(valid_mask), "Disparity should contain valid values"
    print(f"✓ CREStereo disparity computed successfully.")
    print(f"✓ Disparity range: min={np.nanmin(disparity):.1f}px, max={np.nanmax(disparity):.1f}px")


if __name__ == "__main__":
    print("==================================================")
    print("       STAGE 1 STEREO VISION TEST SUITE           ")
    print("==================================================")
    test_calibration()
    test_traditional_sgbm()
    test_depth_conversion_and_legend()
    test_deep_crestereo()
    print("\n==================================================")
    print("       ALL STAGE 1 TESTS PASSED SUCCESSFULLY!     ")
    print("==================================================")

