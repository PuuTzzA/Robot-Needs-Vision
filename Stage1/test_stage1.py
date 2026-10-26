"""Automated unit and integration test suite for Stage 1 Stereo Vision."""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

    # Test dynamic resolution map generation (e.g. 960x540 for bandwidth-saving streams)
    map_lx, map_ly, map_rx, map_ry, fx_half = calib.get_maps_for_size(960, 540)
    assert map_lx.shape == (540, 960)
    assert np.isclose(fx_half, calib.focal_length_px * 0.5, atol=1.0)
    print("✓ Dynamic 960x540 rectification maps verified without full-frame upscaling.")

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

    # Test swapped camera mode
    estimator.swap_cameras = True
    disp_swapped = estimator.estimate_disparity(left, right)
    assert disp_swapped.shape == left.shape[:2]
    print("✓ SGBM camera swapping verified.")


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
    try:
        estimator.initialize()
    except ImportError as e:
        print(f"⚠️ Skipping CREStereo inference test ({e}). Will run on GPU server.")
        return

    print(f"✓ Initialized estimator: {estimator.name}")

    disparity = estimator.estimate_disparity(left, right)
    assert disparity.shape == left.shape[:2]
    assert disparity.dtype == np.float32

    # Verify compactness inside the active evaluated region equals 100.0%
    crop_bounds = estimator.active_crop_bounds
    assert crop_bounds is not None
    active_comp = calculate_compactness(disparity, crop_x_offset=crop_bounds[0], crop_w=crop_bounds[1])
    print(f"✓ CREStereo Compactness in active evaluated FOV: {active_comp:.1f}%")
    assert np.isclose(active_comp, 100.0, atol=0.1), f"Expected ~100% compactness, got {active_comp}%"

    print(f"✓ CREStereo disparity computed successfully.")
    print(f"✓ Disparity range: min={np.nanmin(disparity):.1f}px, max={np.nanmax(disparity):.1f}px")


def test_hitnet() -> None:
    print("\n--- [Test 5] Testing HITNet (CUDA/ONNX) ---")
    import cv2
    from estimators import HitnetEstimator
    model_path = stage1_dir / "models" / "hitnet.onnx"
    assert model_path.exists(), f"Model file missing at {model_path}"

    l_path = stage1_dir / "test_data" / "cones" / "im2.png"
    r_path = stage1_dir / "test_data" / "cones" / "im6.png"
    left = cv2.imread(str(l_path))
    right = cv2.imread(str(r_path))

    estimator = HitnetEstimator(model_path=model_path)
    try:
        estimator.initialize()
    except (ImportError, FileNotFoundError) as e:
        print(f"⚠️ Skipping HITNet inference test ({e}). Will run on GPU server.")
        return
    print(f"✓ Initialized estimator: {estimator.name}")

    disparity = estimator.estimate_disparity(left, right)
    assert disparity.shape == left.shape[:2]
    assert disparity.dtype == np.float32

    compactness = calculate_compactness(disparity)
    print(f"✓ HITNet Compactness: {compactness:.1f}%")
    print(f"✓ HITNet disparity computed successfully.")


def test_far_field_void_resilience() -> None:
    print("\n--- [Test 6] Testing Far-Field Void Resilience ---")
    h, w = 100, 100
    fx = 3800.0
    baseline_m = 0.0598
    max_depth = 15.0
    min_depth = 0.3

    # Create disparity map with far objects and residual noise
    # Some pixels at 20m, 50m, 100m, and residual slightly negative disparity (-0.2px)
    disp = np.full((h, w), 25.0, dtype=np.float32)  # ~9.0m depth
    disp[:30, :] = 10.0   # ~22.7m (beyond 15m)
    disp[30:50, :] = 2.0  # ~113m (far background)
    disp[50:60, :] = -0.2 # residual epipolar noise at optical infinity

    # 1. With resilient_voids=True
    depth_resilient = disparity_to_depth(
        disp, fx, baseline_m, min_depth=min_depth, max_depth=max_depth, resilient_voids=True
    )
    assert not np.isnan(depth_resilient).any(), "Resilient mode should have zero NaN voids for far field!"
    assert np.all(depth_resilient <= max_depth), f"Depth should be capped at max_depth ({max_depth}m)"
    comp_resilient = calculate_compactness(depth_resilient)
    assert np.isclose(comp_resilient, 100.0), f"Expected 100% compactness, got {comp_resilient}%"

    # Colorization check: Far regions should NOT be black (0,0,0)
    depth_vis = colorize_depth(depth_resilient, min_depth=min_depth, max_depth=max_depth)
    assert not np.all(depth_vis[:30, :] == 0), "Far regions should be colorized, not black voids!"
    print(f"✓ Resilient far-field depth: 100% solid, {comp_resilient:.1f}% compactness, non-black visualization.")

    # 2. Without resilient_voids (strict thresholding)
    depth_strict = disparity_to_depth(
        disp, fx, baseline_m, min_depth=min_depth, max_depth=max_depth, resilient_voids=False
    )
    assert np.isnan(depth_strict[:60, :]).any(), "Strict mode should mask out depths beyond max_depth as NaN."
    comp_strict = calculate_compactness(depth_strict)
    assert comp_strict < 60.0, f"Strict mode should have reduced compactness, got {comp_strict}%"
    print(f"✓ Strict mode verified: drops to {comp_strict:.1f}% compactness due to artificial cutoffs.")




def test_compression_disparity_fidelity() -> None:
    print("\n--- [Test 7] Testing JPEG Compression & Disparity Fidelity ---")
    import cv2
    l_path = stage1_dir / "test_data" / "cones" / "im2.png"
    r_path = stage1_dir / "test_data" / "cones" / "im6.png"
    assert l_path.exists() and r_path.exists()

    left = cv2.imread(str(l_path))
    right = cv2.imread(str(r_path))

    estimator = TraditionalSGBMEstimator()
    estimator.initialize()

    # Baseline uncompressed disparity
    disp_raw = estimator.estimate_disparity(left, right)

    # Test JPEG Quality 60 (our chosen sweet-spot)
    composite = np.hstack((left, right))
    _, buf_q80 = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_q60 = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 60])

    size_raw = composite.nbytes
    size_q80 = len(buf_q80)
    size_q60 = len(buf_q60)

    # Decode Q60
    dec_q60 = cv2.imdecode(buf_q60, cv2.IMREAD_COLOR)
    mid_x = dec_q60.shape[1] // 2
    left_q60 = dec_q60[:, :mid_x]
    right_q60 = dec_q60[:, mid_x:]

    disp_q60 = estimator.estimate_disparity(left_q60, right_q60)

    valid_mask = np.isfinite(disp_raw) & np.isfinite(disp_q60) & (disp_raw > 0)
    disp_diff = np.abs(disp_raw[valid_mask] - disp_q60[valid_mask])
    mae = float(np.mean(disp_diff))
    bad_1px = float(np.mean(disp_diff > 1.0) * 100.0)

    print(f"✓ Raw uncompressed: {size_raw / 1024:.1f} KB")
    print(f"✓ JPEG Q80 size:    {size_q80 / 1024:.1f} KB (ratio: {size_raw / size_q80:.1f}x)")
    print(f"✓ JPEG Q60 size:    {size_q60 / 1024:.1f} KB (ratio: {size_raw / size_q60:.1f}x)")
    print(f"✓ Disparity MAE (Raw vs Q60): {mae:.3f} px (Threshold < 0.45 px)")
    print(f"✓ Percentage of pixels > 1.0 px error: {bad_1px:.2f}%")

    assert mae < 0.45, f"Disparity error too high: {mae:.3f} px"
    assert size_q60 < size_q80, "Q60 should be smaller than Q80"


if __name__ == "__main__":
    print("==================================================")
    print("       STAGE 1 STEREO VISION TEST SUITE           ")
    print("==================================================")
    test_calibration()
    test_traditional_sgbm()
    test_depth_conversion_and_legend()
    test_deep_crestereo()
    test_hitnet()
    test_far_field_void_resilience()
    test_compression_disparity_fidelity()
    print("\n==================================================")
    print("       ALL STAGE 1 TESTS PASSED SUCCESSFULLY!     ")
    print("==================================================")

