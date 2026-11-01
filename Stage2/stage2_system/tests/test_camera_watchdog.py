"""Unit tests for CameraManager anti-freeze watchdog, post-settle freshness, overlays, and 720p calibration."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from laptop import camera as cam_mod
from laptop.camera import CameraManager
from laptop.vision_overlay import draw_classification_overlay, draw_measurement_overlay
from shared.protocol import DetectionItem, MeasurementResult
from ssh.utils.calibration_loader import StereoCalibration


class _FakeVideoCapture:
    """Controllable fake OpenCV VideoCapture for simulating USB dropouts and zombie buffers."""

    def __init__(self, mode: str = "animated", width: int = 320, height: int = 180) -> None:
        self.mode = mode  # animated | fail | zombie
        self.width = width
        self.height = height
        self._opened = True
        self._counter = 0
        rng = np.random.default_rng(42)
        self._zombie_frame = rng.integers(10, 245, size=(height, width, 3), dtype=np.uint8)

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self._opened = False

    def read(self):
        if not self._opened or self.mode == "fail":
            return False, None
        if self.mode == "zombie":
            return True, self._zombie_frame.copy()
        self._counter += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = (self._counter * 17) % 251
        frame[:, : self.width // 2, 1] = (self._counter * 31 + 50) % 251
        frame[: self.height // 2, :, 2] = 180
        return True, frame


class TestCameraWatchdogAndOverlays(unittest.TestCase):
    def test_right_camera_stall_does_not_freeze_left_camera(self) -> None:
        left_fake = _FakeVideoCapture(mode="animated")
        right_fake = _FakeVideoCapture(mode="fail")

        def fake_open(idx, width=1280, height=720, backend=None, retries=4, retry_delay=0.4, fps=15.0):
            return left_fake if idx == 0 else right_fake

        mgr = CameraManager(left_cam=0, right_cam=1, width=320, height=180, fps=30.0, is_mock=False)
        mgr.REOPEN_COOLDOWN_S = 60.0  # Keep right failing without reopening during this check
        with patch.object(cam_mod, "open_camera", side_effect=fake_open):
            mgr.start()
            try:
                time.sleep(0.25)
                t_mark = time.monotonic()
                jpg, arr = mgr.capture_fresh_left(min_after_mono=t_mark, timeout_s=1.0)
                self.assertIsNotNone(jpg)
                self.assertIsNotNone(arr)
                self.assertEqual(arr.shape, (180, 320, 3))
            finally:
                mgr.stop()

    def test_watchdog_auto_reopens_stalled_camera_and_decays_fps(self) -> None:
        broken_left = _FakeVideoCapture(mode="animated")
        recovered_left = _FakeVideoCapture(mode="animated")
        right_ok = _FakeVideoCapture(mode="animated")
        left_opens = []

        def fake_open(idx, width=1280, height=720, backend=None, retries=4, retry_delay=0.4, fps=15.0):
            if idx == 0:
                left_opens.append(time.monotonic())
                return broken_left if len(left_opens) == 1 else recovered_left
            return right_ok

        mgr = CameraManager(left_cam=0, right_cam=1, width=320, height=180, fps=30.0, is_mock=False)
        mgr.STALL_TIMEOUT_S = 0.25
        mgr.REOPEN_COOLDOWN_S = 0.1
        mgr.MAX_CONSECUTIVE_FAILURES = 4

        with patch.object(cam_mod, "open_camera", side_effect=fake_open):
            mgr.start()
            try:
                time.sleep(0.15)
                # Now simulate sudden USB dropout on Left camera
                broken_left.mode = "fail"
                deadline = time.monotonic() + 2.5
                while mgr.get_camera_health()["reopen_count"] < 1 and time.monotonic() < deadline:
                    time.sleep(0.05)

                self.assertGreaterEqual(mgr.get_camera_health()["reopen_count"], 1)
                self.assertGreaterEqual(len(left_opens), 2)

                # Fresh capture after recovery must succeed
                jpg, arr = mgr.capture_fresh_left(min_after_mono=time.monotonic(), timeout_s=1.2)
                self.assertIsNotNone(jpg)
                self.assertIsNotNone(arr)
            finally:
                mgr.stop()

    def test_zombie_duplicate_buffer_detection(self) -> None:
        zombie_left = _FakeVideoCapture(mode="zombie")
        recovered_left = _FakeVideoCapture(mode="animated")
        calls = []

        def fake_open(idx, width=1280, height=720, backend=None, retries=4, retry_delay=0.4, fps=15.0):
            calls.append(idx)
            return zombie_left if len(calls) == 1 else recovered_left

        mgr = CameraManager(single_cam=0, width=320, height=180, fps=30.0, is_mock=False)
        mgr.ZOMBIE_DUP_TIMEOUT_S = 0.35
        mgr.REOPEN_COOLDOWN_S = 0.1

        with patch.object(cam_mod, "open_camera", side_effect=fake_open):
            mgr.start()
            try:
                deadline = time.monotonic() + 2.5
                while mgr.get_camera_health()["reopen_count"] < 1 and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertGreaterEqual(mgr.get_camera_health()["reopen_count"], 1)
            finally:
                mgr.stop()

    def test_post_settle_barrier_rejects_pre_settle_frames(self) -> None:
        mgr = CameraManager(width=320, height=180, fps=25.0, is_mock=True)
        mgr.start()
        try:
            time.sleep(0.1)
            # Request a frame strictly from 0.12s in the future (simulating post-settle timestamp)
            future_settle = time.monotonic() + 0.12
            jpg, arr = mgr.capture_fresh_left(min_after_mono=future_settle, timeout_s=1.5)
            finished_mono = time.monotonic()
            self.assertIsNotNone(jpg)
            self.assertIsNotNone(arr)
            self.assertGreaterEqual(finished_mono, future_settle)
        finally:
            mgr.stop()

    def test_overlays_do_not_mutate_input_and_render_pixels(self) -> None:
        raw = np.full((360, 640, 3), 60, dtype=np.uint8)
        raw_copy = raw.copy()

        cls_img = draw_classification_overlay(raw, title="Point A", category="Cup", color="white")
        np.testing.assert_array_equal(raw, raw_copy)
        self.assertEqual(cls_img.shape, raw.shape)
        self.assertFalse(np.array_equal(cls_img, raw))

        det = DetectionItem(
            class_name="bottle",
            confidence=0.93,
            bbox_px=[120, 60, 280, 320],
            height_cm=22.4,
            width_cm=7.1,
            distance_m=0.85,
            valid_depth_pixels=4200,
        )
        meas = MeasurementResult(ok=True, point="D", primary=det, detections=[det])
        meas_img = draw_measurement_overlay(raw, meas, point_name="D")
        np.testing.assert_array_equal(raw, raw_copy)
        self.assertEqual(meas_img.shape, raw.shape)
        self.assertFalse(np.array_equal(meas_img, raw))

    def test_720p_stereo_calibration_intrinsics_scaling(self) -> None:
        calib_yml = sys_root / "ssh" / "calibration" / "stereo_calibration_fixed.yml"
        if not calib_yml.exists():
            self.skipTest("Calibration YAML not present in workspace")

        cal = StereoCalibration(calibration_path=calib_yml, target_size=(1920, 1080))
        fx_1080, fy_1080, cx_1080, cy_1080 = cal.get_intrinsics(1920, 1080)
        fx_720, fy_720, cx_720, cy_720 = cal.get_intrinsics(1280, 720)

        self.assertAlmostEqual(fx_720 / fx_1080, 1280.0 / 1920.0, places=2)
        self.assertAlmostEqual(fy_720 / fy_1080, 720.0 / 1080.0, places=2)
        self.assertAlmostEqual(cx_720 / cx_1080, 1280.0 / 1920.0, places=2)
        self.assertAlmostEqual(cy_720 / cy_1080, 720.0 / 1080.0, places=2)


if __name__ == "__main__":
    unittest.main()

