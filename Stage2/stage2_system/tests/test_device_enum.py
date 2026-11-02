"""Unit tests for DirectShow hardware device enumeration, virtual camera filtering, and index-shift tolerance."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from laptop.device_enum import (
    VideoDevice,
    find_matching_device_index,
    get_device_by_index,
    is_integrated_camera,
    is_virtual_camera,
)
from laptop.camera_selector import (
    load_camera_config,
    save_camera_config,
)


class TestDeviceEnumAndVirtualFiltering(unittest.TestCase):
    def test_virtual_camera_detection(self) -> None:
        self.assertTrue(is_virtual_camera("OBS Virtual Camera"))
        self.assertTrue(is_virtual_camera("obs-camera"))
        self.assertTrue(is_virtual_camera("ManyCam Virtual Webcam"))
        self.assertTrue(is_virtual_camera("XSplit VCam"))
        self.assertTrue(is_virtual_camera("DroidCam Source 1"))
        self.assertTrue(is_virtual_camera("Unity Video Capture"))

        self.assertFalse(is_virtual_camera("H264 USB Camera"))
        self.assertFalse(is_virtual_camera("Integrated Camera"))
        self.assertFalse(is_virtual_camera("Logitech Brio"))

    def test_integrated_camera_detection(self) -> None:
        self.assertTrue(is_integrated_camera("Integrated Camera"))
        self.assertTrue(is_integrated_camera("Integrated IR Camera"))
        self.assertTrue(is_integrated_camera("FaceTime HD Camera (Built-in)"))
        self.assertTrue(is_integrated_camera("Internal Webcam"))

        self.assertFalse(is_integrated_camera("H264 USB Camera"))
        self.assertFalse(is_integrated_camera("OBS Virtual Camera"))
        self.assertFalse(is_integrated_camera("USB Video Device"))

    def test_video_device_labels(self) -> None:
        d_usb = VideoDevice(index=1, name="H264 USB Camera", path="usb#vid_32e4...")
        self.assertIn("USB", d_usb.label)

        d_virt = VideoDevice(index=2, name="OBS Virtual Camera", is_virtual=True)
        self.assertIn("VIRTUAL", d_virt.label)

        d_int = VideoDevice(index=0, name="Integrated Camera", is_integrated=True)
        self.assertIn("INTERNAL", d_int.label)

    def test_get_device_by_index(self) -> None:
        devs = [
            VideoDevice(index=0, name="Integrated Camera", is_integrated=True),
            VideoDevice(index=1, name="OBS Virtual Camera", is_virtual=True),
            VideoDevice(index=2, name="H264 USB Camera", path="usb#port1"),
        ]
        self.assertEqual(get_device_by_index(0, devs).name, "Integrated Camera")
        self.assertEqual(get_device_by_index(1, devs).name, "OBS Virtual Camera")
        self.assertEqual(get_device_by_index(2, devs).name, "H264 USB Camera")
        self.assertIsNone(get_device_by_index(5, devs))

    def test_matching_device_index_by_exact_path(self) -> None:
        devs = [
            VideoDevice(index=0, name="Integrated Camera", path="usb#laptop_internal"),
            VideoDevice(index=1, name="OBS Virtual Camera", path=""),
            VideoDevice(index=2, name="H264 USB Camera", path="usb#port_left_abc"),
            VideoDevice(index=3, name="H264 USB Camera", path="usb#port_right_xyz"),
        ]
        # Match by path finds correct index
        self.assertEqual(
            find_matching_device_index(target_path="usb#port_left_abc", devices=devs),
            2,
        )
        self.assertEqual(
            find_matching_device_index(target_path="usb#port_right_xyz", devices=devs),
            3,
        )

    def test_matching_ignores_virtual_cameras_even_if_index_matches(self) -> None:
        devs = [
            VideoDevice(index=0, name="Integrated Camera", is_integrated=True),
            VideoDevice(index=1, name="OBS Virtual Camera", is_virtual=True),
            VideoDevice(index=2, name="H264 USB Camera"),
        ]
        # Even if fallback_index=1 was requested, virtual camera is NEVER accepted
        self.assertIsNone(
            find_matching_device_index(fallback_index=1, devices=devs)
        )

    def test_index_shift_survival_when_obs_virtual_camera_is_inserted(self) -> None:
        """Simulates opening OBS: indices shift by +1 when OBS Virtual Camera appears."""
        # Initial state before OBS is opened:
        devices_before = [
            VideoDevice(index=0, name="Integrated Camera", path="path_int", is_integrated=True),
            VideoDevice(index=1, name="H264 USB Camera", path="path_usb_left"),
            VideoDevice(index=2, name="H264 USB Camera", path="path_usb_right"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / ".camera_config.json"
            # Save configuration when Left=1, Right=2
            save_camera_config(cfg, left_cam=1, right_cam=2, devices=devices_before)

            # State after OBS is opened and inserts OBS Virtual Camera at Index 1:
            devices_after_obs = [
                VideoDevice(index=0, name="Integrated Camera", path="path_int", is_integrated=True),
                VideoDevice(index=1, name="OBS Virtual Camera", path="", is_virtual=True),
                VideoDevice(index=2, name="H264 USB Camera", path="path_usb_left"),
                VideoDevice(index=3, name="H264 USB Camera", path="path_usb_right"),
            ]

            # load_camera_config must automatically remap Left: 1 -> 2, Right: 2 -> 3
            remapped_left, remapped_right = load_camera_config(
                cfg, auto_remap=True, devices=devices_after_obs
            )
            self.assertEqual(remapped_left, 2)
            self.assertEqual(remapped_right, 3)

    def test_config_backward_compatibility_with_legacy_json(self) -> None:
        """Legacy configs without path/name should load cleanly without breaking."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / ".camera_config.json"
            cfg.write_text(json.dumps({"left_cam": 2, "right_cam": 1}), encoding="utf-8")

            # With no matching devices or legacy format, preserves indices
            left, right = load_camera_config(cfg, auto_remap=True, devices=[])
            self.assertEqual(left, 2)
            self.assertEqual(right, 1)


if __name__ == "__main__":
    unittest.main()

