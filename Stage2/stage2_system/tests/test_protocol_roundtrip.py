"""Tests for protocol serialization and consistency across shared/vm/ssh."""

import json
from pathlib import Path
import sys
import unittest

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from shared.protocol import (
    DEFAULT_VM_CMD_PORT,
    DetectionItem,
    GotoRequest,
    MeasurementResult,
    RegisterRequest,
    RobotStatus,
    TeleopRequest,
)


class TestProtocol(unittest.TestCase):
    def test_robot_status_roundtrip(self):
        status = RobotStatus(
            localized=True,
            x=1.23,
            y=4.56,
            yaw_deg=90.0,
            nav_state="navigating",
            goal_seq=1,
            goal_keypoint_id=2,
            goal_name="B",
            nav2_ready=True,
            teleop_active=False,
            keypoints_sha="abc123",
        )
        d = status.to_dict()
        self.assertEqual(d["x"], 1.23)
        self.assertEqual(d["nav_state"], "navigating")

        restored = RobotStatus.from_dict(d)
        self.assertEqual(restored.x, 1.23)
        self.assertEqual(restored.yaw_deg, 90.0)
        self.assertEqual(restored.goal_name, "B")

    def test_goto_and_teleop_request(self):
        goto = GotoRequest(keypoint_id=5)
        self.assertEqual(goto.to_dict()["keypoint_id"], 5)
        self.assertEqual(GotoRequest.from_dict({"keypoint_id": 5}).keypoint_id, 5)

        teleop = TeleopRequest(key="w", duration=0.6, linear_x=0.15, angular_z=0.0)
        td = teleop.to_dict()
        self.assertEqual(td["key"], "w")
        self.assertEqual(TeleopRequest.from_dict(td).linear_x, 0.15)

    def test_measurement_roundtrip(self):
        det = DetectionItem(
            class_name="cup",
            confidence=0.95,
            bbox_px=[10, 20, 100, 200],
            height_cm=12.5,
            width_cm=8.0,
            distance_m=0.75,
            valid_depth_pixels=500,
        )
        meas = MeasurementResult(
            ok=True,
            point="D",
            primary=det,
            detections=[det],
            timing_ms={"stereo": 15.0, "total": 25.0},
        )
        md = meas.to_dict()
        restored = MeasurementResult.from_dict(md)
        self.assertTrue(restored.ok)
        self.assertEqual(restored.point, "D")
        self.assertIsNotNone(restored.primary)
        self.assertEqual(restored.primary.class_name, "cup")
        self.assertAlmostEqual(restored.primary.height_cm, 12.5)

    def test_file_copies_in_sync(self):
        base_dir = Path(__file__).resolve().parent.parent
        shared_proto = (base_dir / "shared" / "protocol.py").read_text(encoding="utf-8")
        vm_proto = (base_dir / "vm" / "protocol.py").read_text(encoding="utf-8")
        ssh_proto = (base_dir / "ssh" / "protocol.py").read_text(encoding="utf-8")
        self.assertEqual(shared_proto, vm_proto, "vm/protocol.py has drifted from shared/protocol.py")
        self.assertEqual(shared_proto, ssh_proto, "ssh/protocol.py has drifted from shared/protocol.py")

        shared_kp = (base_dir / "shared" / "keypoints.py").read_text(encoding="utf-8")
        vm_kp = (base_dir / "vm" / "keypoints.py").read_text(encoding="utf-8")
        ssh_kp = (base_dir / "ssh" / "keypoints.py").read_text(encoding="utf-8")
        self.assertEqual(shared_kp, vm_kp, "vm/keypoints.py has drifted from shared/keypoints.py")
        self.assertEqual(shared_kp, ssh_kp, "ssh/keypoints.py has drifted from shared/keypoints.py")


if __name__ == "__main__":
    unittest.main()
