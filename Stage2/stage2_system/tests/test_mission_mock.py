"""End-to-end integration test of autonomous mission workflows (Levels 1, 2, and 3)."""

from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from laptop.camera import CameraManager
from laptop.llm import llm_api
from laptop.mission import MissionManager
from laptop.robot_link import RobotLink
from laptop.size_link import SizeLink
from shared.keypoints import KeypointStore
from vm.robot_controller import RobotController


class TestMissionWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        llm_api.set_mock_mode(True)

        cls.temp_dir = Path(tempfile.mkdtemp(prefix="test_mission_"))
        cls.points_file = cls.temp_dir / "keypoints.json"
        sample_kps = {
            "keypoints": [
                {"id": 0, "name": "0", "x": 0.0, "y": 0.0, "visibility": 1.0},
                {"id": 1, "name": "B", "x": 1.0, "y": 1.0, "visibility": 0.85},
                {"id": 2, "name": "C", "x": 2.0, "y": 1.5, "visibility": 0.50},
                {"id": 3, "name": "D", "x": 3.0, "y": 2.0, "visibility": 0.90},
                {"id": 4, "name": "E", "x": 4.0, "y": 2.5, "visibility": 0.80},
            ]
        }
        import json
        cls.points_file.write_text(json.dumps(sample_kps), encoding="utf-8")

        # Start VM Robot Controller in simulated kinematic mode
        cls.vm_port = 8970
        cls.vm_controller = RobotController(
            points_path=cls.points_file,
            cmd_port=cls.vm_port,
            beacon_port=8979,
            sim_mode=True,
        )
        cls.vm_controller.start()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.vm_controller.stop()
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        self.camera = CameraManager(is_mock=True)
        self.camera.start()

        self.robot = RobotLink(vm_ip="127.0.0.1", vm_port=self.vm_port, ping_port=8971)
        self.robot.start()
        time.sleep(0.2)

        self.size_link = SizeLink(is_mock=True)
        self.kps = KeypointStore(self.points_file)

        self.mission = MissionManager(
            camera=self.camera,
            robot=self.robot,
            size_link=self.size_link,
            keypoints=self.kps,
            runs_dir=self.temp_dir / "runs",
        )

    def tearDown(self):
        self.mission.stop()
        self.camera.stop()
        self.robot.stop()

    def test_level_1_workflow(self):
        """Test Level 1 autonomous execution from start to finish."""
        ok = self.mission.start_level(1)
        self.assertTrue(ok)

        # Wait for level 1 to finish
        t_end = time.time() + 20.0
        while self.mission.is_running() and time.time() < t_end:
            time.sleep(0.05)

        self.assertFalse(self.mission.is_running(), "Level 1 timed out")
        result = self.mission.level_results.get(1, "")
        self.assertTrue(len(result) > 0, "Level 1 result should not be empty")
        self.assertIn("Point", result)
        print(f"[TEST PASS] Level 1 Verified: {result}")

    def test_level_2_workflow(self):
        """Test Level 2 autonomous search and early exit when targets found."""
        ok = self.mission.start_level(2)
        self.assertTrue(ok)

        t_end = time.time() + 20.0
        while self.mission.is_running() and time.time() < t_end:
            time.sleep(0.1)

        self.assertFalse(self.mission.is_running(), "Level 2 timed out")
        result = self.mission.level_results.get(2, "")
        self.assertTrue(len(result) > 0, "Level 2 result should not be empty")
        print(f"[TEST PASS] Level 2 Verified: {result}")

    def test_level_3_workflow(self):
        """Test Level 3 autonomous inspection, size estimation, and multimodal solution."""
        ok = self.mission.start_level(3)
        self.assertTrue(ok)

        t_end = time.time() + 30.0
        while self.mission.is_running() and time.time() < t_end:
            time.sleep(0.1)

        self.assertFalse(self.mission.is_running(), "Level 3 timed out")
        result = self.mission.level_results.get(3, "")
        self.assertTrue(len(result) > 0, "Level 3 result should not be empty")
        self.assertIn("Answer:", result)
        print(f"[TEST PASS] Level 3 Verified: {result}")


if __name__ == "__main__":
    unittest.main()
