"""End-to-end integration test of autonomous mission workflows (Levels 1, 2, and 3)."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from laptop.camera import CameraManager
from laptop.llm import llm_api, tasks as llm_tasks
from laptop.mission import MissionManager
from laptop.robot_link import RobotLink
from laptop.size_link import SizeLink
from shared.keypoints import KeypointStore
from shared.protocol import HOME_GOAL_NAME
from vm.robot_controller import RobotController


class TestMissionWorkflows(unittest.TestCase):
    START_POSE = (-0.8, -0.6, 35.0)

    @classmethod
    def setUpClass(cls):
        llm_api.set_mock_mode(True)

        cls.temp_dir = Path(tempfile.mkdtemp(prefix="test_mission_"))
        cls.points_file = cls.temp_dir / "keypoints.json"
        # All object keypoints intentionally have visibility 0.50 to match real stage2_keypoints.json
        sample_kps = {
            "keypoints": [
                {"id": 0, "name": "0", "x": 0.0, "y": 0.0, "visibility": 1.0},
                {"id": 1, "name": "B", "x": 1.0, "y": 1.0, "visibility": 0.50},
                {"id": 2, "name": "C", "x": 2.0, "y": 1.5, "visibility": 0.50},
                {"id": 3, "name": "D", "x": 3.0, "y": 2.0, "visibility": 0.50},
                {"id": 4, "name": "E", "x": 4.0, "y": 2.5, "visibility": 0.50},
            ]
        }
        cls.points_file.write_text(json.dumps(sample_kps), encoding="utf-8")

        # Start VM Robot Controller in simulated kinematic mode
        cls.vm_port = 8970
        cls.vm_controller = RobotController(
            points_path=cls.points_file,
            cmd_port=cls.vm_port,
            beacon_port=8979,
            sim_mode=True,
        )
        cls.vm_controller.sim_speed = 8.0
        cls.vm_controller.start()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.vm_controller.stop()
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        # Reset simulated robot to distinct RViz Start pose (different from Point 0!)
        sx, sy, syaw = self.START_POSE
        self.vm_controller.sim_target_xy = None
        self.vm_controller.nav_state = "idle"
        self.vm_controller.sim_x = sx
        self.vm_controller.sim_y = sy
        self.vm_controller.sim_yaw = syaw
        self.vm_controller.initial_pose = (sx, sy, syaw)

        self.camera = CameraManager(is_mock=True, fps=25.0)
        self.camera.start()

        self.robot = RobotLink(vm_ip="127.0.0.1", vm_port=self.vm_port, ping_port=8971)
        self.robot.start()
        time.sleep(0.2)

        self.size_link = SizeLink(is_mock=True)
        self.kps = KeypointStore(self.points_file)

        # Verify default settle delay is ~1.0s, then use 0.08s for fast end-to-end test execution
        default_m = MissionManager(self.camera, self.robot, self.size_link, self.kps)
        self.assertAlmostEqual(default_m.camera_settle_s, 1.0)

        self.mission = MissionManager(
            camera=self.camera,
            robot=self.robot,
            size_link=self.size_link,
            keypoints=self.kps,
            runs_dir=self.temp_dir / "runs",
            camera_settle_s=0.08,
        )

    def tearDown(self):
        self.mission.stop()
        self.camera.stop()
        self.robot.stop()

    def _assert_returned_to_start(self) -> None:
        sx, sy, syaw = self.START_POSE
        self.assertEqual(self.vm_controller.active_goal_name, HOME_GOAL_NAME)
        self.assertAlmostEqual(self.vm_controller.sim_x, sx, places=2)
        self.assertAlmostEqual(self.vm_controller.sim_y, sy, places=2)
        self.assertAlmostEqual(self.vm_controller.sim_yaw, syaw, places=1)

    def test_level_1_workflow_parallel_return_and_overlay(self):
        """Test Level 1 overlays classification on preview and returns to RViz Start in parallel."""
        orig_classify = llm_tasks.classify_object
        goal_during_last_classify: list[str] = []

        def wrapped_classify(photo_bytes: bytes, point_name: str):
            if point_name == "E":
                goal_during_last_classify.append(self.vm_controller.active_goal_name or "")
            return orig_classify(photo_bytes, point_name)

        with patch.object(llm_tasks, "classify_object", side_effect=wrapped_classify):
            ok = self.mission.start_level(1)
            self.assertTrue(ok)

            t_end = time.time() + 20.0
            while self.mission.is_running() and time.time() < t_end:
                time.sleep(0.05)

        self.assertFalse(self.mission.is_running(), "Level 1 timed out")
        result = self.mission.level_results.get(1, "")
        self.assertTrue(len(result) > 0, "Level 1 result should not be empty")
        self.assertIn("Point", result)
        self.assertEqual(goal_during_last_classify, [HOME_GOAL_NAME])
        self.assertTrue(self.mission.last_processed_name.startswith("classified_point_"))
        self._assert_returned_to_start()
        print(f"[TEST PASS] Level 1 Verified: {result}")

    def test_level_2_workflow_overlay_and_return_to_start(self):
        """Test Level 2 autonomous search, classification overlay, and return to Start."""
        ok = self.mission.start_level(2)
        self.assertTrue(ok)

        t_end = time.time() + 20.0
        while self.mission.is_running() and time.time() < t_end:
            time.sleep(0.08)

        self.assertFalse(self.mission.is_running(), "Level 2 timed out")
        result = self.mission.level_results.get(2, "")
        self.assertTrue(len(result) > 0, "Level 2 result should not be empty")
        self.assertTrue(self.mission.last_processed_name.startswith("classified_point_"))
        self._assert_returned_to_start()
        print(f"[TEST PASS] Level 2 Verified: {result}")

    def test_level_3_async_gpu_no_initial_llm_and_parallel_return(self):
        """Test Level 3: 0 initial LLM calls, closest-point order (vis=0.5), async GPU while driving, 1-shot LLM."""
        orig_measure = self.size_link.measure
        orig_solve = llm_tasks.solve_level3
        measured_points: list[str] = []
        robot_goal_while_gpu_running: list[str] = []
        solve_calls: list[int] = []

        def slow_async_measure(png_bytes: bytes, point_name=None, hint_class=None, timeout_s=15.0):
            # Sleep briefly so MissionManager has time to dispatch the next waypoint / START goal concurrently
            time.sleep(0.15)
            measured_points.append(str(point_name))
            robot_goal_while_gpu_running.append(str(self.vm_controller.active_goal_name))
            return orig_measure(png_bytes, point_name=point_name, hint_class=hint_class, timeout_s=timeout_s)

        def spy_solve(*args, **kwargs):
            solve_calls.append(1)
            return orig_solve(*args, **kwargs)

        with (
            patch.object(llm_tasks, "parse_level3_task", side_effect=AssertionError("Must not call parse_level3_task")) as mock_parse3,
            patch.object(self.size_link, "measure", side_effect=slow_async_measure),
            patch.object(llm_tasks, "solve_level3", side_effect=spy_solve),
        ):
            ok = self.mission.start_level(3)
            self.assertTrue(ok)

            t_end = time.time() + 30.0
            while self.mission.is_running() and time.time() < t_end:
                time.sleep(0.08)

            self.assertEqual(mock_parse3.call_count, 0)

        self.assertFalse(self.mission.is_running(), "Level 3 timed out")
        result = self.mission.level_results.get(3, "")
        self.assertIn("Answer:", result)
        # All 4 object points (all vis=0.50) visited in greedy closest Euclidean order from (0,0)
        self.assertEqual(measured_points, ["B", "C", "D", "E"])
        # While Point B's GPU call ran asynchronously, robot was already driving to Point C (not waiting at B!)
        self.assertNotEqual(robot_goal_while_gpu_running[0], "B")
        # While final Point E's GPU call ran, robot was already driving back to START!
        self.assertEqual(robot_goal_while_gpu_running[-1], HOME_GOAL_NAME)
        self.assertEqual(len(solve_calls), 1)
        self.assertTrue(self.mission.last_processed_name.startswith("measured_"))
        self._assert_returned_to_start()
        print(f"[TEST PASS] Level 3 Verified: {result}")

    def test_pose_confirmation_gate_and_interactive_flow(self):
        """Verify /api/goto is gated until initial pose is confirmed and interactive [y/N] latches initial_pose."""
        self.vm_controller.pose_confirmed = False
        self.vm_controller.sim_x = 1.25
        self.vm_controller.sim_y = -2.50
        self.vm_controller.sim_yaw = 90.0
        self.vm_controller.initial_pose = None
        try:
            ok, _seq, msg = self.vm_controller.request_goto(1)
            self.assertFalse(ok)
            self.assertIn("not confirmed", msg)

            # Simulate user pressing Enter -> 'n' (retry) -> Enter -> 'y' (confirm)
            answers = iter(["", "n", "", "y"])
            confirmed = self.vm_controller.confirm_initial_pose_interactive(
                input_fn=lambda _prompt: next(answers),
                settle_delay_s=0.0,
            )
            self.assertTrue(confirmed)
            self.assertTrue(self.vm_controller.pose_confirmed)
            self.assertEqual(self.vm_controller.initial_pose, (1.25, -2.50, 90.0))

            ok2, _seq2, _msg2 = self.vm_controller.request_goto(0)
            self.assertTrue(ok2)
            self.vm_controller.request_cancel()
        finally:
            self.vm_controller.pose_confirmed = True

    def test_python310_fstring_compatibility_across_stage2_system(self):
        """Ensure no Python 3.12-only backslashes inside f-string expressions exist in .py or .sh files."""
        import io
        import tokenize

        def check_code(src: str, origin: str) -> None:
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type != tokenize.STRING:
                    continue
                s = tok.string
                quote_positions = [idx for idx in (s.find('"'), s.find("'")) if idx != -1]
                if not quote_positions:
                    continue
                prefix = s[: min(quote_positions)].lower()
                if "f" not in prefix:
                    continue
                depth = 0
                i = min(quote_positions)
                while i < len(s):
                    if s[i] == "{":
                        if i + 1 < len(s) and s[i + 1] == "{":
                            i += 2
                            continue
                        depth += 1
                    elif s[i] == "}" and depth > 0:
                        depth -= 1
                    elif depth > 0 and s[i] == "\\":
                        self.fail(
                            f"Python 3.10 incompatible backslash inside f-string in {origin}:{tok.start[0]}: {s}"
                        )
                    i += 1

        for py_file in sys_root.rglob("*.py"):
            check_code(py_file.read_text(encoding="utf-8"), str(py_file))

        bad_sh_pattern = "f" + '"{p[' + "\\" + '"'
        for sh_file in sys_root.rglob("*.sh"):
            text = sh_file.read_text(encoding="utf-8")
            self.assertNotIn(bad_sh_pattern, text, f"Escaped f-string found in {sh_file}")


if __name__ == "__main__":
    unittest.main()
