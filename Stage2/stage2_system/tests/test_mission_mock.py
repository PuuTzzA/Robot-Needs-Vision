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

from laptop.audio_announcer import AudioAnnouncer
from laptop.camera import CameraManager
from laptop.llm import llm_api, tasks as llm_tasks
from laptop.llm.schemas import TaskLegibility
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
        self.vm_controller.sim_speed = 8.0
        self.vm_controller.goal_timeout_s = 60.0
        self.vm_controller.stuck_timeout_s = 10.0
        self.vm_controller.sim_target_xy = None
        self.vm_controller.nav_state = "idle"
        self.vm_controller.nav_message = ""
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

        # Verify default settle delay and navigation timeouts, then use fast settle for tests
        default_m = MissionManager(self.camera, self.robot, self.size_link, self.kps)
        self.assertAlmostEqual(default_m.camera_settle_s, 1.0)
        self.assertAlmostEqual(default_m.goal_timeout_s, 60.0)
        self.assertAlmostEqual(default_m.stuck_timeout_s, 10.0)

        self.mission = MissionManager(
            camera=self.camera,
            robot=self.robot,
            size_link=self.size_link,
            keypoints=self.kps,
            runs_dir=self.temp_dir / "runs",
            camera_settle_s=0.08,
            announcer=AudioAnnouncer(enabled=False),
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
        self.assertIn("Point B: Cup (11 cm)", result)
        self.assertIn("Point E: Banana (15 cm)", result)
        self.assertEqual(goal_during_last_classify, [HOME_GOAL_NAME])
        self.assertTrue(self.mission.last_processed_name.startswith("classified_point_"))
        self._assert_returned_to_start()
        self.assertIn("Starting Level 1", self.mission.announcer.history)
        self.assertIn("Going to Point 0", self.mission.announcer.history)
        self.assertIn("Taking photo of tasks", self.mission.announcer.history)
        self.assertIn("Taking photo of Point B", self.mission.announcer.history)
        self.assertIn("Finished Level 1", self.mission.announcer.history)
        self.assertIn(result, self.mission.announcer.history)
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
        self.assertIn("Starting Level 2", self.mission.announcer.history)
        self.assertIn("Going to Point 0", self.mission.announcer.history)
        self.assertIn("Taking photo of tasks", self.mission.announcer.history)
        self.assertIn("Taking photo of Point B", self.mission.announcer.history)
        self.assertIn("Finished Level 2", self.mission.announcer.history)
        print(f"[TEST PASS] Level 2 Verified: {result}")

    def test_empty_box_none_option_in_level1_and_level2(self):
        """Test empty boxes ('None') are reported cleanly in Level 1 and skipped without false match in Level 2."""
        from laptop.llm.schemas import Category

        # 1. Level 1: Point B is empty, Point E has a Banana (15 cm)
        def mock_lvl1_classify(_photo_bytes: bytes, point_name: str) -> Category:
            if point_name == "B":
                return Category(category="None", color=None, height_cm=None)
            return Category(category="Banana", color="yellow", height_cm=15.0)

        with patch.object(llm_tasks, "classify_object", side_effect=mock_lvl1_classify):
            self.assertTrue(self.mission.start_level(1))
            t_end = time.time() + 20.0
            while self.mission.is_running() and time.time() < t_end:
                time.sleep(0.05)

        self.assertFalse(self.mission.is_running(), "Level 1 empty-box test timed out")
        lvl1_result = self.mission.level_results.get(1, "")
        self.assertEqual(lvl1_result, "Point B: None, Point E: Banana (15 cm)")

        # 2. Level 2: Points B and C are empty ('None'), Point D has Cup and Point E has Book
        visited_lvl2: list[str] = []

        def mock_lvl2_classify(_photo_bytes: bytes, point_name: str) -> Category:
            visited_lvl2.append(point_name)
            if point_name in ("B", "C"):
                return Category(category="None", color=None, height_cm=None)
            if point_name == "D":
                return Category(category="Mug/Cup", color="white", height_cm=11.0)
            return Category(category="Book", color="red", height_cm=22.0)

        with patch.object(llm_tasks, "classify_object", side_effect=mock_lvl2_classify):
            self.assertTrue(self.mission.start_level(2))
            t_end = time.time() + 20.0
            while self.mission.is_running() and time.time() < t_end:
                time.sleep(0.05)

        self.assertFalse(self.mission.is_running(), "Level 2 empty-box test timed out")
        self.assertEqual(visited_lvl2, ["B", "C", "D", "E"])
        lvl2_result = self.mission.level_results.get(2, "")
        self.assertIn("Cup at Point D", lvl2_result)
        self.assertIn("Book at Point E", lvl2_result)

    def test_level_3_async_legibility_immediate_preview_and_parallel_return(self):
        """Test Level 3: async legibility while driving, immediate raw preview -> depth bbox replacement, 1-shot sentence solve."""
        orig_measure = self.size_link.measure
        orig_solve = llm_tasks.solve_level3
        orig_legib = llm_tasks.check_level3_legibility
        measured_points: list[str] = []
        robot_goal_while_gpu_running: list[str] = []
        robot_goal_during_legibility: list[str] = []
        processed_frame_names: list[str] = []
        frame_name_when_measure_started: dict[str, str] = {}
        solve_calls: list[int] = []

        self.mission.add_listener(
            lambda: (
                processed_frame_names.append(self.mission.last_processed_name)
                if (
                    self.mission.last_processed_name
                    and (
                        not processed_frame_names
                        or processed_frame_names[-1] != self.mission.last_processed_name
                    )
                )
                else None
            )
        )

        def async_legibility_spy(photo_bytes: bytes):
            time.sleep(0.08)
            robot_goal_during_legibility.append(str(self.vm_controller.active_goal_name))
            return orig_legib(photo_bytes)

        def slow_async_measure(png_bytes: bytes, point_name=None, hint_class=None, timeout_s=15.0):
            pt_str = str(point_name)
            frame_name_when_measure_started[pt_str] = self.mission.last_processed_name
            # Sleep briefly so MissionManager has time to dispatch the next waypoint / START goal concurrently
            time.sleep(0.12)
            measured_points.append(pt_str)
            robot_goal_while_gpu_running.append(str(self.vm_controller.active_goal_name))
            return orig_measure(png_bytes, point_name=point_name, hint_class=hint_class, timeout_s=timeout_s)

        def spy_solve(*args, **kwargs):
            solve_calls.append(1)
            return orig_solve(*args, **kwargs)

        with (
            patch.object(llm_tasks, "check_level3_legibility", side_effect=async_legibility_spy) as mock_legib,
            patch.object(self.size_link, "measure", side_effect=slow_async_measure),
            patch.object(llm_tasks, "solve_level3", side_effect=spy_solve),
        ):
            ok = self.mission.start_level(3)
            self.assertTrue(ok)

            t_end = time.time() + 30.0
            while self.mission.is_running() and time.time() < t_end:
                time.sleep(0.05)

            self.assertEqual(mock_legib.call_count, 1)

        self.assertFalse(self.mission.is_running(), "Level 3 timed out")
        result = self.mission.level_results.get(3, "")
        self.assertIn("Answer:", result)
        # Full-sentence check: mock answer names points and objects in a sentence
        self.assertGreater(len(result.split()), 6)
        self.assertIn("point", result.lower())
        # Legibility ran asynchronously while robot was already navigating away from Point 0
        self.assertNotEqual(robot_goal_during_legibility, ["0"])
        # All 4 object points visited in greedy closest Euclidean order from (0,0)
        self.assertEqual(measured_points, ["B", "C", "D", "E"])
        # Raw photo was already displayed in UI before GPU measurement call started
        self.assertEqual(frame_name_when_measure_started["B"], "point_B.jpg")
        self.assertIn("point_E.jpg", processed_frame_names)
        self.assertIn("measured_E.jpg", processed_frame_names)
        self.assertLess(
            processed_frame_names.index("point_E.jpg"),
            processed_frame_names.index("measured_E.jpg"),
        )
        # While Point B's GPU call ran asynchronously, robot was already driving to Point C
        self.assertNotEqual(robot_goal_while_gpu_running[0], "B")
        # While final Point E's GPU call ran, robot was already driving back to START!
        self.assertEqual(robot_goal_while_gpu_running[-1], HOME_GOAL_NAME)
        self.assertEqual(len(solve_calls), 1)
        self.assertTrue(self.mission.last_processed_name.startswith("measured_"))
        self._assert_returned_to_start()
        self.assertIn("Starting Level 3", self.mission.announcer.history)
        self.assertIn("Taking photo of tasks", self.mission.announcer.history)
        self.assertIn("Taking photo of Point B", self.mission.announcer.history)
        self.assertIn("Finished Level 3", self.mission.announcer.history)
        print(f"[TEST PASS] Level 3 Verified: {result}")

    def test_level_3_illegible_task_returns_to_point_0_once_without_rechecking(self):
        """If async legibility returns False, robot returns to Point 0 once for retake and does NOT re-check legibility."""
        orig_solve = llm_tasks.solve_level3
        legib_calls: list[bytes] = []
        solve_task_photos: list[bytes] = []

        def fake_illegible_check(photo_bytes: bytes) -> TaskLegibility:
            legib_calls.append(photo_bytes)
            # Delay so robot has already driven to Point B while legibility check runs
            time.sleep(0.25)
            return TaskLegibility(legible=False, reason="screen glare")

        def spy_solve(task_photo_bytes: bytes, *args, **kwargs):
            solve_task_photos.append(task_photo_bytes)
            return orig_solve(task_photo_bytes, *args, **kwargs)

        with (
            patch.object(llm_tasks, "check_level3_legibility", side_effect=fake_illegible_check),
            patch.object(llm_tasks, "solve_level3", side_effect=spy_solve),
        ):
            ok = self.mission.start_level(3)
            self.assertTrue(ok)

            t_end = time.time() + 30.0
            while self.mission.is_running() and time.time() < t_end:
                time.sleep(0.05)

        self.assertFalse(self.mission.is_running(), "Level 3 retake workflow timed out")
        # Legibility check must be called ONCE (not repeated after retake!)
        self.assertEqual(len(legib_calls), 1)
        self.assertEqual(len(solve_task_photos), 1)

        # Verify task_level3_retake.jpg was saved and passed to solve_level3
        retake_files = list((self.temp_dir / "runs").rglob("task_level3_retake.jpg"))
        self.assertGreaterEqual(len(retake_files), 1)
        latest_retake = sorted(retake_files)[-1]
        self.assertEqual(solve_task_photos[0], latest_retake.read_bytes())
        self._assert_returned_to_start()

    def test_navigation_failure_mid_queue_and_last_keypoint_detour_retry(self):
        """Test mid-queue failure ('Failed to get to C, continuing with D' -> retry C) and last-point detour retry."""
        statuses: list[str] = []
        self.mission.add_listener(lambda: statuses.append(self.mission.status_text))

        # --- Part 1: Mid-queue failure (visit B, C, D; C fails first time -> B, D, then retry C) ---
        classified_order_1: list[str] = []
        orig_classify = llm_tasks.classify_object
        orig_goto = self.robot.goto_keypoint
        c_failed_once = {"done": False}

        def fail_c_once(keypoint_id: int):
            kp = self.kps.get_by_id(keypoint_id)
            if kp and kp.name == "C" and not c_failed_once["done"]:
                c_failed_once["done"] = True
                return False, 0, "Simulated Nav2 abort at C"
            return orig_goto(keypoint_id)

        def record_classify_1(photo_bytes: bytes, point_name: str):
            classified_order_1.append(point_name)
            return orig_classify(photo_bytes, point_name)

        with (
            patch.object(llm_tasks, "parse_level1_task", return_value=["B", "C", "D"]),
            patch.object(self.robot, "goto_keypoint", side_effect=fail_c_once),
            patch.object(llm_tasks, "classify_object", side_effect=record_classify_1),
        ):
            self.assertTrue(self.mission.start_level(1))
            t_end = time.time() + 20.0
            while self.mission.is_running() and time.time() < t_end:
                time.sleep(0.05)

        self.assertFalse(self.mission.is_running())
        self.assertEqual(classified_order_1, ["B", "D", "C"])
        self.assertIn("Failed to get to C, continuing with D", statuses)
        self.assertIn("Failed to go to Point C, proceeding to Point D", self.mission.announcer.history)
        self.assertIn("Retrying to go to Point C", self.mission.announcer.history)

        # --- Part 2: Last-keypoint failure (visit B, D; D fails first time -> closest other kp C -> return to D) ---
        statuses.clear()
        classified_order_2: list[str] = []
        goto_history_2: list[str] = []
        d_failed_once = {"done": False}

        def fail_d_once(keypoint_id: int):
            kp = self.kps.get_by_id(keypoint_id)
            if kp:
                goto_history_2.append(kp.name)
            if kp and kp.name == "D" and not d_failed_once["done"]:
                d_failed_once["done"] = True
                return False, 0, "Simulated Nav2 abort at last point D"
            return orig_goto(keypoint_id)

        def record_classify_2(photo_bytes: bytes, point_name: str):
            classified_order_2.append(point_name)
            return orig_classify(photo_bytes, point_name)

        with (
            patch.object(llm_tasks, "parse_level1_task", return_value=["B", "D"]),
            patch.object(self.robot, "goto_keypoint", side_effect=fail_d_once),
            patch.object(llm_tasks, "classify_object", side_effect=record_classify_2),
        ):
            self.assertTrue(self.mission.start_level(1))
            t_end = time.time() + 20.0
            while self.mission.is_running() and time.time() < t_end:
                time.sleep(0.05)

        self.assertFalse(self.mission.is_running())
        # Robot was at B(1.0, 1.0) when D(3.0, 2.0) failed -> closest other keypoint != D is B or C!
        # Verify it announced continuing with the closest other keypoint, drove there, and then retried & classified D
        self.assertEqual(classified_order_2, ["B", "D"])
        self.assertTrue(
            any(s.startswith("Failed to get to D, continuing with ") for s in statuses),
            f"Expected last-point detour status in {statuses}",
        )
        # Navigation sequence must end with [..., 'D (fail)', '<detour != D>', 'D (retry)']
        d_first_idx = goto_history_2.index("D")
        self.assertGreaterEqual(len(goto_history_2), d_first_idx + 3)
        self.assertNotEqual(goto_history_2[d_first_idx + 1], "D")
        self.assertEqual(goto_history_2[d_first_idx + 2], "D")

    def test_vm_stuck_and_goal_timeouts_report_failed_and_reset_on_movement(self):
        """Verify VM controller 10s stuck timeout (resetting on movement) and 60s goal timeout report failed."""
        # 1. Stationary robot triggers stuck timeout -> VM reports nav_state='failed'
        self.vm_controller.sim_speed = 0.0
        self.vm_controller.stuck_timeout_s = 0.22
        self.vm_controller.goal_timeout_s = 5.0
        ok, seq, _ = self.robot.goto_keypoint(4)  # Point E at (4.0, 2.5)
        self.assertTrue(ok)
        reached, reason = self.robot.wait_for_goal(seq, timeout=3.0, stuck_timeout=2.0)
        self.assertFalse(reached)
        self.assertIn("stuck", reason.lower())
        self.assertEqual(self.vm_controller.nav_state, "failed")
        self.assertIn("stuck", self.vm_controller.nav_message.lower())

        # 2. Moving robot continuously resets stuck timeout even when total travel time > stuck_timeout_s
        self.vm_controller.sim_x = 0.0
        self.vm_controller.sim_y = 0.0
        self.vm_controller.sim_speed = 3.5  # Distance to B(1,1) is ~1.41m -> ~0.40s (> 0.22s stuck_timeout!)
        self.vm_controller.stuck_timeout_s = 0.22
        self.vm_controller.goal_timeout_s = 5.0
        ok2, seq2, _ = self.robot.goto_keypoint(1)  # Point B at (1.0, 1.0)
        self.assertTrue(ok2)
        reached2, reason2 = self.robot.wait_for_goal(seq2, timeout=3.0, stuck_timeout=2.0)
        self.assertTrue(reached2, f"Moving robot should reset stuck timer, got: {reason2}")

        # 3. Goal timeout triggers even while moving slowly if total goal_timeout_s is exceeded
        self.vm_controller.sim_x = 0.0
        self.vm_controller.sim_y = 0.0
        self.vm_controller.sim_speed = 1.5  # Distance to E(4, 2.5) is ~4.72m -> ~3.1s
        self.vm_controller.stuck_timeout_s = 5.0
        self.vm_controller.goal_timeout_s = 0.25
        ok3, seq3, _ = self.robot.goto_keypoint(4)
        self.assertTrue(ok3)
        reached3, reason3 = self.robot.wait_for_goal(seq3, timeout=3.0, stuck_timeout=5.0)
        self.assertFalse(reached3)
        self.assertTrue("timed out" in reason3.lower() or "timeout" in reason3.lower())
        self.assertEqual(self.vm_controller.nav_state, "failed")

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
