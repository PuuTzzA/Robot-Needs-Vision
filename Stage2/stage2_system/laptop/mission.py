"""Mission Manager and Autonomous Level State Machine.

Executes Level 1, Level 2, and Level 3 workflows per outline.txt:
- Level 1: Transcribe 2 points -> visit each -> classify object
- Level 2: Transcribe 2 categories -> search points in greedy order until both found
- Level 3: Transcribe task & high-visibility flag -> inspect all points (with stereo height if needed) -> solve in one shot
"""

from __future__ import annotations

import base64
from datetime import datetime
import io
from pathlib import Path
import threading
import time
from typing import Callable, Optional
import cv2
import numpy as np

from shared.keypoints import Keypoint, KeypointStore
from shared.protocol import (
    TASK_POINT_NAME,
    VISIBILITY_DEFAULT,
    VISIBILITY_HIGH,
    MeasurementResult,
    RobotStatus,
)
from .camera import CameraManager
from .llm import tasks as llm_tasks
from .robot_link import RobotLink
from .size_link import SizeLink


class MissionManager:
    """Orchestrates autonomous perception and navigation workflows."""

    def __init__(
        self,
        camera: CameraManager,
        robot: RobotLink,
        size_link: SizeLink,
        keypoints: KeypointStore,
        runs_dir: Optional[Path] = None,
    ) -> None:
        self.camera = camera
        self.robot = robot
        self.size_link = size_link
        self.keypoints = keypoints
        self.runs_dir = runs_dir or Path("runs")

        self.current_level: int = 0
        self.status_text: str = "Idle (Ready)"
        self.extra_info: str = ""
        self.level_results: dict[int, str] = {1: "", 2: "", 3: ""}

        self.last_processed_frame: Optional[np.ndarray] = None
        self.last_processed_name: str = ""
        self.target_keypoint_id: Optional[int] = None

        self._active_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[], None]] = []

    def add_listener(self, cb: Callable[[], None]) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in self._listeners:
            try:
                cb()
            except Exception:
                pass

    def set_status(self, text: str, extra: str = "") -> None:
        with self._lock:
            self.status_text = text
            if extra:
                self.extra_info = extra
        print(f"[Mission] {text} {f'({extra})' if extra else ''}")
        self._notify()

    def set_processed_frame(self, frame: np.ndarray, name: str) -> None:
        with self._lock:
            self.last_processed_frame = frame.copy()
            self.last_processed_name = name
        self._notify()

    def set_result(self, level: int, result_str: str) -> None:
        with self._lock:
            self.level_results[level] = result_str
        print(f"\n=======================================================")
        print(f"[MISSION COMPLETE] Level {level} Result:\n{result_str}")
        print(f"=======================================================\n")
        self._notify()

    def is_running(self) -> bool:
        return self._active_thread is not None and self._active_thread.is_alive()

    def stop(self) -> None:
        """Cancel active navigation and abort running mission."""
        self._cancel_event.set()
        self.robot.cancel_navigation()
        self.set_status("Aborted / Stopped by User")

    def start_level(self, level: int) -> bool:
        """Start execution of Level 1, 2, or 3 in a background worker thread."""
        if self.is_running():
            print(f"[Mission] [WARN] Cannot start Level {level}: mission already active.")
            return False

        if level not in (1, 2, 3):
            print(f"[Mission] [ERROR] Unknown Level {level}. Expected 1, 2, or 3.")
            return False

        self.current_level = level
        self._cancel_event.clear()

        # Create session directory
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.runs_dir / f"{stamp}_level{level}"
        session_dir.mkdir(parents=True, exist_ok=True)

        if level == 1:
            target_fn = self._run_level_1
        elif level == 2:
            target_fn = self._run_level_2
        else:
            target_fn = self._run_level_3

        self._active_thread = threading.Thread(
            target=self._worker_wrapper,
            args=(target_fn, session_dir),
            daemon=True,
        )
        self._active_thread.start()
        return True

    def restart_level(self, level: int) -> bool:
        """Cancel current execution, clear result, and restart specified level."""
        self.stop()
        if self._active_thread and self._active_thread.is_alive():
            self._active_thread.join(timeout=1.5)
        self.set_result(level, "")
        return self.start_level(level)

    def _worker_wrapper(self, target_fn: Callable[[Path], None], session_dir: Path) -> None:
        try:
            target_fn(session_dir)
        except Exception as err:
            import traceback
            traceback.print_exc()
            self.set_status(f"Error: {err}")
        finally:
            with self._lock:
                self.target_keypoint_id = None
            self._notify()

    # -------------------------------------------------------------------------
    # Helper: Navigation & Photo Capture
    # -------------------------------------------------------------------------

    def _navigate_to_entry(self, entry: Keypoint, timeout: float = 65.0) -> bool:
        with self._lock:
            self.target_keypoint_id = entry.id
        self.set_status(f"Navigating to {entry.name} (ID {entry.id}, Vis {entry.visibility:.2f})...")

        ok, seq, msg = self.robot.goto_keypoint(entry.id)
        if not ok:
            self.set_status(f"Navigation dispatch failed: {msg}")
            return False

        success, res_msg = self.robot.wait_for_goal(seq, timeout=timeout, cancel_event=self._cancel_event)
        if not success:
            self.set_status(f"Failed reaching {entry.name}: {res_msg}")
            return False

        # Brief settle time for camera stability
        time.sleep(0.5)
        return True

    def _capture_screen_or_object(self, name_tag: str, session_dir: Path) -> tuple[bytes, np.ndarray]:
        """Capture left frame JPEG bytes and numpy image, saving to session directory."""
        left_bytes = self.camera.snapshot_left_jpeg(quality=90)
        left_arr, _, _ = self.camera.get_latest_pair()

        if left_bytes is None or left_arr is None:
            # Fallback mock frame
            left_arr = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(left_arr, f"SNAPSHOT: {name_tag}", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            _, buf = cv2.imencode(".jpg", left_arr)
            left_bytes = buf.tobytes()

        # Save to disk
        out_path = session_dir / f"{name_tag}.jpg"
        out_path.write_bytes(left_bytes)

        self.set_processed_frame(left_arr, out_path.name)
        return left_bytes, left_arr

    # -------------------------------------------------------------------------
    # LEVEL 1 WORKFLOW
    # -------------------------------------------------------------------------
    def _run_level_1(self, session_dir: Path) -> None:
        self.set_status("Level 1: Moving to Task Center (Point 0)...")

        # 1. Go to Point 0
        task_kp = self.keypoints.select_entry(TASK_POINT_NAME, min_visibility=VISIBILITY_DEFAULT)
        if not task_kp:
            raise RuntimeError("Task Point '0' not found in keypoints store!")

        if not self._navigate_to_entry(task_kp):
            return

        # 2. Take picture of task instructions
        self.set_status("Level 1: Reading task instructions on laptop screen...")
        task_bytes, _ = self._capture_screen_or_object("task_level1", session_dir)

        # 3. LLM call to get points
        known_objs = self.keypoints.object_point_names()
        points_to_visit = llm_tasks.parse_level1_task(task_bytes, known_locations=known_objs)
        print(f"[Level 1] Target Points identified by LLM: {points_to_visit}")

        if len(points_to_visit) == 0:
            self.set_status("LLM returned no target points!")
            return

        # 4. Visit each point and classify
        identified_objects = {}
        for pt_name in points_to_visit:
            if self._cancel_event.is_set():
                return

            entry = self.keypoints.select_entry(pt_name, min_visibility=VISIBILITY_DEFAULT)
            if not entry:
                print(f"[Level 1] [WARN] No recorded keypoint entry for Point {pt_name}!")
                continue

            if not self._navigate_to_entry(entry):
                continue

            self.set_status(f"Level 1: Inspecting Point {pt_name}...")
            photo_bytes, _ = self._capture_screen_or_object(f"point_{pt_name}", session_dir)

            self.set_status(f"Level 1: Classifying object at Point {pt_name}...")
            category_res = llm_tasks.classify_object(photo_bytes, pt_name)
            identified_objects[pt_name] = category_res.category
            print(f"[Level 1] Point {pt_name} -> {category_res.category} ({category_res.color})")

        # 5. Build final result string
        res_parts = [f"Point {p}: {c}" for p, c in identified_objects.items()]
        final_str = ", ".join(res_parts) if res_parts else "No objects identified."
        self.set_result(1, final_str)
        self.set_status("Level 1 Complete!", final_str)

    # -------------------------------------------------------------------------
    # LEVEL 2 WORKFLOW
    # -------------------------------------------------------------------------
    def _run_level_2(self, session_dir: Path) -> None:
        self.set_status("Level 2: Moving to Task Center (Point 0)...")

        # 1. Go to Point 0
        task_kp = self.keypoints.select_entry(TASK_POINT_NAME, min_visibility=VISIBILITY_DEFAULT)
        if not task_kp:
            raise RuntimeError("Task Point '0' not found in keypoints store!")

        if not self._navigate_to_entry(task_kp):
            return

        # 2. Take picture of task instructions
        self.set_status("Level 2: Reading task instructions on laptop screen...")
        task_bytes, _ = self._capture_screen_or_object("task_level2", session_dir)

        # 3. LLM call to get target categories
        target_categories = llm_tasks.parse_level2_task(task_bytes)
        print(f"[Level 2] Target Categories to find: {target_categories}")
        target_norm = [c.strip().lower() for c in target_categories]

        if not target_norm:
            self.set_status("LLM returned no target categories!")
            return

        # 4. Search points one by one in greedy order
        candidate_points = self.keypoints.object_point_names()
        cur_pos = (task_kp.x, task_kp.y)
        visit_order = self.keypoints.order_greedy(candidate_points, start_xy=cur_pos, min_visibility=VISIBILITY_DEFAULT)

        found_map = {}  # category -> point_name
        for pt_name in visit_order:
            if self._cancel_event.is_set():
                return

            entry = self.keypoints.select_entry(pt_name, min_visibility=VISIBILITY_DEFAULT)
            if not entry:
                continue

            if not self._navigate_to_entry(entry):
                continue

            self.set_status(f"Level 2: Inspecting Point {pt_name}...")
            photo_bytes, _ = self._capture_screen_or_object(f"point_{pt_name}", session_dir)

            self.set_status(f"Level 2: Classifying object at Point {pt_name}...")
            cat_res = llm_tasks.classify_object(photo_bytes, pt_name)
            detected_cat = cat_res.category.strip()
            print(f"[Level 2] Point {pt_name} contains: {detected_cat}")

            # Check if this matches one of our target categories
            for target in target_norm:
                if target in detected_cat.lower() or detected_cat.lower() in target:
                    if target not in found_map:
                        found_map[target] = (pt_name, detected_cat)
                        print(f"[Level 2] MATCH! Found target '{target}' at Point {pt_name}!")

            # If both targets found, exit immediately
            if len(found_map) >= len(target_norm):
                print(f"[Level 2] All target categories discovered! Exiting early.")
                break

        # 5. Build final result string
        res_parts = [f"{orig_c.title()} at Point {pt}" for orig_c, (pt, _) in found_map.items()]
        final_str = ", ".join(res_parts) if res_parts else "None of the target objects were found."
        self.set_result(2, final_str)
        self.set_status("Level 2 Complete!", final_str)

    # -------------------------------------------------------------------------
    # LEVEL 3 WORKFLOW
    # -------------------------------------------------------------------------
    def _run_level_3(self, session_dir: Path) -> None:
        self.set_status("Level 3: Moving to Task Center (Point 0)...")

        # 1. Go to Point 0
        task_kp = self.keypoints.select_entry(TASK_POINT_NAME, min_visibility=VISIBILITY_DEFAULT)
        if not task_kp:
            raise RuntimeError("Task Point '0' not found in keypoints store!")

        if not self._navigate_to_entry(task_kp):
            return

        # 2. Take picture of task instructions
        self.set_status("Level 3: Reading task instructions on laptop screen...")
        task_bytes, _ = self._capture_screen_or_object("task_level3", session_dir)

        # 3. LLM call to get task description and high_visibility requirement
        task_info = llm_tasks.parse_level3_task(task_bytes)
        print(f"[Level 3] Task: '{task_info.task_description}'")
        print(f"[Level 3] Requires Size / Height Estimation (vis > 0.75): {task_info.use_high_visibility}")

        # Choose threshold based on task requirement
        vis_thresh = VISIBILITY_HIGH if task_info.use_high_visibility else VISIBILITY_DEFAULT

        # 4. Visit ALL object points in greedy order
        candidate_points = self.keypoints.object_point_names()
        cur_pos = (task_kp.x, task_kp.y)
        visit_order = self.keypoints.order_greedy(candidate_points, start_xy=cur_pos, min_visibility=vis_thresh)

        point_photos: dict[str, bytes] = {}
        sensor_estimates: dict[str, Optional[dict]] = {}

        for pt_name in visit_order:
            if self._cancel_event.is_set():
                return

            entry = self.keypoints.select_entry(pt_name, min_visibility=vis_thresh)
            if not entry:
                continue

            if not self._navigate_to_entry(entry):
                continue

            self.set_status(f"Level 3: Capturing photos at Point {pt_name}...")
            photo_bytes, _ = self._capture_screen_or_object(f"point_{pt_name}", session_dir)
            point_photos[pt_name] = photo_bytes

            # If height measurement is needed, query SSH size server
            if task_info.use_high_visibility:
                self.set_status(f"Level 3: Estimating object height at Point {pt_name} (SSH Server)...")
                composite_png = self.camera.snapshot_stereo_composite_png()
                if composite_png:
                    meas = self.size_link.measure(composite_png, point_name=pt_name)
                    if meas.ok and meas.primary:
                        sensor_estimates[pt_name] = meas.to_dict()
                        h = meas.primary.height_cm
                        c = meas.primary.class_name
                        print(f"[Level 3] Stereo Size: Point {pt_name} -> {c}, Height: {h:.1f} cm")

                        # If annotated image is returned, update UI "Last processed frame"
                        if meas.annotated_jpeg_b64:
                            try:
                                jpg_bytes = base64.b64decode(meas.annotated_jpeg_b64)
                                arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                                ann_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if ann_img is not None:
                                    self.set_processed_frame(ann_img, f"measured_{pt_name}.jpg")
                            except Exception:
                                pass
                    else:
                        print(f"[Level 3] [WARN] Size estimation at Point {pt_name} failed: {meas.error_message}")
                        sensor_estimates[pt_name] = None
                else:
                    sensor_estimates[pt_name] = None

        # 5. One-shot solve with LLM
        self.set_status("Level 3: Performing multimodal reasoning and inventory synthesis...")
        solution = llm_tasks.solve_level3(
            task_photo_bytes=task_bytes,
            task_info=task_info,
            point_photos=point_photos,
            sensor_estimates=sensor_estimates,
        )

        final_str = f"Answer: {solution.answer}"
        self.set_result(3, final_str)
        self.set_status("Level 3 Complete!", solution.answer)

