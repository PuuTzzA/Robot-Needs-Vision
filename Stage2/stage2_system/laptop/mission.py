"""Mission Manager and Autonomous Level State Machine.

Executes Level 1, Level 2, and Level 3 workflows:
- Level 1: Transcribe 2 points -> visit closest entries -> classify & overlay -> return to Start in parallel
- Level 2: Transcribe 2 categories -> search closest points greedily & overlay -> return to Start when found
- Level 3: Capture task screen -> visit all object points by closest distance -> async GPU server bbox/class/height
           overlay while driving -> parallel return to Start -> solve in one multimodal LLM call
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Callable, Optional
import cv2
import numpy as np

from shared.keypoints import Keypoint, KeypointStore
from shared.protocol import (
    HOME_GOAL_NAME,
    TASK_POINT_NAME,
)
from .camera import CameraManager
from .llm import tasks as llm_tasks
from .robot_link import RobotLink
from .size_link import SizeLink
from .vision_overlay import draw_classification_overlay, draw_measurement_overlay


class MissionManager:
    """Orchestrates autonomous perception and navigation workflows."""

    def __init__(
        self,
        camera: CameraManager,
        robot: RobotLink,
        size_link: SizeLink,
        keypoints: KeypointStore,
        runs_dir: Optional[Path] = None,
        camera_settle_s: float = 1.0,
    ) -> None:
        self.camera = camera
        self.robot = robot
        self.size_link = size_link
        self.keypoints = keypoints
        self.runs_dir = runs_dir or Path("runs")
        self.camera_settle_s = max(0.0, float(camera_settle_s))

        self.current_level: int = 0
        self.status_text: str = "Idle (Ready)"
        self.extra_info: str = ""
        self.level_results: dict[int, str] = {1: "", 2: "", 3: ""}

        self.last_processed_frame: Optional[np.ndarray] = None
        self.last_processed_name: str = ""
        self.processed_frame_seq: int = 0
        self.target_keypoint_id: Optional[int] = None

        self._last_settled_mono: float = 0.0
        self._fallback_home_pose: Optional[tuple[float, float, float]] = None

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
        suffix = f"({extra})" if extra else ""
        print(f"[Mission] {text} {suffix}".rstrip())
        self._notify()

    def set_processed_frame(self, frame: np.ndarray, name: str) -> None:
        with self._lock:
            self.last_processed_frame = frame.copy()
            self.last_processed_name = name
            self.processed_frame_seq += 1
        self._notify()

    def set_result(self, level: int, result_str: str) -> None:
        with self._lock:
            self.level_results[level] = result_str
        print("\n=======================================================")
        print(f"[MISSION COMPLETE] Level {level} Result:\n{result_str}")
        print("=======================================================\n")
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
        self._snapshot_start_pose_if_needed()

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
    # Helpers: Pose Tracking, Navigation, Settle Delay & Fresh Photo Capture
    # -------------------------------------------------------------------------

    def _snapshot_start_pose_if_needed(self) -> None:
        """Record a fallback home pose on the laptop in case VM controller had none latched."""
        if self._fallback_home_pose is not None:
            return
        st = self.robot.latest_status
        if st.localized:
            self._fallback_home_pose = (st.x, st.y, st.yaw_deg)

    def _current_robot_xy(self, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
        st = self.robot.latest_status
        if st.localized:
            return (st.x, st.y)
        return default

    def _navigate_to_entry(self, entry: Keypoint, timeout: float = 65.0) -> bool:
        with self._lock:
            self.target_keypoint_id = entry.id
        self.set_status(f"Navigating to {entry.name} (ID {entry.id})...")

        ok, seq, msg = self.robot.goto_keypoint(entry.id)
        if not ok:
            self.set_status(f"Navigation dispatch failed: {msg}")
            return False

        success, res_msg = self.robot.wait_for_goal(seq, timeout=timeout, cancel_event=self._cancel_event)
        if not success:
            self.set_status(f"Failed reaching {entry.name}: {res_msg}")
            return False

        # Wait ~1.0s after stopping so the top camera platform finishes shaking
        if self.camera_settle_s > 0.0:
            self.set_status(f"Stabilizing camera at Point {entry.name} ({self.camera_settle_s:.1f}s)...")
            if self._cancel_event.wait(self.camera_settle_s):
                return False

        self._last_settled_mono = time.monotonic()
        return True

    def _start_return_home(self) -> Optional[int]:
        """Dispatch non-blocking navigation back to the initial RViz Start pose."""
        if self._cancel_event.is_set():
            return None

        with self._lock:
            self.target_keypoint_id = None

        ok, seq, msg = self.robot.goto_home()
        if not ok and self._fallback_home_pose is not None:
            hx, hy, hyaw = self._fallback_home_pose
            ok, seq, msg = self.robot.goto_pose(hx, hy, hyaw, label=HOME_GOAL_NAME)

        if ok:
            print(f"[Mission] Dispatched return to Start position (Goal #{seq}).")
            return seq

        print(f"[Mission] [WARN] Could not dispatch return to Start position: {msg}")
        return None

    def _await_return_home(
        self,
        home_seq: Optional[int],
        level: int,
        summary_extra: str = "",
        timeout: float = 75.0,
    ) -> None:
        """Wait for an in-flight return-to-Start goal to complete."""
        if home_seq is None or self._cancel_event.is_set():
            self.set_status(f"Level {level} Complete!", summary_extra)
            return

        self.set_status(f"Level {level}: Returning to Start position...", summary_extra)
        ok, res_msg = self.robot.wait_for_goal(home_seq, timeout=timeout, cancel_event=self._cancel_event)
        if ok:
            self.set_status(f"Level {level} Complete! (Returned to Start)", summary_extra)
        else:
            print(f"[Mission] [WARN] Return to Start ended with: {res_msg}")
            self.set_status(f"Level {level} Complete! (Return: {res_msg})", summary_extra)

    def _save_jpeg(self, frame: np.ndarray, path: Path, quality: int = 90) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            path.write_bytes(buf.tobytes())

    def _capture_screen_or_object(
        self,
        name_tag: str,
        session_dir: Path,
        min_after_mono: Optional[float] = None,
    ) -> tuple[bytes, np.ndarray]:
        """Capture a fresh post-settle left frame, save to session_dir, and display in GUI."""
        eff_after = min_after_mono if min_after_mono is not None else self._last_settled_mono
        left_bytes, left_arr = self.camera.capture_fresh_left(
            min_after_mono=eff_after,
            timeout_s=4.5,
            quality=90,
        )

        if left_bytes is None or left_arr is None:
            if not self.camera.is_mock:
                raise RuntimeError(
                    f"Camera froze or failed to deliver a fresh post-settle frame at '{name_tag}'."
                )
            left_arr = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                left_arr,
                f"SNAPSHOT: {name_tag}",
                (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            _, buf = cv2.imencode(".jpg", left_arr)
            left_bytes = buf.tobytes()

        out_path = session_dir / f"{name_tag}.jpg"
        out_path.write_bytes(left_bytes)
        self.set_processed_frame(left_arr, out_path.name)
        return left_bytes, left_arr

    def _capture_stereo_object(
        self,
        name_tag: str,
        session_dir: Path,
        min_after_mono: Optional[float] = None,
    ) -> tuple[bytes, np.ndarray, Optional[bytes]]:
        """Capture fresh post-settle Left JPEG + Stereo Composite PNG for Level 3."""
        eff_after = min_after_mono if min_after_mono is not None else self._last_settled_mono
        left_bytes, left_arr, composite_png = self.camera.capture_fresh_stereo(
            min_after_mono=eff_after,
            timeout_s=5.0,
            quality=90,
        )

        if left_bytes is None or left_arr is None:
            if not self.camera.is_mock:
                raise RuntimeError(
                    f"Stereo camera failed to deliver a fresh post-settle frame at '{name_tag}'."
                )
            left_arr = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                left_arr,
                f"SNAPSHOT: {name_tag}",
                (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            _, jpg_buf = cv2.imencode(".jpg", left_arr)
            left_bytes = jpg_buf.tobytes()
            _, png_buf = cv2.imencode(".png", np.hstack((left_arr, left_arr)))
            composite_png = png_buf.tobytes()

        out_path = session_dir / f"{name_tag}.jpg"
        out_path.write_bytes(left_bytes)
        self.set_processed_frame(left_arr, out_path.name)
        return left_bytes, left_arr, composite_png

    # -------------------------------------------------------------------------
    # LEVEL 1 WORKFLOW
    # -------------------------------------------------------------------------
    def _run_level_1(self, session_dir: Path) -> None:
        self.set_status("Level 1: Moving to Task Center (Point 0)...")

        # 1. Go to Point 0 (closest entry)
        task_kp = self.keypoints.select_entry(
            TASK_POINT_NAME,
            robot_xy=self._current_robot_xy(),
            min_visibility=0.0,
        )
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
            home_seq = self._start_return_home()
            self._await_return_home(home_seq, level=1)
            return

        cur_xy = (task_kp.x, task_kp.y)
        ordered_points = self.keypoints.order_greedy(points_to_visit, start_xy=cur_xy, min_visibility=0.0)
        valid_points = [pt for pt in ordered_points if self.keypoints.get_by_name(pt)]
        for pt in ordered_points:
            if pt not in valid_points:
                print(f"[Level 1] [WARN] No recorded keypoint entry for Point {pt}!")

        # 4. Visit each point, classify, and overlay result on lower-left preview
        identified_objects: dict[str, str] = {}
        home_seq: Optional[int] = None

        for idx, pt_name in enumerate(valid_points):
            if self._cancel_event.is_set():
                return

            entry = self.keypoints.select_entry(pt_name, robot_xy=cur_xy, min_visibility=0.0)
            if not entry:
                continue

            if not self._navigate_to_entry(entry):
                continue
            cur_xy = (entry.x, entry.y)

            self.set_status(f"Level 1: Capturing Point {pt_name}...")
            photo_bytes, photo_frame = self._capture_screen_or_object(f"point_{pt_name}", session_dir)

            # After the last photo, immediately start driving back to Start in parallel
            if idx == len(valid_points) - 1:
                home_seq = self._start_return_home()

            self.set_status(f"Level 1: Classifying object at Point {pt_name}...")
            category_res = llm_tasks.classify_object(photo_bytes, pt_name)
            identified_objects[pt_name] = category_res.category
            print(f"[Level 1] Point {pt_name} -> {category_res.category} ({category_res.color})")

            overlaid = draw_classification_overlay(
                photo_frame,
                title=f"Point {pt_name}",
                category=category_res.category,
                color=category_res.color,
            )
            ann_name = f"classified_point_{pt_name}.jpg"
            self._save_jpeg(overlaid, session_dir / ann_name)
            self.set_processed_frame(overlaid, ann_name)

        if self._cancel_event.is_set():
            return

        # 5. Build final result string and finish return to Start
        res_parts = [f"Point {p}: {c}" for p, c in identified_objects.items()]
        final_str = ", ".join(res_parts) if res_parts else "No objects identified."
        self.set_result(1, final_str)

        if home_seq is None:
            home_seq = self._start_return_home()
        self._await_return_home(home_seq, level=1, summary_extra=final_str)

    # -------------------------------------------------------------------------
    # LEVEL 2 WORKFLOW
    # -------------------------------------------------------------------------
    def _run_level_2(self, session_dir: Path) -> None:
        self.set_status("Level 2: Moving to Task Center (Point 0)...")

        # 1. Go to Point 0
        task_kp = self.keypoints.select_entry(
            TASK_POINT_NAME,
            robot_xy=self._current_robot_xy(),
            min_visibility=0.0,
        )
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
        target_norm = [c.strip().lower() for c in target_categories if c.strip()]

        if not target_norm:
            self.set_status("LLM returned no target categories!")
            home_seq = self._start_return_home()
            self._await_return_home(home_seq, level=2)
            return

        # 4. Search points one by one in greedy closest-point order (no visibility filtering)
        candidate_points = self.keypoints.object_point_names()
        cur_xy = (task_kp.x, task_kp.y)
        visit_order = self.keypoints.order_greedy(candidate_points, start_xy=cur_xy, min_visibility=0.0)
        valid_points = [pt for pt in visit_order if self.keypoints.get_by_name(pt)]

        found_map: dict[str, tuple[str, str]] = {}
        home_seq: Optional[int] = None

        for idx, pt_name in enumerate(valid_points):
            if self._cancel_event.is_set():
                return

            entry = self.keypoints.select_entry(pt_name, robot_xy=cur_xy, min_visibility=0.0)
            if not entry:
                continue

            if not self._navigate_to_entry(entry):
                continue
            cur_xy = (entry.x, entry.y)

            self.set_status(f"Level 2: Capturing Point {pt_name}...")
            photo_bytes, photo_frame = self._capture_screen_or_object(f"point_{pt_name}", session_dir)

            # If this is the final candidate point in the venue, return home in parallel with classification
            if idx == len(valid_points) - 1:
                home_seq = self._start_return_home()

            self.set_status(f"Level 2: Classifying object at Point {pt_name}...")
            cat_res = llm_tasks.classify_object(photo_bytes, pt_name)
            detected_cat = cat_res.category.strip()
            print(f"[Level 2] Point {pt_name} contains: {detected_cat} ({cat_res.color})")

            overlaid = draw_classification_overlay(
                photo_frame,
                title=f"Point {pt_name}",
                category=detected_cat,
                color=cat_res.color,
            )
            ann_name = f"classified_point_{pt_name}.jpg"
            self._save_jpeg(overlaid, session_dir / ann_name)
            self.set_processed_frame(overlaid, ann_name)

            # Check if this matches one of our target categories
            for target in target_norm:
                if target in detected_cat.lower() or detected_cat.lower() in target:
                    if target not in found_map:
                        found_map[target] = (pt_name, detected_cat)
                        print(f"[Level 2] MATCH! Found target '{target}' at Point {pt_name}!")

            # If all targets found, exit immediately and return to Start
            if len(found_map) >= len(target_norm):
                print("[Level 2] All target categories discovered! Exiting early.")
                break

        if self._cancel_event.is_set():
            return

        # 5. Build final result string and return to Start
        res_parts = [f"{orig_c.title()} at Point {pt}" for orig_c, (pt, _) in found_map.items()]
        final_str = ", ".join(res_parts) if res_parts else "None of the target objects were found."
        self.set_result(2, final_str)

        if home_seq is None:
            home_seq = self._start_return_home()
        self._await_return_home(home_seq, level=2, summary_extra=final_str)

    # -------------------------------------------------------------------------
    # LEVEL 3 WORKFLOW
    # -------------------------------------------------------------------------
    def _run_level_3(self, session_dir: Path) -> None:
        self.set_status("Level 3: Moving to Task Center (Point 0)...")

        # 1. Go to Point 0
        task_kp = self.keypoints.select_entry(
            TASK_POINT_NAME,
            robot_xy=self._current_robot_xy(),
            min_visibility=0.0,
        )
        if not task_kp:
            raise RuntimeError("Task Point '0' not found in keypoints store!")

        if not self._navigate_to_entry(task_kp):
            return

        # 2. Capture task screen photo ONLY (no initial LLM API call; solved in 1 shot at end)
        self.set_status("Level 3: Capturing task instructions at Point 0...")
        task_bytes, _ = self._capture_screen_or_object("task_level3", session_dir)

        # 3. Order ALL object points greedily by closest Euclidean distance (no visibility filtering)
        candidate_points = self.keypoints.object_point_names()
        cur_xy = (task_kp.x, task_kp.y)
        visit_order = self.keypoints.order_greedy(candidate_points, start_xy=cur_xy, min_visibility=0.0)
        valid_points = [pt for pt in visit_order if self.keypoints.get_by_name(pt)]
        print(f"[Level 3] Closest-point visit schedule ({len(valid_points)} points): {valid_points}")

        point_photos: dict[str, bytes] = {}
        sensor_estimates: dict[str, Optional[dict]] = {}
        est_lock = threading.Lock()
        gpu_futures: list[Future[None]] = []
        home_seq: Optional[int] = None

        def _async_gpu_task(pt: str, raw_bgr: np.ndarray, png_payload: Optional[bytes]) -> None:
            if self._cancel_event.is_set():
                return
            if not png_payload:
                with est_lock:
                    sensor_estimates[pt] = None
                print(f"[Level 3] [WARN] No stereo pair for Point {pt}; skipping GPU measurement.")
                return

            meas = self.size_link.measure(png_payload, point_name=pt)
            if self._cancel_event.is_set():
                return

            if meas.ok and meas.primary is not None:
                with est_lock:
                    sensor_estimates[pt] = meas.to_dict()
                h_val = meas.primary.height_cm
                h_txt = f"{h_val:.1f} cm" if h_val is not None else "n/a"
                print(
                    f"[Level 3] Async GPU (Point {pt}): class={meas.primary.class_name}, "
                    f"height={h_txt}, bbox={meas.primary.bbox_px}"
                )
            else:
                with est_lock:
                    sensor_estimates[pt] = meas.to_dict() if meas.ok else None
                err_txt = meas.error_message or "no object detected"
                print(f"[Level 3] [WARN] Async GPU (Point {pt}): {err_txt}")

            ann_img = draw_measurement_overlay(raw_bgr, meas, point_name=pt)
            ann_name = f"measured_{pt}.jpg"
            self._save_jpeg(ann_img, session_dir / ann_name)
            self.set_processed_frame(ann_img, ann_name)

        gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Level3GPU")
        try:
            for idx, pt_name in enumerate(valid_points):
                if self._cancel_event.is_set():
                    return

                entry = self.keypoints.select_entry(pt_name, robot_xy=cur_xy, min_visibility=0.0)
                if not entry:
                    continue

                if not self._navigate_to_entry(entry):
                    continue
                cur_xy = (entry.x, entry.y)

                self.set_status(f"Level 3: Capturing stereo pair at Point {pt_name}...")
                photo_bytes, photo_frame, composite_png = self._capture_stereo_object(
                    f"point_{pt_name}", session_dir
                )
                point_photos[pt_name] = photo_bytes

                # Dispatch GPU estimation asynchronously while immediately driving to next point / Start
                fut = gpu_pool.submit(_async_gpu_task, pt_name, photo_frame.copy(), composite_png)
                gpu_futures.append(fut)

                if idx == len(valid_points) - 1:
                    home_seq = self._start_return_home()

            if self._cancel_event.is_set():
                return

            if home_seq is None:
                home_seq = self._start_return_home()

            # Wait for any remaining async GPU calls while robot is already driving back to Start
            pending = [f for f in gpu_futures if not f.done()]
            if pending:
                self.set_status(f"Level 3: Waiting for {len(pending)} async GPU measurement(s)...")
            while any(not f.done() for f in gpu_futures):
                if self._cancel_event.is_set():
                    return
                time.sleep(0.05)

            for f in gpu_futures:
                exc = f.exception()
                if exc is not None:
                    print(f"[Level 3] [WARN] Async GPU worker exception: {exc}")
        finally:
            gpu_pool.shutdown(wait=False)

        if self._cancel_event.is_set():
            return

        # 4. Single one-shot multimodal LLM solve (transcribes task_level3.jpg + solves from photos & GPU data)
        self.set_status("Level 3: Solving task in one shot (LLM + GPU estimates)...")
        solution = llm_tasks.solve_level3(
            task_photo_bytes=task_bytes,
            point_photos=point_photos,
            sensor_estimates=sensor_estimates,
        )
        if solution.task_description:
            print(f"[Level 3] Transcribed Task: '{solution.task_description}'")

        final_str = f"Answer: {solution.answer}"
        self.set_result(3, final_str)
        self._await_return_home(home_seq, level=3, summary_extra=solution.answer)
