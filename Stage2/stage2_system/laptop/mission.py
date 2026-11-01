"""Mission Manager and Autonomous Level State Machine.

Executes Level 1, Level 2, and Level 3 workflows:
- Level 1: Transcribe 2 points -> visit closest entries -> classify & overlay -> return to Start in parallel
- Level 2: Transcribe 2 categories -> search closest points greedily & overlay -> return to Start when found
- Level 3: Capture task screen + async legibility check (single retake at Point 0 if illegible) ->
           visit all object points by closest distance -> show raw photo immediately, then replace with
           disparity/depth + bbox + size overlay from GPU server -> parallel return to Start ->
           solve in one multimodal LLM call with a full-sentence answer
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import math
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
from .llm.schemas import TaskLegibility
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
        goal_timeout_s: float = 60.0,
        stuck_timeout_s: float = 10.0,
        max_point_retries: int = 1,
    ) -> None:
        self.camera = camera
        self.robot = robot
        self.size_link = size_link
        self.keypoints = keypoints
        self.runs_dir = runs_dir or Path("runs")
        self.camera_settle_s = max(0.0, float(camera_settle_s))
        self.goal_timeout_s = max(0.5, float(goal_timeout_s))
        self.stuck_timeout_s = max(0.1, float(stuck_timeout_s))
        self.max_point_retries = max(0, int(max_point_retries))

        self.current_level: int = 0
        self.status_text: str = "Idle (Ready)"
        self.extra_info: str = ""
        self.last_nav_error: str = ""
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

    def get_processed_frame(self) -> tuple[Optional[np.ndarray], str, int]:
        """Atomically return (frame, name, seq) for thread-safe GUI polling."""
        with self._lock:
            return self.last_processed_frame, self.last_processed_name, self.processed_frame_seq

    def replace_processed_frame_for_point(self, frame: np.ndarray, name: str, point_name: str) -> None:
        """Replace bottom-left processed preview with GPU result unless a newer point's raw photo is awaiting GPU."""
        expected_raw = f"point_{point_name}.jpg"
        updated = False
        with self._lock:
            cur_name = self.last_processed_name
            if not cur_name.startswith("point_") or cur_name == expected_raw:
                self.last_processed_frame = frame.copy()
                self.last_processed_name = name
                self.processed_frame_seq += 1
                updated = True
        if updated:
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
    # Helpers: Pose Tracking, Navigation, Recovery/Retry, Settle & Capture
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

    def _navigate_to_entry(
        self,
        entry: Keypoint,
        timeout: Optional[float] = None,
        status_text: Optional[str] = None,
        settle: bool = True,
    ) -> bool:
        eff_timeout = self.goal_timeout_s if timeout is None else float(timeout)
        with self._lock:
            self.target_keypoint_id = entry.id
        self.set_status(status_text or f"Navigating to {entry.name} (ID {entry.id})...")

        ok, seq, msg = self.robot.goto_keypoint(entry.id)
        if not ok:
            self.last_nav_error = msg
            self.set_status(f"Failed to get to {entry.name}: {msg}")
            return False

        success, res_msg = self.robot.wait_for_goal(
            seq,
            timeout=eff_timeout,
            stuck_timeout=self.stuck_timeout_s,
            cancel_event=self._cancel_event,
        )
        if not success:
            self.last_nav_error = res_msg
            if not self._cancel_event.is_set():
                self.set_status(f"Failed to get to {entry.name}: {res_msg}")
            return False

        self.last_nav_error = ""
        # Wait ~1.0s after stopping so the top camera platform finishes shaking
        if settle and self.camera_settle_s > 0.0:
            self.set_status(f"Stabilizing camera at Point {entry.name} ({self.camera_settle_s:.1f}s)...")
            if self._cancel_event.wait(self.camera_settle_s):
                return False

        self._last_settled_mono = time.monotonic()
        return True

    def _select_closest_other_keypoint(
        self,
        failed_name: str,
        robot_xy: tuple[float, float],
        failed_entry: Optional[Keypoint] = None,
    ) -> Optional[Keypoint]:
        """Select the closest keypoint with a different name than `failed_name`."""
        other_kps = [kp for kp in self.keypoints.get_all() if kp.name != failed_name]
        if not other_kps:
            return None

        # Prefer keypoints meaningfully separated (> 0.20m) from the failed pose when available
        if failed_entry is not None:
            distinct = [
                kp
                for kp in other_kps
                if math.hypot(kp.x - failed_entry.x, kp.y - failed_entry.y) > 0.20
            ]
            if distinct:
                other_kps = distinct

        return min(
            other_kps,
            key=lambda kp: math.hypot(kp.x - robot_xy[0], kp.y - robot_xy[1]),
        )

    def _handle_failed_keypoint(
        self,
        failed_pt: str,
        failed_entry: Keypoint,
        pending: list[str],
        retry_counts: dict[str, int],
        cur_xy: tuple[float, float],
    ) -> tuple[tuple[float, float], Optional[str]]:
        """Apply navigation failure recovery policy:

        - If `pending` still has keypoints (next is `Y`): announce
          `"Failed to get to X, continuing with Y"`, schedule `X` right after `Y`,
          and keep that message visible while driving to `Y`.
        - If `X` was the last keypoint (`pending` is empty): go to the closest other
          keypoint `Y` (`"Failed to get to X, continuing with Y"`), then return to `X`.
        """
        cur_xy = self._current_robot_xy(default=cur_xy)
        if self._cancel_event.is_set():
            return cur_xy, None

        attempts = retry_counts.get(failed_pt, 0)
        can_retry = attempts < self.max_point_retries

        if pending:
            next_pt = pending[0]
            msg = f"Failed to get to {failed_pt}, continuing with {next_pt}"
            self.set_status(msg)
            if can_retry:
                retry_counts[failed_pt] = attempts + 1
                pending.insert(1, failed_pt)
            else:
                print(
                    f"[Mission] [WARN] Skipping Point {failed_pt} after {attempts + 1} failed attempt(s)."
                )
            return cur_xy, msg

        if not can_retry:
            print(
                f"[Mission] [WARN] Giving up on last Point {failed_pt} after {attempts + 1} failed attempt(s)."
            )
            return cur_xy, None

        detour_kp = self._select_closest_other_keypoint(
            failed_name=failed_pt,
            robot_xy=cur_xy,
            failed_entry=failed_entry,
        )
        retry_counts[failed_pt] = attempts + 1
        if detour_kp is None:
            retry_msg = f"Failed to get to {failed_pt}, retrying {failed_pt}"
            self.set_status(retry_msg)
            pending.insert(0, failed_pt)
            return cur_xy, retry_msg

        detour_msg = f"Failed to get to {failed_pt}, continuing with {detour_kp.name}"
        self.set_status(detour_msg)
        detour_ok = self._navigate_to_entry(detour_kp, status_text=detour_msg, settle=False)
        if self._cancel_event.is_set():
            return cur_xy, None
        if detour_ok:
            cur_xy = (detour_kp.x, detour_kp.y)
        else:
            cur_xy = self._current_robot_xy(default=cur_xy)

        pending.insert(0, failed_pt)
        return cur_xy, f"Returning to {failed_pt} after visiting {detour_kp.name}..."

    def _navigate_to_initial_task_point(self, level: int) -> Optional[Keypoint]:
        """Navigate to Task Point '0' at level start, recovering via closest other keypoint if blocked."""
        cur_xy = self._current_robot_xy()
        retry_counts: dict[str, int] = {}
        pending: list[str] = [TASK_POINT_NAME]
        next_status: Optional[str] = f"Level {level}: Moving to Task Center (Point 0)..."

        while pending and not self._cancel_event.is_set():
            pt_name = pending.pop(0)
            task_kp = self.keypoints.select_entry(pt_name, robot_xy=cur_xy, min_visibility=0.0)
            if not task_kp:
                raise RuntimeError("Task Point '0' not found in keypoints store!")

            if self._navigate_to_entry(task_kp, status_text=next_status):
                return task_kp

            next_status = None
            cur_xy, next_status = self._handle_failed_keypoint(
                failed_pt=pt_name,
                failed_entry=task_kp,
                pending=pending,
                retry_counts=retry_counts,
                cur_xy=cur_xy,
            )

        return None

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
        timeout: Optional[float] = None,
    ) -> None:
        """Wait for an in-flight return-to-Start goal to complete."""
        if home_seq is None or self._cancel_event.is_set():
            self.set_status(f"Level {level} Complete!", summary_extra)
            return

        eff_timeout = self.goal_timeout_s if timeout is None else float(timeout)
        self.set_status(f"Level {level}: Returning to Start position...", summary_extra)
        ok, res_msg = self.robot.wait_for_goal(
            home_seq,
            timeout=eff_timeout,
            stuck_timeout=self.stuck_timeout_s,
            cancel_event=self._cancel_event,
        )
        if ok:
            self.set_status(f"Level {level} Complete! (Returned to Start)", summary_extra)
        else:
            print(f"[Mission] [WARN] Return to Start ended with: {res_msg}")
            self.set_status(f"Level {level} Complete! (Return failed: {res_msg})", summary_extra)

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
        """Capture a fresh post-settle left frame, immediately show in GUI, and save to session_dir."""
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
        self.set_processed_frame(left_arr, out_path.name)
        out_path.write_bytes(left_bytes)
        return left_bytes, left_arr

    def _capture_stereo_object(
        self,
        name_tag: str,
        session_dir: Path,
        min_after_mono: Optional[float] = None,
    ) -> tuple[bytes, np.ndarray, Optional[bytes]]:
        """Capture fresh post-settle Left JPEG + Stereo Composite PNG (legacy helper)."""
        left_bytes, left_arr, right_arr = self._capture_stereo_frames_immediate(
            name_tag=name_tag,
            session_dir=session_dir,
            min_after_mono=min_after_mono,
        )
        composite_png: Optional[bytes] = None
        if right_arr is not None:
            if left_arr.shape[:2] != right_arr.shape[:2]:
                right_arr = cv2.resize(right_arr, (left_arr.shape[1], left_arr.shape[0]))
            ok_png, png_buf = cv2.imencode(".png", np.hstack((left_arr, right_arr)))
            if ok_png:
                composite_png = png_buf.tobytes()
        return left_bytes, left_arr, composite_png

    def _capture_stereo_frames_immediate(
        self,
        name_tag: str,
        session_dir: Path,
        min_after_mono: Optional[float] = None,
    ) -> tuple[bytes, np.ndarray, Optional[np.ndarray]]:
        """Capture fresh post-settle stereo frames, immediately display raw picture in UI, and save Left JPEG."""
        eff_after = min_after_mono if min_after_mono is not None else self._last_settled_mono
        left_arr, right_arr = self.camera.capture_fresh_stereo_frames(
            min_after_mono=eff_after,
            timeout_s=5.0,
        )

        if left_arr is None:
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
            right_arr = left_arr.copy()

        out_path = session_dir / f"{name_tag}.jpg"
        # Show the raw captured picture immediately in bottom-left UI before GPU round-trip
        self.set_processed_frame(left_arr, out_path.name)

        ok_jpg, jpg_buf = cv2.imencode(".jpg", left_arr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok_jpg:
            raise RuntimeError(f"Failed encoding JPEG for '{name_tag}'.")
        left_bytes = jpg_buf.tobytes()
        out_path.write_bytes(left_bytes)
        return left_bytes, left_arr, right_arr

    # -------------------------------------------------------------------------
    # LEVEL 1 WORKFLOW
    # -------------------------------------------------------------------------
    def _run_level_1(self, session_dir: Path) -> None:
        # 1. Go to Point 0 (closest entry)
        task_kp = self._navigate_to_initial_task_point(level=1)
        if not task_kp:
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

        cur_xy = self._current_robot_xy(default=(task_kp.x, task_kp.y))
        ordered_points = self.keypoints.order_greedy(points_to_visit, start_xy=cur_xy, min_visibility=0.0)
        valid_points = [pt for pt in ordered_points if self.keypoints.get_by_name(pt)]
        for pt in ordered_points:
            if pt not in valid_points:
                print(f"[Level 1] [WARN] No recorded keypoint entry for Point {pt}!")

        # 4. Visit each point (with retry/detour on failure), classify, and overlay result
        identified_objects: dict[str, str] = {}
        home_seq: Optional[int] = None
        pending: list[str] = list(valid_points)
        retry_counts: dict[str, int] = {}
        next_nav_status: Optional[str] = None

        while pending:
            if self._cancel_event.is_set():
                return

            pt_name = pending.pop(0)
            entry = self.keypoints.select_entry(pt_name, robot_xy=cur_xy, min_visibility=0.0)
            if not entry:
                continue

            nav_ok = self._navigate_to_entry(entry, status_text=next_nav_status)
            next_nav_status = None
            if not nav_ok:
                cur_xy, next_nav_status = self._handle_failed_keypoint(
                    failed_pt=pt_name,
                    failed_entry=entry,
                    pending=pending,
                    retry_counts=retry_counts,
                    cur_xy=cur_xy,
                )
                continue

            cur_xy = (entry.x, entry.y)

            self.set_status(f"Level 1: Capturing Point {pt_name}...")
            photo_bytes, photo_frame = self._capture_screen_or_object(f"point_{pt_name}", session_dir)

            # After the final required photo (no pending retries left), start driving back to Start in parallel
            if not pending:
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
        # 1. Go to Point 0
        task_kp = self._navigate_to_initial_task_point(level=2)
        if not task_kp:
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

        # 4. Search points one by one in greedy closest-point order (with retry/detour on failure)
        candidate_points = self.keypoints.object_point_names()
        cur_xy = self._current_robot_xy(default=(task_kp.x, task_kp.y))
        visit_order = self.keypoints.order_greedy(candidate_points, start_xy=cur_xy, min_visibility=0.0)
        valid_points = [pt for pt in visit_order if self.keypoints.get_by_name(pt)]

        found_map: dict[str, tuple[str, str]] = {}
        home_seq: Optional[int] = None
        pending: list[str] = list(valid_points)
        retry_counts: dict[str, int] = {}
        next_nav_status: Optional[str] = None

        while pending:
            if self._cancel_event.is_set():
                return

            pt_name = pending.pop(0)
            entry = self.keypoints.select_entry(pt_name, robot_xy=cur_xy, min_visibility=0.0)
            if not entry:
                continue

            nav_ok = self._navigate_to_entry(entry, status_text=next_nav_status)
            next_nav_status = None
            if not nav_ok:
                cur_xy, next_nav_status = self._handle_failed_keypoint(
                    failed_pt=pt_name,
                    failed_entry=entry,
                    pending=pending,
                    retry_counts=retry_counts,
                    cur_xy=cur_xy,
                )
                continue

            cur_xy = (entry.x, entry.y)

            self.set_status(f"Level 2: Capturing Point {pt_name}...")
            photo_bytes, photo_frame = self._capture_screen_or_object(f"point_{pt_name}", session_dir)

            # If this is the final candidate point in the venue (no pending points left), return home in parallel
            if not pending:
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
        # 1. Go to Point 0
        task_kp = self._navigate_to_initial_task_point(level=3)
        if not task_kp:
            return

        # 2. Capture task screen photo and immediately launch async LLM legibility check while driving
        self.set_status("Level 3: Capturing task instructions at Point 0...")
        task_bytes, _ = self._capture_screen_or_object("task_level3", session_dir)

        llm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Level3Legibility")
        legibility_future: Future[TaskLegibility] = llm_pool.submit(
            llm_tasks.check_level3_legibility,
            task_bytes,
        )
        legibility_handled = False
        retake_scheduled = False

        # 3. Order ALL object points greedily by closest Euclidean distance (no visibility filtering)
        candidate_points = self.keypoints.object_point_names()
        cur_xy = self._current_robot_xy(default=(task_kp.x, task_kp.y))
        visit_order = self.keypoints.order_greedy(candidate_points, start_xy=cur_xy, min_visibility=0.0)
        valid_points = [pt for pt in visit_order if self.keypoints.get_by_name(pt)]
        print(f"[Level 3] Closest-point visit schedule ({len(valid_points)} points): {valid_points}")

        point_photos: dict[str, bytes] = {}
        sensor_estimates: dict[str, Optional[dict]] = {}
        est_lock = threading.Lock()
        gpu_futures: list[Future[None]] = []
        home_seq: Optional[int] = None
        pending: list[str] = list(valid_points)
        retry_counts: dict[str, int] = {}
        next_nav_status: Optional[str] = None

        def _consume_legibility_if_ready(block: bool = False) -> None:
            nonlocal legibility_handled, retake_scheduled
            if legibility_handled:
                return
            if not block and not legibility_future.done():
                return
            if block and not legibility_future.done():
                self.set_status("Level 3: Waiting for async task legibility check...")
                while not legibility_future.done():
                    if self._cancel_event.is_set():
                        return
                    time.sleep(0.05)

            legibility_handled = True
            try:
                leg_res = legibility_future.result()
            except Exception as exc:
                print(
                    f"[Level 3] [WARN] Async legibility check failed ({exc}); proceeding with initial task photo."
                )
                return

            if leg_res.legible:
                snippet = f" ('{leg_res.transcribed_text}')" if leg_res.transcribed_text else ""
                print(f"[Level 3] Async legibility check passed: question is legible{snippet}.")
            elif not retake_scheduled:
                retake_scheduled = True
                reason_str = leg_res.reason or "illegible text"
                print(
                    f"[Level 3] [WARN] Task photo not legible ({reason_str}). "
                    "Scheduling single return to Point 0 for retake."
                )
                if TASK_POINT_NAME not in pending:
                    pending.insert(0, TASK_POINT_NAME)

        def _async_gpu_task(
            pt: str,
            raw_bgr: np.ndarray,
            right_bgr: Optional[np.ndarray],
        ) -> None:
            if self._cancel_event.is_set():
                return
            if right_bgr is None:
                with est_lock:
                    sensor_estimates[pt] = None
                print(f"[Level 3] [WARN] No stereo pair for Point {pt}; skipping GPU measurement.")
                return

            if raw_bgr.shape[:2] != right_bgr.shape[:2]:
                right_bgr = cv2.resize(right_bgr, (raw_bgr.shape[1], raw_bgr.shape[0]))
            composite = np.hstack((raw_bgr, right_bgr))
            ok_png, png_buf = cv2.imencode(".png", composite)
            if not ok_png:
                with est_lock:
                    sensor_estimates[pt] = None
                print(f"[Level 3] [WARN] Failed encoding stereo PNG for Point {pt}.")
                return

            meas = self.size_link.measure(png_buf.tobytes(), point_name=pt)
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
            self.replace_processed_frame_for_point(ann_img, ann_name, pt)

        gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Level3GPU")
        try:
            while True:
                while pending:
                    if self._cancel_event.is_set():
                        return

                    pt_name = pending.pop(0)
                    entry = self.keypoints.select_entry(pt_name, robot_xy=cur_xy, min_visibility=0.0)
                    if not entry:
                        continue

                    if pt_name == TASK_POINT_NAME and next_nav_status is None:
                        nav_prompt: Optional[str] = (
                            "Level 3: Question not legible — returning to Point 0 for retake..."
                        )
                    else:
                        nav_prompt = next_nav_status

                    nav_ok = self._navigate_to_entry(entry, status_text=nav_prompt)
                    next_nav_status = None
                    if not nav_ok:
                        _consume_legibility_if_ready(block=False)
                        cur_xy, next_nav_status = self._handle_failed_keypoint(
                            failed_pt=pt_name,
                            failed_entry=entry,
                            pending=pending,
                            retry_counts=retry_counts,
                            cur_xy=cur_xy,
                        )
                        continue

                    cur_xy = (entry.x, entry.y)

                    if pt_name == TASK_POINT_NAME:
                        # Single retake at Point 0 — do NOT check legibility a second time
                        self.set_status("Level 3: Retaking task instructions photo at Point 0...")
                        task_bytes, _ = self._capture_screen_or_object("task_level3_retake", session_dir)
                        continue

                    self.set_status(f"Level 3: Capturing stereo pair at Point {pt_name}...")
                    photo_bytes, photo_frame, right_frame = self._capture_stereo_frames_immediate(
                        f"point_{pt_name}",
                        session_dir,
                    )
                    point_photos[pt_name] = photo_bytes

                    # Send stereo picture to GPU server asynchronously while driving to next point
                    fut = gpu_pool.submit(
                        _async_gpu_task,
                        pt_name,
                        photo_frame.copy(),
                        right_frame.copy() if right_frame is not None else None,
                    )
                    gpu_futures.append(fut)

                    # Check if async legibility result arrived while we were driving/capturing
                    _consume_legibility_if_ready(block=False)

                    # If legibility check already finished and no more points remain, return to Start in parallel
                    if not pending and legibility_handled and home_seq is None:
                        home_seq = self._start_return_home()

                if self._cancel_event.is_set():
                    return

                # Ensure async legibility check has finished before returning to Start; if illegible, loop again for Point 0
                if not legibility_handled:
                    _consume_legibility_if_ready(block=True)
                    if pending:
                        continue
                break

            if self._cancel_event.is_set():
                return

            if home_seq is None:
                home_seq = self._start_return_home()

            # Wait for any remaining async GPU calls while robot is already driving back to Start
            pending_gpu = [f for f in gpu_futures if not f.done()]
            if pending_gpu:
                self.set_status(f"Level 3: Waiting for {len(pending_gpu)} async GPU measurement(s)...")
            while any(not f.done() for f in gpu_futures):
                if self._cancel_event.is_set():
                    return
                time.sleep(0.05)

            for f in gpu_futures:
                exc = f.exception()
                if exc is not None:
                    print(f"[Level 3] [WARN] Async GPU worker exception: {exc}")
        finally:
            llm_pool.shutdown(wait=False)
            gpu_pool.shutdown(wait=False)

        if self._cancel_event.is_set():
            return

        # 4. Single one-shot multimodal LLM solve at the very end (using retaken task_bytes if retaken)
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
