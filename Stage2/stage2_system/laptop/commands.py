"""Command parser and execution engine for Windows Mission Control."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
import cv2
import numpy as np

if TYPE_CHECKING:
    from .mission import MissionManager

from .llm import llm_api
from .llm import tasks as llm_tasks
from .vision_overlay import draw_classification_overlay, draw_measurement_overlay


HELP_BANNER = """
================================================================================
AVAILABLE MISSION CONTROL COMMANDS:
--------------------------------------------------------------------------------
Level <1|2|3>          - Start autonomous execution of specified level
Restart <1|2|3>        - Abort, clear results, and restart specified level
Stop                   - Abort active navigation and cancel current level
Goto <Point>           - Send robot to closest keypoint entry (e.g. 'Goto B' or 'Goto 0')
Goto #<id>             - Send robot to specific unique keypoint ID (e.g. 'Goto #2')
Home                   - Navigate robot back to its initial RViz Start position
Ping <VM|SSH|LLM>      - Test round-trip latency to specified subsystem
W / A / S / D          - Teleop robot: forward, left, backward, right burst (0.6s)
Space                  - Stop robot teleop motion immediately
Photo                  - Capture fresh camera frame and display in 'Last Processed Frame'
Measure [hint]         - Request stereo height & bbox from SSH GPU server + overlay preview
Classify               - Classify current camera frame via LLM + overlay result on preview
CamReset               - Force-reopen USB camera streams if a webcam glitches
Points / List          - Display all loaded keypoints and their visibility values
Status                 - Print current subsystem, camera, and robot pose status
Help                   - Show this command reference
================================================================================
"""


class CommandDispatcher:
    """Parses and executes user commands typed into GUI command input or terminal."""

    ASYNC_COMMANDS = {"PING", "PHOTO", "MEASURE", "CLASSIFY", "CAMRESET", "RESETCAM"}

    def __init__(self, mission: MissionManager) -> None:
        self.m = mission

    def _grab_left_for_command(self) -> tuple[ bytes | None, np.ndarray | None ]:
        """Grab a fresh left frame when camera loop is active, falling back to latest buffer."""
        left_bytes, left_arr = self.m.camera.capture_fresh_left(timeout_s=2.5, quality=90)
        if left_bytes is not None and left_arr is not None:
            return left_bytes, left_arr

        left_arr_fallback, _, _ = self.m.camera.get_latest_pair()
        left_bytes_fallback = self.m.camera.snapshot_left_jpeg(quality=90)
        if left_bytes_fallback is not None and left_arr_fallback is None:
            arr = np.frombuffer(left_bytes_fallback, dtype=np.uint8)
            left_arr_fallback = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return left_bytes_fallback, left_arr_fallback

    def execute(self, line: str) -> str:
        line = line.strip()
        if not line:
            return ""

        parts = line.split()
        cmd = parts[0].upper()
        args = parts[1:]

        try:
            if cmd == "LEVEL":
                if not args or args[0] not in ("1", "2", "3"):
                    return "[ERROR] Usage: Level <1|2|3>"
                lvl = int(args[0])
                ok = self.m.start_level(lvl)
                return f"Started Level {lvl}" if ok else f"Could not start Level {lvl} (mission running)"

            elif cmd == "RESTART":
                if not args or args[0] not in ("1", "2", "3"):
                    return "[ERROR] Usage: Restart <1|2|3>"
                lvl = int(args[0])
                self.m.restart_level(lvl)
                return f"Restarted Level {lvl}"

            elif cmd in ("STOP", "CANCEL"):
                self.m.stop()
                return "Mission and navigation stopped."

            elif cmd == "HOME":
                ok, _seq, msg = self.m.robot.goto_home()
                return "Navigating back to initial Start position..." if ok else f"Home failed: {msg}"

            elif cmd == "GOTO":
                if not args:
                    return "[ERROR] Usage: Goto <PointName> (e.g. 'Goto B'), Goto Start, or Goto #<id>"
                target = args[0]
                if target.upper() in ("START", "HOME"):
                    self.m.announcer.play_sequence(["returning_to_start.mp3"], fallback_text="Returning to start position")
                    ok, _seq, msg = self.m.robot.goto_home()
                    return "Navigating back to initial Start position..." if ok else f"Goto Start failed: {msg}"
                if target.startswith("#"):
                    try:
                        kp_id = int(target[1:])
                        kp = self.m.keypoints.get_by_id(kp_id)
                        if not kp:
                            return f"[ERROR] Keypoint ID {kp_id} not found."
                        self.m.announcer.announce_going_to_point(kp.name)
                        ok, _seq, msg = self.m.robot.goto_keypoint(kp.id)
                        return f"Navigating to Keypoint ID {kp.id} ('{kp.name}')..." if ok else f"Goto failed: {msg}"
                    except ValueError:
                        return f"[ERROR] Invalid ID '{target}'"
                else:
                    pt_name = target.upper()
                    cur_pose = (self.m.robot.latest_status.x, self.m.robot.latest_status.y)
                    kp = self.m.keypoints.select_entry(pt_name, robot_xy=cur_pose, min_visibility=0.0)
                    if not kp:
                        return f"[ERROR] Keypoint '{pt_name}' not found in keypoint store."
                    self.m.announcer.announce_going_to_point(pt_name)
                    ok, _seq, msg = self.m.robot.goto_keypoint(kp.id)
                    return f"Navigating to Point {pt_name} (ID {kp.id})..." if ok else f"Goto failed: {msg}"

            elif cmd in ("W", "A", "S", "D", "SPACE"):
                key_map = {"W": "w", "A": "a", "S": "s", "D": "d", "SPACE": "space"}
                k = key_map[cmd]
                ok = self.m.robot.teleop(k)
                return f"Teleop: {cmd}" if ok else "Teleop command failed"

            elif cmd == "PING":
                target = args[0].upper() if args else "VM"
                if target == "VM":
                    ok, rtt, info = self.m.robot.ping()
                    return f"Ping VM ({info}): {'OK' if ok else 'FAIL'} ({rtt:.1f} ms)" if ok else f"Ping VM failed: {info}"
                elif target == "SSH":
                    ok, rtt, info = self.m.size_link.ping()
                    return f"Ping SSH ({info}): {'OK' if ok else 'FAIL'} ({rtt:.1f} ms)" if ok else f"Ping SSH failed: {info}"
                elif target == "LLM":
                    ok, info = llm_api.health_check()
                    return f"Ping LLM: {'OK' if ok else 'FAIL'} ({info})"
                return "[ERROR] Usage: Ping <VM|SSH|LLM>"

            elif cmd in ("CAMRESET", "RESETCAM"):
                return self.m.camera.reset_cameras()

            elif cmd == "PHOTO":
                _left_bytes, left_arr = self._grab_left_for_command()
                if left_arr is not None:
                    ts = time.strftime("%H%M%S")
                    self.m.set_processed_frame(left_arr, f"snapshot_{ts}.jpg")
                    return f"Captured photo (snapshot_{ts}.jpg)"
                return "Failed capturing photo (no active camera frame)"

            elif cmd == "MEASURE":
                hint = args[0] if args else None
                left_arr, right_arr = self.m.camera.capture_fresh_stereo_frames(timeout_s=3.0)
                ts = time.strftime("%H%M%S")
                if left_arr is not None:
                    self.m.set_processed_frame(left_arr, f"measure_capture_{ts}.jpg")

                composite_png: bytes | None = None
                if left_arr is not None and right_arr is not None:
                    if left_arr.shape[:2] != right_arr.shape[:2]:
                        right_arr = cv2.resize(right_arr, (left_arr.shape[1], left_arr.shape[0]))
                    ok_png, png_buf = cv2.imencode(".png", np.hstack((left_arr, right_arr)))
                    if ok_png:
                        composite_png = png_buf.tobytes()

                if not composite_png:
                    composite_png = self.m.camera.snapshot_stereo_composite_png()
                    if left_arr is None:
                        left_arr, _, _ = self.m.camera.get_latest_pair()
                        if left_arr is not None:
                            self.m.set_processed_frame(left_arr, f"measure_capture_{ts}.jpg")
                if not composite_png:
                    return "[ERROR] Could not capture stereo pair from camera."

                res = self.m.size_link.measure(composite_png, hint_class=hint)
                if left_arr is not None:
                    ann_img = draw_measurement_overlay(left_arr, res)
                    self.m.set_processed_frame(ann_img, f"measure_result_{ts}.jpg")

                if not res.ok:
                    return f"[ERROR] Size estimation failed: {res.error_message}"

                p = res.primary
                if p:
                    h_str = f"{p.height_cm:.1f} cm" if p.height_cm is not None else "n/a"
                    w_str = f"{p.width_cm:.1f} cm" if p.width_cm is not None else "n/a"
                    z_str = f"{p.distance_m:.2f} m" if p.distance_m is not None else "n/a"
                    return f"Measure Result: {p.class_name} | Height: {h_str} | Width: {w_str} | Dist: {z_str}"
                return "Measure completed, but no object detected."

            elif cmd == "CLASSIFY":
                left_bytes, left_arr = self._grab_left_for_command()
                if not left_bytes or left_arr is None:
                    return "[ERROR] Could not grab a fresh camera frame."

                ts = time.strftime("%H%M%S")
                self.m.set_processed_frame(left_arr, f"classify_capture_{ts}.jpg")

                cat_res = llm_tasks.classify_object(left_bytes, "CURRENT_VIEW")
                overlaid = draw_classification_overlay(
                    left_arr,
                    title="Manual Capture",
                    category=cat_res.category,
                    color=cat_res.color,
                )
                self.m.set_processed_frame(overlaid, f"classified_{ts}.jpg")
                return f"Classification: {cat_res.category} (Color: {cat_res.color or 'n/a'})"

            elif cmd in ("POINTS", "LIST"):
                kps = self.m.keypoints.get_all()
                if not kps:
                    return "No keypoints loaded."
                lines = [f"{'ID':<4} {'Name':<6} {'X (m)':<8} {'Y (m)':<8} {'Yaw':<8} {'Vis':<6}"]
                lines.append("-" * 45)
                for kp in kps:
                    lines.append(f"{kp.id:<4} {kp.name:<6} {kp.x:<8.2f} {kp.y:<8.2f} {kp.yaw_degrees:<8.1f} {kp.visibility:<6.2f}")
                return "\n".join(lines)

            elif cmd == "STATUS":
                st = self.m.robot.latest_status
                c_status = "Connected" if self.m.robot.is_connected else "Disconnected"
                health = self.m.camera.get_camera_health()
                nav_msg_suffix = f", Detail: {st.nav_message}" if st.nav_message else ""
                lines = [
                    f"Robot Status:     {c_status} (Pose: X={st.x:.2f}m, Y={st.y:.2f}m, Yaw={st.yaw_deg:.1f}°, Home={'Yes' if st.has_home_pose else 'Auto'})",
                    f"Nav2 State:       {st.nav_state} (Goal Seq: {st.goal_seq}, Target: {st.goal_name or '-'}{nav_msg_suffix})",
                    f"Camera Health:    Left={health['left_state']} ({health['left_fps']:.1f} fps) | Right={health['right_state']} ({health['right_fps']:.1f} fps) | Reopens={health['reopen_count']}",
                    f"Mission Manager:  {self.m.status_text} (Level: {self.m.current_level})",
                    f"Keypoints Loaded: {len(self.m.keypoints.get_all())} entries",
                ]
                return "\n".join(lines)

            elif cmd == "HELP":
                return HELP_BANNER

            elif cmd in ("QUIT", "EXIT"):
                self.m.stop()
                return "Exiting..."

            else:
                return f"Unknown command '{cmd}'. Type 'Help' for list of commands."

        except Exception as err:
            return f"Command execution error: {err}"
