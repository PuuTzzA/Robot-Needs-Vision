"""Command parser and execution engine for Windows Mission Control."""

from __future__ import annotations

import base64
import shlex
import time
from typing import TYPE_CHECKING, Tuple
import cv2
import numpy as np

if TYPE_CHECKING:
    from .mission import MissionManager

from shared.protocol import VISIBILITY_DEFAULT
from .llm import llm_api
from .llm import tasks as llm_tasks


HELP_BANNER = """
================================================================================
AVAILABLE MISSION CONTROL COMMANDS:
--------------------------------------------------------------------------------
Level <1|2|3>          - Start autonomous execution of specified level
Restart <1|2|3>        - Abort, clear results, and restart specified level
Stop                   - Abort active navigation and cancel current level
Goto <Point>           - Send robot to closest keypoint entry (e.g. 'Goto B' or 'Goto 0')
Goto #<id>             - Send robot to specific unique keypoint ID (e.g. 'Goto #2')
Ping <VM|SSH|LLM>      - Test round-trip latency to specified subsystem
W / A / S / D          - Teleop robot: forward, left, backward, right burst (0.6s)
Space                  - Stop robot teleop motion immediately
Photo                  - Capture camera frame and display in 'Last Processed Frame'
Measure [hint]         - Request stereo height estimate from SSH server (e.g. 'Measure bottle')
Classify               - Send current camera frame to LLM to identify object category
Points / List          - Display all loaded keypoints and their visibility values
Status                 - Print current subsystem status and robot pose
Help                   - Show this command reference
================================================================================
"""


class CommandDispatcher:
    """Parses and executes user commands typed into GUI command input or terminal."""

    def __init__(self, mission: MissionManager) -> None:
        self.m = mission

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

            elif cmd == "GOTO":
                if not args:
                    return "[ERROR] Usage: Goto <PointName> (e.g. 'Goto B') or Goto #<id> (e.g. 'Goto #2')"
                target = args[0]
                if target.startswith("#"):
                    # ID specified
                    try:
                        kp_id = int(target[1:])
                        kp = self.m.keypoints.get_by_id(kp_id)
                        if not kp:
                            return f"[ERROR] Keypoint ID {kp_id} not found."
                        ok, seq, msg = self.m.robot.goto_keypoint(kp.id)
                        return f"Navigating to Keypoint ID {kp.id} ('{kp.name}')..." if ok else f"Goto failed: {msg}"
                    except ValueError:
                        return f"[ERROR] Invalid ID '{target}'"
                else:
                    pt_name = target.upper()
                    cur_pose = (self.m.robot.latest_status.x, self.m.robot.latest_status.y)
                    kp = self.m.keypoints.select_entry(pt_name, robot_xy=cur_pose, min_visibility=VISIBILITY_DEFAULT)
                    if not kp:
                        return f"[ERROR] Keypoint '{pt_name}' not found in keypoint store."
                    ok, seq, msg = self.m.robot.goto_keypoint(kp.id)
                    return f"Navigating to Point {pt_name} (ID {kp.id}, Vis {kp.visibility:.2f})..." if ok else f"Goto failed: {msg}"

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

            elif cmd == "PHOTO":
                left_bytes = self.m.camera.snapshot_left_jpeg()
                left_arr, _, _ = self.m.camera.get_latest_pair()
                if left_arr is not None:
                    ts = time.strftime("%H%M%S")
                    self.m.set_processed_frame(left_arr, f"snapshot_{ts}.jpg")
                    return f"Captured photo (snapshot_{ts}.jpg)"
                return "Failed capturing photo (no active camera frame)"

            elif cmd == "MEASURE":
                hint = args[0] if args else None
                composite_png = self.m.camera.snapshot_stereo_composite_png()
                if not composite_png:
                    return "[ERROR] Could not capture stereo pair from camera."

                res = self.m.size_link.measure(composite_png, hint_class=hint)
                if not res.ok:
                    return f"[ERROR] Size estimation failed: {res.error_message}"

                if res.annotated_jpeg_b64:
                    jpg_bytes = base64.b64decode(res.annotated_jpeg_b64)
                    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    ann_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if ann_img is not None:
                        self.m.set_processed_frame(ann_img, "measure_result.jpg")

                p = res.primary
                if p:
                    h_str = f"{p.height_cm:.1f} cm" if p.height_cm is not None else "n/a"
                    w_str = f"{p.width_cm:.1f} cm" if p.width_cm is not None else "n/a"
                    z_str = f"{p.distance_m:.2f} m" if p.distance_m is not None else "n/a"
                    return f"Measure Result: {p.class_name} | Height: {h_str} | Width: {w_str} | Dist: {z_str}"
                return "Measure completed, but no object detected."

            elif cmd == "CLASSIFY":
                left_bytes = self.m.camera.snapshot_left_jpeg()
                if not left_bytes:
                    return "[ERROR] Could not grab camera frame."
                cat_res = llm_tasks.classify_object(left_bytes, "CURRENT_VIEW")
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
                lines = [
                    f"Robot Status:     {c_status} (Pose: X={st.x:.2f}m, Y={st.y:.2f}m, Yaw={st.yaw_deg:.1f}°)",
                    f"Nav2 State:       {st.nav_state} (Goal Seq: {st.goal_seq}, Keypoint: {st.goal_name or '-'})",
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

