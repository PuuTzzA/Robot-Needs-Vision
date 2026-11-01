#!/usr/bin/env python3
"""Stage 2 Robot Controller for Ubuntu VM / ROS 2 Humble.

Receives navigation and teleop commands from Windows Mission Control over HTTP,
executes Nav2 NavigateToPose actions and Twist velocity commands, tracks map
localization via ROS 2 TF + /amcl_pose, walks through RViz 2D Pose Estimate
confirmation (with Nav2 costmap clearing), and pushes periodic status pings to
the laptop.

Features:
- Nav2 action client for autonomous waypoint navigation
- Interactive RViz 2D Pose Estimate confirmation + runtime 'p' re-pose shortcut
- Automatic Nav2 local & global costmap clearing upon pose confirmation
- Short-burst `/cmd_vel` teleop publisher
- Real-time map pose tracking via ROS 2 TF (/tf map -> base_link) and /amcl_pose
- HTTP REST API for commands (/api/goto, /api/teleop, /api/cancel)
- Periodic status pusher (2 Hz) to registered laptop IP/port
- UDP discovery beacon on port 8079
- Kinematic simulation mode (--sim) for testing without ROS hardware
"""

from __future__ import annotations

import argparse
import base64
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import socket
import socketserver
import sys
import threading
import time
from typing import Any, Callable, Optional

# Check ROS 2 availability
try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
    from nav2_msgs.action import NavigateToPose
    from tf2_ros import Buffer, TransformListener

    try:
        from nav2_msgs.srv import ClearEntireCostmap
    except ImportError:
        ClearEntireCostmap = None  # type: ignore

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    ClearEntireCostmap = None  # type: ignore

# Import shared modules
try:
    from keypoints import Keypoint, KeypointStore, MapInfo
    from protocol import (
        DEFAULT_LAPTOP_PING_PORT,
        DEFAULT_VM_BEACON_PORT,
        DEFAULT_VM_CMD_PORT,
        DISCOVERY_BEACON_REQUEST,
        DISCOVERY_BEACON_RESPONSE_PREFIX,
        GotoPoseRequest,
        GotoRequest,
        HOME_GOAL_NAME,
        RegisterRequest,
        RobotStatus,
        TeleopRequest,
    )
except ImportError:
    from .keypoints import Keypoint, KeypointStore, MapInfo
    from .protocol import (
        DEFAULT_LAPTOP_PING_PORT,
        DEFAULT_VM_BEACON_PORT,
        DEFAULT_VM_CMD_PORT,
        DISCOVERY_BEACON_REQUEST,
        DISCOVERY_BEACON_RESPONSE_PREFIX,
        GotoPoseRequest,
        GotoRequest,
        HOME_GOAL_NAME,
        RegisterRequest,
        RobotStatus,
        TeleopRequest,
    )


def _quat_to_yaw_deg(qx: float, qy: float, qz: float, qw: float) -> Optional[float]:
    norm = math.hypot(qx, qy, qz, qw)
    if norm < 1e-9:
        return None
    rx, ry, rz, rw = qx / norm, qy / norm, qz / norm, qw / norm
    return math.degrees(math.atan2(2.0 * (rw * rz + rx * ry), 1.0 - 2.0 * (ry * ry + rz * rz)))


def _yaw_deg_to_quat(yaw_deg: float) -> dict[str, float]:
    rad = math.radians(yaw_deg) / 2.0
    return {"x": 0.0, "y": 0.0, "z": math.sin(rad), "w": math.cos(rad)}


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ControllerHttpHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests from Windows Mission Control."""

    server: ThreadedHTTPServer

    @property
    def controller(self) -> RobotController:
        return self.server.controller  # type: ignore

    def do_GET(self) -> None:
        if self.path == "/api/hello":
            data = {
                "status": "ok",
                "name": "TurtleBot4 Stage 2 Robot Controller",
                "version": "2.0",
                "ros_available": ROS_AVAILABLE,
                "sim_mode": self.controller.sim_mode,
                "pose_confirmed": self.controller.pose_confirmed,
                "keypoints_sha": self.controller.keypoints.sha256,
            }
            self._send_json(data)

        elif self.path == "/api/status":
            st = self.controller.get_status()
            self._send_json(st.to_dict())

        elif self.path == "/api/keypoints":
            kps = [kp.to_dict() for kp in self.controller.keypoints.get_all()]
            data = {"keypoints": kps, "sha256": self.controller.keypoints.sha256}
            self._send_json(data)

        elif self.path == "/api/map":
            data = self.controller.get_map_payload()
            self._send_json(data)

        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        except Exception:
            payload = {}

        if self.path == "/api/register":
            caller_ip = self.client_address[0]
            ping_port = int(payload.get("ping_port", DEFAULT_LAPTOP_PING_PORT))
            self.controller.register_laptop(caller_ip, ping_port)
            self._send_json({"status": "registered", "ip": caller_ip, "port": ping_port})

        elif self.path == "/api/goto":
            req = GotoRequest.from_dict(payload)
            ok, seq, msg = self.controller.request_goto(req.keypoint_id)
            self._send_json({"accepted": ok, "goal_seq": seq, "message": msg})

        elif self.path == "/api/goto_pose":
            req = GotoPoseRequest.from_dict(payload)
            ok, seq, msg = self.controller.request_goto_pose(
                req.x, req.y, req.yaw_deg, label=req.label
            )
            self._send_json({"accepted": ok, "goal_seq": seq, "message": msg})

        elif self.path == "/api/goto_home":
            ok, seq, msg = self.controller.request_goto_home()
            self._send_json({"accepted": ok, "goal_seq": seq, "message": msg})

        elif self.path == "/api/set_home":
            ok, msg = self.controller.latch_current_as_home()
            self._send_json({"ok": ok, "message": msg})

        elif self.path == "/api/teleop":
            req = TeleopRequest.from_dict(payload)
            ok = self.controller.request_teleop(req)
            self._send_json({"ok": ok})

        elif self.path == "/api/cancel":
            self.controller.request_cancel()
            self._send_json({"ok": True})

        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(body_len := len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class RobotController:
    """Core controller coordinating ROS 2 Nav2, TF, HTTP API, and status pings."""

    def __init__(
        self,
        points_path: Path,
        map_path: Optional[Path] = None,
        cmd_port: int = DEFAULT_VM_CMD_PORT,
        beacon_port: int = DEFAULT_VM_BEACON_PORT,
        sim_mode: bool = False,
        pose_confirmed: bool = True,
        goal_timeout_s: float = 60.0,
        stuck_timeout_s: float = 10.0,
        stuck_min_move_m: float = 0.05,
    ) -> None:
        self.cmd_port = cmd_port
        self.beacon_port = beacon_port
        self.sim_mode = sim_mode or (not ROS_AVAILABLE)
        self.sim_speed = 3.0
        self.pose_confirmed = bool(pose_confirmed)
        self.goal_timeout_s = float(goal_timeout_s)
        self.stuck_timeout_s = float(stuck_timeout_s)
        self.stuck_min_move_m = float(stuck_min_move_m)

        # In-place keypoint and map store
        self.keypoints = KeypointStore(file_path=points_path)
        self.map_info = MapInfo(yaml_path=map_path) if map_path else None

        # Registered laptop for pushed pings
        self.registered_laptop_ip: Optional[str] = None
        self.registered_laptop_port: int = DEFAULT_LAPTOP_PING_PORT

        # State tracking
        self._state_lock = threading.Lock()
        self.goal_counter = 0
        self.active_goal_seq = 0
        self.active_keypoint_id: Optional[int] = None
        self.active_goal_name: Optional[str] = None
        self.nav_state = "idle"  # idle | navigating | succeeded | failed | aborted | canceled
        self.nav_message = ""
        self.teleop_active = False

        # Goal timeout & stuck detection tracking
        self._goal_start_mono: float = 0.0
        self._last_move_mono: float = 0.0
        self._last_move_pose: Optional[tuple[float, float, float]] = None
        self._active_target_xy: Optional[tuple[float, float]] = None

        # Simulation kinematic state
        self.sim_x = 0.0
        self.sim_y = 0.0
        self.sim_yaw = 0.0
        self.sim_target_xy: Optional[tuple[float, float, float]] = None

        # Latched initial/home pose (x, y, yaw_deg) from RViz localization or sim start
        self.initial_pose: Optional[tuple[float, float, float]] = (
            (self.sim_x, self.sim_y, self.sim_yaw) if self.sim_mode else None
        )

        # Latest /amcl_pose cache (used as immediate fallback alongside TF lookup)
        self._amcl_pose_lock = threading.Lock()
        self._latest_amcl_pose: Optional[tuple[float, float, float]] = None

        # ROS 2 Node components
        self.node = None
        self.executor = None
        self.tf_buffer = None
        self.tf_listener = None
        self.amcl_sub = None
        self.nav_client = None
        self.goal_handle = None
        self.cmd_vel_pub = None
        self._clear_local_costmap_cli = None
        self._clear_global_costmap_cli = None

        if not self.sim_mode:
            self._init_ros2()

        # Teleop timer state
        self._teleop_end_time = 0.0
        self._teleop_twist = (0.0, 0.0)

        # Discovery & HTTP Server threads
        self._running = True
        self._server = None
        self._server_thread = None
        self._beacon_sock = None
        self._beacon_thread = None
        self._ping_thread = None
        self._exec_thread = None
        self._tick_thread = None
        self._spin_thread = None

    def _init_ros2(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("stage2_robot_controller")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.nav_client = ActionClient(self.node, NavigateToPose, "/navigate_to_pose")
        self.cmd_vel_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        amcl_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.amcl_sub = self.node.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._on_amcl_pose,
            amcl_qos,
        )

        if ClearEntireCostmap is not None:
            self._clear_local_costmap_cli = self.node.create_client(
                ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap"
            )
            self._clear_global_costmap_cli = self.node.create_client(
                ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap"
            )

        print("[RobotController] ROS 2 Humble node and Nav2 ActionClient initialized.")

    def _on_amcl_pose(self, msg: Any) -> None:
        try:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            yaw = _quat_to_yaw_deg(ori.x, ori.y, ori.z, ori.w)
            if yaw is not None:
                with self._amcl_pose_lock:
                    self._latest_amcl_pose = (float(pos.x), float(pos.y), yaw)
        except Exception:
            pass

    def start(self) -> None:
        self._start_http_server()
        self._start_udp_beacon()
        self._start_status_pusher()
        self._start_ros_executor()
        self._start_tick_thread()

    def _start_ros_executor(self) -> None:
        """Run dedicated ROS 2 executor to process callbacks instantly without throttled sleeps."""
        if self.sim_mode or not self.executor:
            return

        def exec_loop():
            try:
                self.executor.spin()
            except Exception:
                pass

        self._exec_thread = threading.Thread(target=exec_loop, daemon=True)
        self._exec_thread.start()

    def _start_tick_thread(self) -> None:
        """Periodic logic loop (~25 Hz) for teleop timeout watchdog and simulation integration."""
        def tick_loop():
            while self._running:
                self.tick_step()
                time.sleep(0.04)

        self._tick_thread = threading.Thread(target=tick_loop, daemon=True)
        self._tick_thread.start()

    def _start_spin_thread(self) -> None:
        """Backward-compatible alias for starting tick loop."""
        self._start_tick_thread()

    def _start_http_server(self) -> None:
        for port in [self.cmd_port, 8075, 8076]:
            try:
                server = ThreadedHTTPServer(("0.0.0.0", port), ControllerHttpHandler)
                server.controller = self  # type: ignore
                self._server = server
                self.cmd_port = port
                self._server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                self._server_thread.start()
                print(f"[RobotController] HTTP API listening on port {port}...")
                return
            except OSError:
                continue
        print(f"[RobotController] [ERROR] Could not bind command port {self.cmd_port}!")

    def _start_udp_beacon(self) -> None:
        def beacon_loop() -> None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", self.beacon_port))
                self._beacon_sock = sock
            except Exception as e:
                print(f"[RobotController] [WARN] Beacon bind failed on port {self.beacon_port}: {e}")
                return

            while self._running:
                try:
                    data, addr = sock.recvfrom(512)
                    if DISCOVERY_BEACON_REQUEST in data:
                        resp = f"{DISCOVERY_BEACON_RESPONSE_PREFIX}{self.cmd_port}".encode("utf-8")
                        sock.sendto(resp, addr)
                except Exception:
                    time.sleep(0.02)

        self._beacon_thread = threading.Thread(target=beacon_loop, daemon=True)
        self._beacon_thread.start()

    def _push_status_once(self) -> None:
        if not self.registered_laptop_ip:
            return
        status = self.get_status()
        payload = json.dumps(status.to_dict()).encode("utf-8")
        conn = None
        try:
            conn = http.client.HTTPConnection(
                self.registered_laptop_ip, self.registered_laptop_port, timeout=0.8
            )
            headers = {"Content-Type": "application/json"}
            conn.request("POST", "/api/robot_status", body=payload, headers=headers)
            res = conn.getresponse()
            res.read()
        except Exception:
            pass
        finally:
            if conn:
                conn.close()

    def _push_status_async(self) -> None:
        if not self.registered_laptop_ip:
            return
        threading.Thread(target=self._push_status_once, daemon=True).start()

    def _start_status_pusher(self) -> None:
        def push_loop() -> None:
            while self._running:
                time.sleep(0.5)  # 2 Hz
                self._push_status_once()

        self._ping_thread = threading.Thread(target=push_loop, daemon=True)
        self._ping_thread.start()

    def register_laptop(self, ip: str, port: int) -> None:
        self.registered_laptop_ip = ip
        self.registered_laptop_port = port
        print(f"[RobotController] Registered Windows Mission Control at {ip}:{port} for status pings.")

    def get_robot_pose(self) -> tuple[bool, float, float, float]:
        """Query TF (or /amcl_pose fallback) or simulation pose: (localized, x, y, yaw_deg)."""
        if self.sim_mode or self.node is None or self.tf_buffer is None:
            return True, self.sim_x, self.sim_y, self.sim_yaw

        for frame in ("base_link", "base_footprint"):
            try:
                t = self.tf_buffer.lookup_transform(
                    "map",
                    frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.08),
                )
                trans = t.transform.translation
                rot = t.transform.rotation
                yaw = _quat_to_yaw_deg(rot.x, rot.y, rot.z, rot.w)
                if yaw is None:
                    continue
                return True, float(trans.x), float(trans.y), yaw
            except Exception:
                continue

        with self._amcl_pose_lock:
            if self._latest_amcl_pose is not None:
                ax, ay, ayaw = self._latest_amcl_pose
                return True, ax, ay, ayaw

        return False, 0.0, 0.0, 0.0

    def get_status(self) -> RobotStatus:
        loc, x, y, yaw = self.get_robot_pose()
        nav2_ready = (
            True
            if self.sim_mode
            else (bool(self.nav_client and self.nav_client.server_is_ready()) and self.pose_confirmed)
        )
        return RobotStatus(
            t=time.time(),
            localized=loc,
            x=round(x, 3),
            y=round(y, 3),
            yaw_deg=round(yaw, 1),
            nav_state=self.nav_state,
            goal_seq=self.active_goal_seq,
            goal_keypoint_id=self.active_keypoint_id,
            goal_name=self.active_goal_name,
            nav2_ready=nav2_ready,
            teleop_active=self.teleop_active,
            has_home_pose=(self.initial_pose is not None),
            keypoints_sha=self.keypoints.sha256,
            nav_message=self.nav_message,
        )

    def get_map_payload(self) -> dict:
        if not self.map_info:
            return {"error": "No map loaded"}
        meta = {
            "resolution": self.map_info.resolution,
            "origin": [self.map_info.origin_x, self.map_info.origin_y, self.map_info.origin_yaw],
            "width_px": self.map_info.width_px,
            "height_px": self.map_info.height_px,
        }
        b64 = ""
        if self.map_info.image_bytes:
            b64 = base64.b64encode(self.map_info.image_bytes).decode("ascii")
        return {"meta": meta, "pgm_b64": b64}

    # -------------------------------------------------------------------------
    # Nav2 Readiness, Initial Pose Confirmation & Costmap Reset
    # -------------------------------------------------------------------------

    def wait_for_nav2_ready(self, timeout_s: float = 300.0) -> bool:
        """Wait until the Nav2 /navigate_to_pose action server is reachable."""
        if self.sim_mode:
            return True
        if self.nav_client is None:
            return False

        start_t = time.monotonic()
        warned = False
        while self._running:
            if self.nav_client.server_is_ready():
                elapsed = time.monotonic() - start_t
                print(
                    f"\r  [OK] Nav2 /navigate_to_pose action server is ONLINE ({elapsed:.1f}s).             "
                )
                return True

            elapsed = time.monotonic() - start_t
            if elapsed >= timeout_s:
                print(f"\n  [ERROR] Timed out waiting {timeout_s:.0f}s for Nav2 /navigate_to_pose server.")
                return False

            if elapsed >= 5.0 and not warned:
                print(
                    "\n  [INFO] Still waiting for Nav2 -- make sure './auto_bringup.sh' finished Steps 1-5 in Terminal 1."
                )
                warned = True

            sys.stdout.write(
                f"\r  [WAIT] [{int(elapsed):3d}s/{int(timeout_s)}s] Waiting for Nav2 /navigate_to_pose action server..."
            )
            sys.stdout.flush()
            time.sleep(0.5)

        return False

    def clear_costmaps(self, timeout_s: float = 1.5) -> None:
        """Clear Nav2 local and global costmaps (removes stale obstacles after moving robot by hand)."""
        if self.sim_mode or ClearEntireCostmap is None:
            return

        futures = []
        for label, cli in (
            ("local_costmap", self._clear_local_costmap_cli),
            ("global_costmap", self._clear_global_costmap_cli),
        ):
            if cli is not None and cli.service_is_ready():
                try:
                    futures.append((label, cli.call_async(ClearEntireCostmap.Request())))
                except Exception:
                    pass

        if not futures:
            return

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if all(f.done() for _, f in futures):
                break
            time.sleep(0.03)

    def confirm_initial_pose_interactive(
        self,
        input_fn: Callable[[str], str] = input,
        settle_delay_s: float = 0.25,
    ) -> bool:
        """Prompt user to set 2D Pose Estimate in RViz, display live pose, and confirm [y/N]."""
        if self.nav_state == "navigating":
            self.request_cancel()
        self.pose_confirmed = False

        print("\n" + "-" * 70)
        print("  [Initial Pose] Set & Confirm the Robot's Pose in RViz")
        print("  1. In RViz, click '2D Pose Estimate' on the top toolbar.")
        print("  2. Click the robot's real position on the map and drag in the direction it faces.")
        print("  3. Check that the laser scan (red dots) lines up with the map walls.")
        print("  4. Come back to this terminal and press Enter to confirm.")
        print("-" * 70)

        while self._running:
            ans = input_fn(
                "\n  -> Press Enter once the initial pose is set in RViz (or type 'q' to abort): "
            ).strip()
            if ans.lower() == "q":
                raise KeyboardInterrupt("Aborted by user at initial pose confirmation.")

            if not self.sim_mode and settle_delay_s > 0:
                time.sleep(settle_delay_s)

            loc, rx, ry, ryaw = self.get_robot_pose()
            if loc:
                print(f"  * Robot pose (map): x={rx:.2f} m, y={ry:.2f} m, yaw={ryaw:.1f} deg")
                if not self.sim_mode and abs(rx) < 0.05 and abs(ry) < 0.05 and abs(ryaw) < 0.5:
                    print(
                        "  [WARN] Pose is still near (0.00, 0.00, 0.0 deg) -- did you click '2D Pose Estimate' in RViz?"
                    )
            else:
                print(
                    "  [WARN] Could not read robot pose yet (set '2D Pose Estimate' in RViz first)."
                )

            confirm = input_fn("  -> Is the robot correctly localized on the map? [y/N]: ").strip().lower()
            if confirm in ("y", "yes"):
                self.clear_costmaps()
                loc_now, lx, ly, lyaw = self.get_robot_pose()
                if loc_now:
                    self.initial_pose = (lx, ly, lyaw)
                    print(
                        f"  [OK] Latched Start/Home pose: x={lx:.2f} m, y={ly:.2f} m, yaw={lyaw:.1f} deg"
                    )
                elif loc:
                    self.initial_pose = (rx, ry, ryaw)
                    print(
                        f"  [OK] Latched Start/Home pose: x={rx:.2f} m, y={ry:.2f} m, yaw={ryaw:.1f} deg"
                    )
                self.pose_confirmed = True
                print("  [OK] Initial pose confirmed (Nav2 costmaps cleared).")
                return True

            print("  Okay -- set the pose again with '2D Pose Estimate' in RViz, then press Enter.")

        return False

    def latch_current_as_home(self) -> tuple[bool, str]:
        """Record the robot's current localized pose as the Start/Home return pose."""
        loc, rx, ry, ryaw = self.get_robot_pose()
        if not loc:
            return False, "Robot pose is not localized yet"
        self.initial_pose = (rx, ry, ryaw)
        msg = f"Latched Start/Home pose at ({rx:.2f}, {ry:.2f}, {ryaw:.1f} deg)"
        print(f"[RobotController] {msg}")
        return True, msg

    def _ensure_initial_pose_latched(self) -> None:
        """Lazily capture initial pose before first goal if --skip-pose-confirm was used."""
        if self.initial_pose is not None:
            return
        loc, rx, ry, ryaw = self.get_robot_pose()
        if loc:
            self.initial_pose = (rx, ry, ryaw)
            print(
                f"[RobotController] Auto-latched Start/Home pose before first goal: ({rx:.2f}, {ry:.2f}, {ryaw:.1f} deg)"
            )

    # -------------------------------------------------------------------------
    # Navigation & Teleop Handlers
    # -------------------------------------------------------------------------

    def request_goto_pose(
        self,
        x: float,
        y: float,
        yaw_deg: float = 0.0,
        label: str = HOME_GOAL_NAME,
        keypoint_id: Optional[int] = None,
        quat: Optional[dict[str, float]] = None,
    ) -> tuple[bool, int, str]:
        if not self.pose_confirmed:
            return (
                False,
                0,
                "Initial pose not confirmed yet -- confirm [y/N] in the VM robot_controller terminal first",
            )

        self._ensure_initial_pose_latched()

        now_mono = time.monotonic()
        loc, rx, ry, ryaw = self.get_robot_pose()
        with self._state_lock:
            self.goal_counter += 1
            seq = self.goal_counter
            self.active_goal_seq = seq
            self.active_keypoint_id = keypoint_id
            self.active_goal_name = label
            self.nav_state = "navigating"
            self.nav_message = ""
            self._goal_start_mono = now_mono
            self._last_move_mono = now_mono
            self._last_move_pose = (rx, ry, ryaw) if loc else None
            self._active_target_xy = (float(x), float(y))

        print(f"[RobotController] Goal #{seq}: Navigating to {label} ({x:.2f}, {y:.2f}, {yaw_deg:.1f} deg)...")

        if self.sim_mode:
            self.sim_target_xy = (float(x), float(y), float(yaw_deg))
            return True, seq, "Goal dispatched (simulated)"

        if not self.nav_client.server_is_ready():
            self._fail_active_goal(seq, "Nav2 server not ready")
            return False, seq, "Nav2 server not ready"

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)

        eff_quat = quat if quat else _yaw_deg_to_quat(yaw_deg)
        goal_msg.pose.pose.orientation.x = float(eff_quat.get("x", 0.0))
        goal_msg.pose.pose.orientation.y = float(eff_quat.get("y", 0.0))
        goal_msg.pose.pose.orientation.z = float(eff_quat.get("z", 0.0))
        goal_msg.pose.pose.orientation.w = float(eff_quat.get("w", 1.0))

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(lambda f: self._on_goal_response(f, seq))
        return True, seq, "Goal accepted by Nav2"

    def request_goto(self, keypoint_id: int) -> tuple[bool, int, str]:
        if not self.pose_confirmed:
            return (
                False,
                0,
                "Initial pose not confirmed yet -- confirm [y/N] in the VM robot_controller terminal first",
            )

        kp = self.keypoints.get_by_id(keypoint_id)
        if not kp:
            return False, 0, f"Keypoint ID {keypoint_id} not found"

        return self.request_goto_pose(
            x=kp.x,
            y=kp.y,
            yaw_deg=kp.yaw_degrees,
            label=kp.name,
            keypoint_id=keypoint_id,
            quat=kp.quaternion,
        )

    def request_goto_home(self) -> tuple[bool, int, str]:
        if not self.pose_confirmed:
            return (
                False,
                0,
                "Initial pose not confirmed yet -- confirm [y/N] in the VM robot_controller terminal first",
            )

        self._ensure_initial_pose_latched()
        if self.initial_pose is None:
            return False, 0, "No Start/Home pose recorded yet"

        hx, hy, hyaw = self.initial_pose
        return self.request_goto_pose(
            x=hx,
            y=hy,
            yaw_deg=hyaw,
            label=HOME_GOAL_NAME,
            keypoint_id=None,
        )

    def _fail_active_goal(self, seq: int, reason: str) -> None:
        """Mark active goal as failed, cancel underlying Nav2 action if running, and push failed status to UI."""
        goal_handle_to_cancel = None
        with self._state_lock:
            if self.active_goal_seq != seq or self.nav_state not in ("navigating", "rejected"):
                return
            self.nav_state = "failed"
            self.nav_message = reason
            goal_label = self.active_goal_name or f"#{seq}"
            if self.sim_mode:
                self.sim_target_xy = None
            else:
                goal_handle_to_cancel = self.goal_handle

        print(f"[RobotController] Goal #{seq} ({goal_label}) FAILED: {reason}")
        if goal_handle_to_cancel is not None:
            try:
                goal_handle_to_cancel.cancel_goal_async()
            except Exception:
                pass
        if not self.sim_mode and self.cmd_vel_pub:
            try:
                self.cmd_vel_pub.publish(Twist())
            except Exception:
                pass
        self._push_status_async()

    def _on_goal_response(self, future, seq: int) -> None:
        try:
            self.goal_handle = future.result()
            if not self.goal_handle.accepted:
                self._fail_active_goal(seq, "Goal rejected by Nav2")
                return

            res_future = self.goal_handle.get_result_async()
            res_future.add_done_callback(lambda f: self._on_goal_completed(f, seq))
        except Exception:
            self._fail_active_goal(seq, "Goal dispatch failed")

    def _on_goal_completed(self, future, seq: int) -> None:
        with self._state_lock:
            if self.active_goal_seq != seq or self.nav_state != "navigating":
                return
        try:
            status = future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                with self._state_lock:
                    if self.active_goal_seq == seq and self.nav_state == "navigating":
                        self.nav_state = "succeeded"
                        self.nav_message = "Reached destination"
                print(f"[RobotController] Goal #{seq} SUCCEEDED!")
                self._push_status_async()
            elif status == GoalStatus.STATUS_CANCELED:
                with self._state_lock:
                    if self.active_goal_seq == seq and self.nav_state == "navigating":
                        self.nav_state = "canceled"
                        self.nav_message = "Navigation canceled"
                print(f"[RobotController] Goal #{seq} CANCELED.")
                self._push_status_async()
            else:
                self._fail_active_goal(seq, "Nav2 failed to reach goal")
        except Exception:
            self._fail_active_goal(seq, "Nav2 goal execution error")

    def request_cancel(self) -> None:
        print("[RobotController] Canceling active navigation...")
        goal_handle_to_cancel = None
        with self._state_lock:
            if self.sim_mode:
                self.sim_target_xy = None
                self.nav_state = "canceled"
                self.nav_message = "Navigation canceled"
            else:
                if self.goal_handle and self.nav_state == "navigating":
                    goal_handle_to_cancel = self.goal_handle
                self.nav_state = "canceled"
                self.nav_message = "Navigation canceled"
        if goal_handle_to_cancel is not None:
            try:
                goal_handle_to_cancel.cancel_goal_async()
            except Exception:
                pass
        self._push_status_async()

    def request_teleop(self, req: TeleopRequest) -> bool:
        lin_x = req.linear_x if req.linear_x is not None else 0.0
        ang_z = req.angular_z if req.angular_z is not None else 0.0

        if req.key == "w":
            lin_x = 0.15
        elif req.key == "s":
            lin_x = -0.15
        elif req.key == "a":
            ang_z = 0.50
        elif req.key == "d":
            ang_z = -0.50
        elif req.key == "space":
            lin_x = 0.0
            ang_z = 0.0

        self._teleop_twist = (lin_x, ang_z)
        self._teleop_end_time = time.time() + req.duration
        self.teleop_active = (lin_x != 0.0 or ang_z != 0.0)

        if not self.sim_mode and self.cmd_vel_pub:
            t = Twist()
            t.linear.x = float(lin_x)
            t.angular.z = float(ang_z)
            self.cmd_vel_pub.publish(t)

        return True

    def _check_navigation_timeouts(self) -> None:
        """Enforce 60s total goal timeout and 10s no-movement stuck timeout while navigating."""
        with self._state_lock:
            if self.nav_state != "navigating":
                return
            seq = self.active_goal_seq
            start_mono = self._goal_start_mono
            last_move_mono = self._last_move_mono
            last_pose = self._last_move_pose
            target_xy = self._active_target_xy

        now_mono = time.monotonic()

        # 1. Total goal timeout (default 60s)
        if self.goal_timeout_s > 0.0 and (now_mono - start_mono) >= self.goal_timeout_s:
            self._fail_active_goal(seq, f"Timed out after {self.goal_timeout_s:.0f}s")
            return

        # 2. Stuck / no-movement timeout (default 10s; reset whenever robot moves)
        loc, rx, ry, ryaw = self.get_robot_pose()
        if not loc:
            return

        if last_pose is None:
            with self._state_lock:
                if self.active_goal_seq == seq and self.nav_state == "navigating":
                    self._last_move_pose = (rx, ry, ryaw)
                    self._last_move_mono = now_mono
            return

        ax, ay, ayaw = last_pose
        moved_m = math.hypot(rx - ax, ry - ay)
        near_goal = False
        if target_xy is not None:
            near_goal = math.hypot(target_xy[0] - rx, target_xy[1] - ry) <= 0.20
        yaw_diff = abs((ryaw - ayaw + 180.0) % 360.0 - 180.0)

        if moved_m >= self.stuck_min_move_m or (near_goal and yaw_diff >= 8.0):
            with self._state_lock:
                if self.active_goal_seq == seq and self.nav_state == "navigating":
                    self._last_move_pose = (rx, ry, ryaw)
                    self._last_move_mono = now_mono
            return

        if self.stuck_timeout_s > 0.0 and (now_mono - last_move_mono) >= self.stuck_timeout_s:
            self._fail_active_goal(seq, f"Stuck (not moving for {self.stuck_timeout_s:.0f}s)")

    def tick_step(self) -> None:
        """Periodic logic tick (teleop watchdog, kinematic simulation, nav timeouts) at ~25 Hz."""
        # 1. Update Teleop Twist
        if self.teleop_active:
            if time.time() > self._teleop_end_time:
                self.teleop_active = False
                self._teleop_twist = (0.0, 0.0)
                if not self.sim_mode and self.cmd_vel_pub:
                    self.cmd_vel_pub.publish(Twist())

        # 2. Update Kinematic Sim if active
        if self.sim_mode:
            self._update_sim_step()

        # 3. Check navigation goal & stuck timeouts
        self._check_navigation_timeouts()

    def spin_tick(self) -> None:
        """Backward-compatible alias for tick_step."""
        self.tick_step()

    def _update_sim_step(self) -> None:
        dt = 0.05
        # Teleop velocity integration
        if self.teleop_active:
            vx, wz = self._teleop_twist
            self.sim_yaw = (self.sim_yaw + math.degrees(wz * dt)) % 360.0
            rad = math.radians(self.sim_yaw)
            self.sim_x += vx * math.cos(rad) * dt
            self.sim_y += vx * math.sin(rad) * dt

        # Autonomous goal navigation simulation
        elif self.nav_state == "navigating" and self.sim_target_xy:
            tx, ty, tyaw = self.sim_target_xy
            dx = tx - self.sim_x
            dy = ty - self.sim_y
            dist = math.hypot(dx, dy)

            if dist > 0.05:
                speed = getattr(self, "sim_speed", 3.0)  # m/s
                move = min(dist, speed * dt)
                target_angle = math.atan2(dy, dx)
                self.sim_x += move * math.cos(target_angle)
                self.sim_y += move * math.sin(target_angle)
                self.sim_yaw = math.degrees(target_angle)
            else:
                self.sim_x = tx
                self.sim_y = ty
                self.sim_yaw = tyaw
                self.sim_target_xy = None
                self.nav_state = "succeeded"
                self.nav_message = "Reached destination"
                print(f"[RobotController Sim] Reached destination for Goal #{self.active_goal_seq}!")
                self._push_status_async()

    def stop(self) -> None:
        self._running = False
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        if self.executor:
            try:
                self.executor.shutdown()
            except Exception:
                pass
        if self._beacon_sock:
            try:
                self._beacon_sock.close()
            except Exception:
                pass
        if self.node and ROS_AVAILABLE:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Robot Controller (Ubuntu VM)")
    script_dir = Path(__file__).resolve().parent
    stage2_maps = script_dir.parent.parent / "vm_map_tools" / "maps"
    default_pts = stage2_maps / "stage2_keypoints.json"
    default_map = stage2_maps / "final_room.yaml"

    parser.add_argument(
        "--points", type=Path, default=default_pts, help=f"Path to keypoints JSON (default: {default_pts})"
    )
    parser.add_argument(
        "--map", type=Path, default=default_map, help=f"Path to map YAML (default: {default_map})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_VM_CMD_PORT, help=f"Command port (default: {DEFAULT_VM_CMD_PORT})"
    )
    parser.add_argument("--sim", action="store_true", help="Run in kinematic simulation mode without ROS 2")
    parser.add_argument(
        "--skip-pose-confirm",
        action="store_true",
        help="Skip interactive RViz 2D Pose Estimate confirmation at startup.",
    )
    args = parser.parse_args()

    if not args.sim and not ROS_AVAILABLE:
        print("=" * 70)
        print("  [ERROR] ROS 2 Humble (rclpy) is not sourced in this terminal!")
        print("  Start the controller via:")
        print(f"    cd {script_dir} && ./start_robot_controller.sh")
        print("  (Or pass --sim for offline kinematic simulation.)")
        print("=" * 70)
        sys.exit(1)

    interactive_tty = bool(sys.stdin is not None and sys.stdin.isatty())
    need_pose_confirm = (not args.sim) and (not args.skip_pose_confirm) and interactive_tty

    print("=" * 70)
    print("  STAGE 2 ROBOT CONTROLLER (UBUNTU VM)")
    print(f"  Mode: {'SIMULATION' if args.sim else 'ROS 2 HUMBLE'}")
    print(f"  Keypoints: {args.points.name} | Command Port: {args.port}")
    print("=" * 70)

    controller = RobotController(
        points_path=args.points,
        map_path=args.map,
        cmd_port=args.port,
        sim_mode=args.sim,
        pose_confirmed=(not need_pose_confirm),
    )
    controller.start()

    try:
        if not args.sim:
            if not controller.wait_for_nav2_ready(timeout_s=300.0):
                sys.exit(1)

        if need_pose_confirm:
            controller.confirm_initial_pose_interactive()

        print(
            f"\n[Ready] Robot Controller active on port {controller.cmd_port}. "
            "Waiting for commands from Windows Mission Control..."
        )
        if interactive_tty:
            print(
                "        Tip: Carry robot back to start? Type 'p' + Enter here to reset & confirm pose\n"
                "             (or press Ctrl+C and re-run ./start_robot_controller.sh)."
            )

        while True:
            if interactive_tty:
                try:
                    cmd = input().strip().lower()
                except EOFError:
                    interactive_tty = False
                    continue
                if cmd in ("p", "pose", "repose"):
                    controller.confirm_initial_pose_interactive()
                    print(
                        f"\n[Ready] Pose updated! Robot Controller active on port {controller.cmd_port}..."
                    )
                elif cmd in ("q", "quit", "exit"):
                    break
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[RobotController] Interrupted by user.")
    finally:
        controller.stop()
        print("[RobotController] Shutdown complete.")


if __name__ == "__main__":
    main()
