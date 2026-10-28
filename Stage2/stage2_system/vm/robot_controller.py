#!/usr/bin/env python3
"""Stage 2 Robot Controller for Ubuntu VM / ROS 2 Humble.

Receives navigation and teleop commands from Windows Mission Control over HTTP,
executes Nav2 NavigateToPose actions and Twist velocity commands, tracks map
localization via ROS 2 TF, and pushes periodic status pings to the laptop.

Features:
- Nav2 action client for autonomous waypoint navigation
- Short-burst `/cmd_vel` teleop publisher
- Real-time map pose tracking via ROS 2 TF (/tf map -> base_link)
- HTTP REST API for commands (/api/goto, /api/teleop, /api/cancel)
- Periodic status pusher (2 Hz) to registered laptop IP/port
- UDP discovery beacon on port 8079
- Kinematic simulation mode (--sim) for testing without ROS hardware
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import queue
import socket
import socketserver
import sys
import threading
import time
from typing import Any, Optional

# Check ROS 2 availability
try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped, Twist
    from nav2_msgs.action import NavigateToPose
    from tf2_ros import Buffer, TransformListener
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Import shared modules
try:
    from keypoints import Keypoint, KeypointStore, MapInfo
    from protocol import (
        DEFAULT_LAPTOP_PING_PORT,
        DEFAULT_VM_BEACON_PORT,
        DEFAULT_VM_CMD_PORT,
        DISCOVERY_BEACON_REQUEST,
        DISCOVERY_BEACON_RESPONSE_PREFIX,
        GotoRequest,
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
        GotoRequest,
        RegisterRequest,
        RobotStatus,
        TeleopRequest,
    )


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
            # Caller wants status pings sent back
            caller_ip = self.client_address[0]
            ping_port = int(payload.get("ping_port", DEFAULT_LAPTOP_PING_PORT))
            self.controller.register_laptop(caller_ip, ping_port)
            self._send_json({"status": "registered", "ip": caller_ip, "port": ping_port})

        elif self.path == "/api/goto":
            req = GotoRequest.from_dict(payload)
            ok, seq, msg = self.controller.request_goto(req.keypoint_id)
            self._send_json({"accepted": ok, "goal_seq": seq, "message": msg})

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
        self.send_header("Content-Length", str(len(body)))
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
    ) -> None:
        self.cmd_port = cmd_port
        self.beacon_port = beacon_port
        self.sim_mode = sim_mode or (not ROS_AVAILABLE)
        self.sim_speed = 3.0

        # In-place keypoint and map store
        self.keypoints = KeypointStore(file_path=points_path)
        self.map_info = MapInfo(yaml_path=map_path) if map_path else None

        # Registered laptop for pushed pings
        self.registered_laptop_ip: Optional[str] = None
        self.registered_laptop_port: int = DEFAULT_LAPTOP_PING_PORT

        # State tracking
        self.goal_counter = 0
        self.active_goal_seq = 0
        self.active_keypoint_id: Optional[int] = None
        self.active_goal_name: Optional[str] = None
        self.nav_state = "idle"  # idle | navigating | succeeded | aborted | canceled
        self.teleop_active = False

        # Simulation kinematic state
        self.sim_x = 0.0
        self.sim_y = 0.0
        self.sim_yaw = 0.0
        self.sim_target_xy: Optional[tuple[float, float, float]] = None

        # ROS 2 Node components
        self.node = None
        self.tf_buffer = None
        self.tf_listener = None
        self.nav_client = None
        self.goal_handle = None
        self.cmd_vel_pub = None

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

    def _init_ros2(self) -> None:
        rclpy.init()
        self.node = Node("stage2_robot_controller")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.nav_client = ActionClient(self.node, NavigateToPose, "/navigate_to_pose")
        self.cmd_vel_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        print("[RobotController] ROS 2 Humble node and Nav2 ActionClient initialized.")

    def start(self) -> None:
        self._start_http_server()
        self._start_udp_beacon()
        self._start_status_pusher()
        self._start_spin_thread()

    def _start_spin_thread(self) -> None:
        def spin_loop():
            while self._running:
                self.spin_tick()
                time.sleep(0.04)

        self._spin_thread = threading.Thread(target=spin_loop, daemon=True)
        self._spin_thread.start()

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
        def beacon_loop():
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

    def _start_status_pusher(self) -> None:
        def push_loop():
            while self._running:
                time.sleep(0.5)  # 2 Hz
                if not self.registered_laptop_ip:
                    continue

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

        self._ping_thread = threading.Thread(target=push_loop, daemon=True)
        self._ping_thread.start()

    def register_laptop(self, ip: str, port: int) -> None:
        self.registered_laptop_ip = ip
        self.registered_laptop_port = port
        print(f"[RobotController] Registered Windows Mission Control at {ip}:{port} for status pings.")

    def get_robot_pose(self) -> tuple[bool, float, float, float]:
        """Query TF or simulation pose: (localized, x, y, yaw_deg)."""
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
                norm = math.hypot(rot.x, rot.y, rot.z, rot.w)
                if norm < 1e-9:
                    continue
                rx, ry, rz, rw = rot.x / norm, rot.y / norm, rot.z / norm, rot.w / norm
                yaw = math.degrees(math.atan2(2 * (rw * rz + rx * ry), 1 - 2 * (ry * ry + rz * rz)))
                return True, trans.x, trans.y, yaw
            except Exception:
                continue

        return False, 0.0, 0.0, 0.0

    def get_status(self) -> RobotStatus:
        loc, x, y, yaw = self.get_robot_pose()
        nav2_ready = True if self.sim_mode else (self.nav_client.server_is_ready() if self.nav_client else False)
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
            keypoints_sha=self.keypoints.sha256,
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
    # Navigation & Teleop Handlers
    # -------------------------------------------------------------------------

    def request_goto(self, keypoint_id: int) -> tuple[bool, int, str]:
        kp = self.keypoints.get_by_id(keypoint_id)
        if not kp:
            return False, 0, f"Keypoint ID {keypoint_id} not found"

        self.goal_counter += 1
        seq = self.goal_counter
        self.active_goal_seq = seq
        self.active_keypoint_id = keypoint_id
        self.active_goal_name = kp.name
        self.nav_state = "navigating"

        print(f"[RobotController] Goal #{seq}: Navigating to {kp.name} ({kp.x:.2f}, {kp.y:.2f})...")

        if self.sim_mode:
            # Kinematic simulation
            self.sim_target_xy = (kp.x, kp.y, kp.yaw_degrees)
            return True, seq, "Goal dispatched (simulated)"

        if not self.nav_client.server_is_ready():
            self.nav_state = "rejected"
            return False, seq, "Nav2 server not ready"

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(kp.x)
        goal_msg.pose.pose.position.y = float(kp.y)

        quat = kp.quaternion
        goal_msg.pose.pose.orientation.x = float(quat.get("x", 0.0))
        goal_msg.pose.pose.orientation.y = float(quat.get("y", 0.0))
        goal_msg.pose.pose.orientation.z = float(quat.get("z", 0.0))
        goal_msg.pose.pose.orientation.w = float(quat.get("w", 1.0))

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(lambda f: self._on_goal_response(f, seq))
        return True, seq, "Goal accepted by Nav2"

    def _on_goal_response(self, future, seq: int) -> None:
        try:
            self.goal_handle = future.result()
            if not self.goal_handle.accepted:
                if self.active_goal_seq == seq:
                    self.nav_state = "rejected"
                return

            res_future = self.goal_handle.get_result_async()
            res_future.add_done_callback(lambda f: self._on_goal_completed(f, seq))
        except Exception:
            if self.active_goal_seq == seq:
                self.nav_state = "aborted"

    def _on_goal_completed(self, future, seq: int) -> None:
        if self.active_goal_seq != seq:
            return
        try:
            status = future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.nav_state = "succeeded"
                print(f"[RobotController] Goal #{seq} SUCCEEDED!")
            elif status == GoalStatus.STATUS_CANCELED:
                self.nav_state = "canceled"
                print(f"[RobotController] Goal #{seq} CANCELED.")
            else:
                self.nav_state = "aborted"
                print(f"[RobotController] Goal #{seq} ABORTED.")
        except Exception:
            self.nav_state = "aborted"

    def request_cancel(self) -> None:
        print("[RobotController] Canceling active navigation...")
        if self.sim_mode:
            self.sim_target_xy = None
            self.nav_state = "canceled"
            return

        if self.goal_handle and self.nav_state == "navigating":
            self.goal_handle.cancel_goal_async()
        self.nav_state = "canceled"

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

    def spin_tick(self) -> None:
        """Call periodically (10-20 Hz) in main loop."""
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

        # 3. Spin ROS 2 events
        if not self.sim_mode and self.node:
            rclpy.spin_once(self.node, timeout_sec=0.02)

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
                print(f"[RobotController Sim] Reached destination for Goal #{self.active_goal_seq}!")

    def stop(self) -> None:
        self._running = False
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        if self.node and ROS_AVAILABLE:
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Robot Controller (Ubuntu VM)")
    script_dir = Path(__file__).resolve().parent
    stage2_maps = script_dir.parent / "vm_map_tools" / "maps"
    default_pts = stage2_maps / "stage2_keypoints.json"
    default_map = stage2_maps / "final_room.yaml"

    parser.add_argument("--points", type=Path, default=default_pts, help=f"Path to keypoints JSON (default: {default_pts})")
    parser.add_argument("--map", type=Path, default=default_map, help=f"Path to map YAML (default: {default_map})")
    parser.add_argument("--port", type=int, default=DEFAULT_VM_CMD_PORT, help=f"Command port (default: {DEFAULT_VM_CMD_PORT})")
    parser.add_argument("--sim", action="store_true", help="Run in kinematic simulation mode without ROS 2")
    args = parser.parse_args()

    print("=" * 70)
    print("  STAGE 2 ROBOT CONTROLLER (UBUNTU VM)")
    print(f"  Mode: {'SIMULATION' if args.sim or not ROS_AVAILABLE else 'ROS 2 HUMBLE'}")
    print(f"  Keypoints: {args.points.name} | Command Port: {args.port}")
    print("=" * 70)

    controller = RobotController(
        points_path=args.points,
        map_path=args.map,
        cmd_port=args.port,
        sim_mode=args.sim,
    )
    controller.start()

    print("[Ready] Robot Controller active. Waiting for commands from Windows Mission Control...")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[RobotController] Interrupted by user.")
    finally:
        controller.stop()
        print("[RobotController] Shutdown complete.")


if __name__ == "__main__":
    main()
