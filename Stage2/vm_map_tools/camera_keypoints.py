#!/usr/bin/env python3
"""Stage 2: Camera View Inspection & Robot Keypoint Collector.

Opens camera feed(s) under Ubuntu Linux, tracks the robot's real-time map pose
via ROS 2 TF, enables Nav2 goal testing (GOTO <ID>), and records keypoint entries
with custom visibility values into a JSON file.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np

# Check ROS 2 availability
try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PointStamped, PoseStamped, Twist
    from nav2_msgs.action import NavigateToPose
    from tf2_ros import Buffer, TransformListener
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from camera_streamer import CameraStreamer


DESCRIPTION_BANNER = """
================================================================================
Stage 2: Camera View & Robot Keypoint Collector
================================================================================
This tool allows you to drive the TurtleBot 4, inspect what the camera sees
in real-time, and record keypoint positions, rotations, and visibility scores
(0.0 to 1.0) into a structured JSON file for Stage 2 perception tasks.

Features:
  • Live OpenCV camera preview with robot map pose and status HUD
  • Real-time map localization via ROS 2 TF (/tf lookup map -> base_link)
  • Nav2 goal dispatching to verify recorded poses (GOTO <ID>)
  • Automatic unique ID numbering with support for multiple entries per name
  • Instant JSON persistence to safeguard against losing progress
  • Optional keyboard teleop in camera window (W/A/S/D to drive, Space to stop)

Available Terminal Commands:
  SAVE <NAME> <VISIBILITY>  - Save current robot pose (e.g. 'SAVE A 0.5')
  SAVE                     - Immediately flush/save all keypoints to JSON file
  GOTO <ID>                - Send robot to keypoint with ID <ID> (e.g. 'GOTO 2')
  LIST                     - Display a formatted table of all saved keypoints
  DELETE <ID>              - Delete keypoint entry with unique ID <ID>
  STATUS                   - Show localization, camera FPS, and Nav2 status
  CANCEL                   - Abort active robot navigation
  HELP                     - Show this command reference
  QUIT                     - Cleanly shut down cameras, save file, and exit

OpenCV Window Shortcuts:
  's' - Save snapshot image from camera feed
  'q' - Quit application
================================================================================
"""


def compute_map_hash(path: Path) -> str:
    """Compute sha256 of map YAML and referenced image."""
    try:
        import yaml
        config = yaml.safe_load(path.read_text())
        image = Path(config["image"])
        if not image.is_absolute():
            image = path.parent / image
        return hashlib.sha256(path.read_bytes() + image.read_bytes()).hexdigest()
    except Exception:
        return ""


class KeypointConsoleNode(Node if ROS_AVAILABLE else object):  # type: ignore
    """ROS 2 Node handling TF pose lookup, Nav2 goals, and keypoint recording."""

    def __init__(
        self,
        points_path: Path,
        map_path: Path | None,
        save_snapshots: bool = True,
    ) -> None:
        if not ROS_AVAILABLE:
            raise RuntimeError(
                "ROS 2 (rclpy) is not available. Ensure you have run: source /opt/ros/humble/setup.bash"
            )

        super().__init__("stage2_keypoint_collector")
        self.points_path = points_path.expanduser().resolve()
        self.map_path = map_path.expanduser().resolve() if map_path else None
        self.save_snapshots = save_snapshots
        self.snapshots_dir = self.points_path.parent / "snapshots"

        # Calculate map hash if provided
        self.map_hash = compute_map_hash(self.map_path) if self.map_path and self.map_path.exists() else ""

        # TF Buffer and TransformListener for real-time robot pose in 'map' frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Nav2 ActionClient for GOTO <ID>
        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.goal_handle = None
        self.nav_in_progress = False
        self.current_goal_id: int | None = None
        self.cancel_requested = False

        # Optional teleop publisher
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Subscriptions for RViz interaction tools
        self.rviz_clicked_point: tuple[float, float] | None = None
        self.rviz_clicked_yaw: float | None = None
        self.create_subscription(PointStamped, "/clicked_point", self._receive_point, 10)
        self.create_subscription(PoseStamped, "/waypoint_pose", self._receive_waypoint_pose, 10)

        # In-memory keypoint storage
        self.keypoints: list[dict] = []
        self.next_id = 0
        self._load_keypoints()

    def _load_keypoints(self) -> None:
        if not self.points_path.exists():
            return

        try:
            content = self.points_path.read_text()
            data = json.loads(content)
            entries = []
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                entries = data.get("keypoints", [])
                if not entries and "points" in data:
                    # Backward compatibility with go_points.py dict format
                    legacy = data["points"]
                    auto_idx = 0
                    for name, pt in legacy.items():
                        entries.append({
                            "id": auto_idx,
                            "name": name,
                            "x": float(pt["x"]),
                            "y": float(pt["y"]),
                            "yaw_degrees": float(pt.get("yaw_degrees", 0.0)),
                            "visibility": 1.0,
                        })
                        auto_idx += 1

            self.keypoints = entries
            max_id = -1
            for pt in self.keypoints:
                if "id" in pt and isinstance(pt["id"], int) and pt["id"] > max_id:
                    max_id = pt["id"]
            self.next_id = max_id + 1
            print(f"[Storage] Loaded {len(self.keypoints)} existing keypoint(s) from {self.points_path}. Next ID: {self.next_id}")
        except Exception as e:
            print(f"[Storage] [WARN] Could not parse existing points file: {e}. Starting fresh.")

    def persist(self) -> None:
        """Atomically write keypoints to disk."""
        self.points_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.points_path.with_suffix(".tmp")
        payload = {
            "map_file": str(self.map_path.name) if self.map_path else None,
            "map_sha256": self.map_hash if self.map_hash else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "keypoints": self.keypoints,
        }
        temp_file.write_text(json.dumps(payload, indent=2) + "\n")
        temp_file.replace(self.points_path)

    def _receive_point(self, msg: PointStamped) -> None:
        if msg.header.frame_id != "map":
            print("[RViz] Please set RViz Fixed Frame to 'map'.")
            return
        self.rviz_clicked_point = (msg.point.x, msg.point.y)
        self.rviz_clicked_yaw = None
        print(f"[RViz] Selected point ({msg.point.x:.2f}, {msg.point.y:.2f}).")

    def _receive_waypoint_pose(self, msg: PoseStamped) -> None:
        if msg.header.frame_id != "map":
            print("[RViz] Please set RViz Fixed Frame to 'map'.")
            return
        q = msg.pose.orientation
        norm = math.hypot(q.x, q.y, q.z, q.w)
        if norm < 1e-9:
            return
        x, y, z, w = (v / norm for v in (q.x, q.y, q.z, q.w))
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        self.rviz_clicked_point = (msg.pose.position.x, msg.pose.position.y)
        self.rviz_clicked_yaw = yaw
        print(f"[RViz] Selected pose ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}), yaw: {yaw:.1f}°.")

    def get_current_robot_pose(self) -> tuple[float, float, float, dict] | None:
        """Query real-time TF transform from 'map' to 'base_link' or 'base_footprint'."""
        for base_frame in ("base_link", "base_footprint"):
            try:
                t = self.tf_buffer.lookup_transform(
                    "map",
                    base_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
                trans = t.transform.translation
                rot = t.transform.rotation
                norm = math.hypot(rot.x, rot.y, rot.z, rot.w)
                if norm < 1e-9:
                    continue
                rx, ry, rz, rw = rot.x / norm, rot.y / norm, rot.z / norm, rot.w / norm
                yaw = math.degrees(math.atan2(2 * (rw * rz + rx * ry), 1 - 2 * (ry * ry + rz * rz)))
                quat_dict = {"x": rx, "y": ry, "z": rz, "w": rw}
                return trans.x, trans.y, yaw, quat_dict
            except Exception:
                continue
        return None

    def save_keypoint(
        self,
        name: str,
        visibility: float,
        streamer: CameraStreamer | None = None,
    ) -> dict | None:
        """Create and store a new keypoint entry."""
        name = name.strip().upper()
        if not (0.0 <= visibility <= 1.0):
            print("[ERROR] Visibility must be a float between 0.0 and 1.0 (e.g. SAVE A 0.5).")
            return None

        # 1. Try real-time robot pose in map
        current_pose = self.get_current_robot_pose()
        if current_pose is not None:
            x, y, yaw_deg, quat_dict = current_pose
        elif self.rviz_clicked_point is not None:
            # Fallback to clicked RViz point/pose if TF isn't localized
            x, y = self.rviz_clicked_point
            yaw_deg = self.rviz_clicked_yaw if self.rviz_clicked_yaw is not None else 0.0
            angle_rad = math.radians(yaw_deg) / 2.0
            quat_dict = {"x": 0.0, "y": 0.0, "z": math.sin(angle_rad), "w": math.cos(angle_rad)}
            print("[INFO] Using RViz clicked point as robot pose was unavailable via TF.")
        else:
            print("[ERROR] Robot pose unknown in 'map' frame! Make sure localization (AMCL/Nav2) is running.")
            return None

        item_id = self.next_id
        self.next_id += 1

        # Optional snapshot capture
        snapshot_rel_path = None
        if self.save_snapshots and streamer is not None:
            snap_filename = f"keypoint_{item_id:03d}_{name}.jpg"
            snap_full_path = self.snapshots_dir / snap_filename
            if streamer.save_snapshot(str(snap_full_path)):
                snapshot_rel_path = f"snapshots/{snap_filename}"
                print(f"[Snapshot] Saved camera snapshot: {snapshot_rel_path}")

        entry = {
            "id": item_id,
            "name": name,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "yaw_degrees": round(float(yaw_deg), 2),
            "position": {
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "z": 0.0,
            },
            "rotation": {
                "yaw_degrees": round(float(yaw_deg), 2),
                "quaternion": quat_dict,
            },
            "visibility": round(float(visibility), 3),
            "timestamp": time.time(),
        }
        if snapshot_rel_path:
            entry["snapshot"] = snapshot_rel_path

        self.keypoints.append(entry)
        self.persist()

        print(
            f"[SAVED] ID: {item_id} | Name: {name} | "
            f"Pose: ({x:.2f}, {y:.2f}, {yaw_deg:.1f}°) | "
            f"Visibility: {visibility:.2f}"
        )
        return entry

    def navigate_to_id(self, target_id: int) -> bool:
        """Send a Nav2 NavigateToPose goal for the given keypoint ID."""
        matching = [pt for pt in self.keypoints if pt.get("id") == target_id]
        if not matching:
            print(f"[ERROR] No keypoint found with ID {target_id}. Use 'LIST' to view available IDs.")
            return False

        point = matching[0]
        if not self.nav_client.server_is_ready():
            print("[ERROR] Nav2 server '/navigate_to_pose' is not ready. Check localization and Nav2 launch.")
            return False

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(point["x"])
        goal.pose.pose.position.y = float(point["y"])

        # Orientation
        quat = point.get("rotation", {}).get("quaternion")
        if quat and isinstance(quat, dict):
            goal.pose.pose.orientation.x = float(quat.get("x", 0.0))
            goal.pose.pose.orientation.y = float(quat.get("y", 0.0))
            goal.pose.pose.orientation.z = float(quat.get("z", 0.0))
            goal.pose.pose.orientation.w = float(quat.get("w", 1.0))
        else:
            angle_rad = math.radians(float(point.get("yaw_degrees", 0.0))) / 2.0
            goal.pose.pose.orientation.z = math.sin(angle_rad)
            goal.pose.pose.orientation.w = math.cos(angle_rad)

        self.nav_in_progress = True
        self.current_goal_id = target_id
        self.cancel_requested = False

        print(
            f"[NAV] Sending robot to Keypoint ID {target_id} ('{point.get('name')}') "
            f"at ({point['x']:.2f}, {point['y']:.2f}, {point.get('yaw_degrees', 0.0):.1f}°)..."
        )

        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_callback)
        return True

    def _goal_response_callback(self, future) -> None:
        try:
            self.goal_handle = future.result()
            if not self.goal_handle.accepted:
                print(f"[NAV] Goal for ID {self.current_goal_id} was rejected by Nav2.")
                self.nav_in_progress = False
                self.current_goal_id = None
                return

            print(f"[NAV] Goal accepted! Navigating to ID {self.current_goal_id}...")
            result_future = self.goal_handle.get_result_async()
            result_future.add_done_callback(self._goal_completed_callback)
        except Exception as e:
            print(f"[NAV] Communication error: {e}")
            self.nav_in_progress = False
            self.current_goal_id = None

    def _goal_completed_callback(self, future) -> None:
        try:
            status = future.result().status
            gid = self.current_goal_id
            if status == GoalStatus.STATUS_SUCCEEDED:
                print(f"\n[NAV SUCCESS] Reached destination for Keypoint ID {gid}!")
            elif status == GoalStatus.STATUS_CANCELED:
                print(f"\n[NAV CANCELED] Navigation to ID {gid} was canceled.")
            elif status == GoalStatus.STATUS_ABORTED:
                print(f"\n[NAV ABORTED] Navigation to ID {gid} failed/aborted by Nav2.")
            else:
                print(f"\n[NAV RESULT] Navigation to ID {gid} completed with status {status}.")
        except Exception as e:
            print(f"\n[NAV ERROR] Failed to retrieve navigation result: {e}")
        finally:
            self.nav_in_progress = False
            self.current_goal_id = None
            self.goal_handle = None

    def cancel_navigation(self) -> None:
        """Cancel active Nav2 goal."""
        if self.goal_handle is not None and self.nav_in_progress:
            print("[NAV] Canceling active navigation goal...")
            self.cancel_requested = True
            self.goal_handle.cancel_goal_async()
        else:
            print("[NAV] No active navigation to cancel.")

    def delete_keypoint(self, target_id: int) -> bool:
        """Delete keypoint by unique ID."""
        initial_len = len(self.keypoints)
        self.keypoints = [pt for pt in self.keypoints if pt.get("id") != target_id]
        if len(self.keypoints) < initial_len:
            self.persist()
            print(f"[Storage] Deleted Keypoint ID {target_id}. Remaining: {len(self.keypoints)}.")
            return True
        else:
            print(f"[Storage] No keypoint found with ID {target_id}.")
            return False

    def list_keypoints(self) -> None:
        """Print formatted table of saved keypoints."""
        if not self.keypoints:
            print("\nNo keypoints recorded yet. Drive the robot and type: SAVE <NAME> <VISIBILITY>\n")
            return

        print("\n" + "=" * 80)
        print(f"{'ID':<5} {'Name':<8} {'X (m)':<10} {'Y (m)':<10} {'Yaw (deg)':<12} {'Visibility':<12} {'Snapshot'}")
        print("-" * 80)
        for pt in self.keypoints:
            pid = pt.get("id", "-")
            name = pt.get("name", "-")
            x = f"{float(pt.get('x', 0.0)):.2f}"
            y = f"{float(pt.get('y', 0.0)):.2f}"
            yaw = f"{float(pt.get('yaw_degrees', 0.0)):.1f}°"
            vis = f"{float(pt.get('visibility', 1.0)):.2f}"
            snap = pt.get("snapshot", "")
            print(f"{pid:<5} {name:<8} {x:<10} {y:<10} {yaw:<12} {vis:<12} {snap}")
        print("=" * 80)
        print(f"Total: {len(self.keypoints)} keypoint(s) | Stored at: {self.points_path}\n")

    def print_status(self, streamer: CameraStreamer | None) -> None:
        """Print system status."""
        pose = self.get_current_robot_pose()
        pose_str = f"X={pose[0]:.2f}m, Y={pose[1]:.2f}m, Yaw={pose[2]:.1f}°" if pose else "Not Localized (Check TF/AMCL)"
        cam_fps = streamer.grab_fps if streamer else 0.0
        nav_state = f"Navigating to ID {self.current_goal_id}" if self.nav_in_progress else "Idle"

        print("\n--- System Status ---")
        print(f"Robot Map Pose:   {pose_str}")
        print(f"Camera Stream:    {cam_fps:.1f} FPS (Active: {streamer.active_indices if streamer else 'None'})")
        print(f"Nav2 Status:      {nav_state}")
        print(f"Keypoints File:   {self.points_path} ({len(self.keypoints)} entries, Next ID: {self.next_id})")
        print("---------------------\n")

    def teleop_twist(self, linear_x: float, angular_z: float) -> None:
        """Send quick velocity command for camera adjustment."""
        t = Twist()
        t.linear.x = float(linear_x)
        t.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(t)


def draw_hud(
    frame: np.ndarray,
    node: KeypointConsoleNode | None,
    fps: float,
    last_action: str = "",
) -> np.ndarray:
    """Overlay map coordinates, camera FPS, and instructions on OpenCV frame."""
    h, w = frame.shape[:2]
    # Draw top banner background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    # Bottom banner background
    cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # Pose text
    pose_text = "Pose: [Waiting for TF localization]"
    if node:
        pose = node.get_current_robot_pose()
        if pose:
            pose_text = f"Pose: X={pose[0]:.2f}m | Y={pose[1]:.2f}m | Yaw={pose[2]:.1f}deg"

    cv2.putText(frame, pose_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    count = len(node.keypoints) if node else 0
    stat_text = f"Saved: {count} | Cam: {fps:.1f} FPS"
    cv2.putText(frame, stat_text, (w - 280, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

    # Navigation or last action text
    if node and node.nav_in_progress:
        action_text = f"Navigating to ID {node.current_goal_id}..."
        color = (0, 165, 255)
    elif last_action:
        action_text = last_action
        color = (255, 255, 255)
    else:
        action_text = "Enter 'SAVE <NAME> <VIS>' in terminal (e.g. SAVE A 0.5) | 's' for snapshot | 'q' to quit"
        color = (200, 200, 200)

    cv2.putText(frame, action_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Teleop (window focused): W/S=drive, A/D=turn, Space=stop",
        (15, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return frame


def terminal_input_thread(cmd_queue: queue.Queue, stop_event: threading.Event) -> None:
    """Read lines from standard input in background thread."""
    while not stop_event.is_set():
        try:
            line = input()
            cmd_queue.put(line.strip())
            if line.strip().upper() in ("QUIT", "EXIT"):
                break
        except (EOFError, KeyboardInterrupt):
            cmd_queue.put("QUIT")
            break


def main() -> None:
    # Print description string on startup
    print(DESCRIPTION_BANNER)

    script_dir = Path(__file__).resolve().parent
    default_points = str(script_dir / "maps" / "stage2_keypoints.json")
    default_map = str(script_dir / "maps" / "final_room.yaml")

    parser = argparse.ArgumentParser(description="Stage 2 Camera View & Keypoint Collector.")
    parser.add_argument(
        "--points",
        "-p",
        type=str,
        default=default_points,
        help=f"Path where keypoints JSON file will be placed (default: {default_points}).",
    )
    parser.add_argument(
        "--map",
        "-m",
        type=str,
        default=default_map,
        help=f"Optional path to map YAML file (default: {default_map}).",
    )
    parser.add_argument(
        "--cameras",
        "--indices",
        "-c",
        dest="cameras",
        type=int,
        nargs="+",
        default=None,
        help="Camera device index/indices (e.g. 0 for single camera, or 0 2 for stereo). If omitted, auto-detects USB cameras.",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640; try 1280 or 1920).")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480; try 720 or 1080).")
    parser.add_argument("--fps", type=float, default=30.0, help="Target capture FPS (default: 30.0; try 15.0).")
    parser.add_argument("--no-snapshots", action="store_true", help="Disable saving snapshot images per keypoint.")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window.")
    args = parser.parse_args()

    points_path = Path(args.points)
    map_path = Path(args.map) if args.map else None

    # Check display server for GUI
    has_display = ("DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ) and not args.no_display
    if not has_display:
        print("[INFO] Headless environment detected: running without OpenCV GUI window.")

    # Initialize ROS 2
    if not ROS_AVAILABLE:
        print("\n[FATAL] ROS 2 Humble packages not found.")
        print("Please source ROS 2 first: source /opt/ros/humble/setup.bash\n")
        sys.exit(1)

    rclpy.init()
    node = KeypointConsoleNode(
        points_path=points_path,
        map_path=map_path,
        save_snapshots=not args.no_snapshots,
    )

    # Start camera grabber
    print(f"[INFO] Initializing camera(s) (Indices: {args.cameras or 'auto'})...")
    streamer = CameraStreamer(
        camera_indices=args.cameras,
        width=args.width,
        height=args.height,
        fps=args.fps,
        allow_mock=True,
    )
    streamer.start()

    cmd_queue: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    input_t = threading.Thread(target=terminal_input_thread, args=(cmd_queue, stop_event), daemon=True)
    input_t.start()

    print(f"[Ready] Keypoint Collector is active! Target file: {points_path}")
    print("Drive the robot and enter commands in this terminal (e.g. 'SAVE A 0.5', 'GOTO 2', 'SAVE', 'LIST').\n")

    last_action = ""
    running = True

    try:
        while rclpy.ok() and running:
            # Spin ROS callbacks non-blockingly
            rclpy.spin_once(node, timeout_sec=0.01)

            # Check for terminal commands
            try:
                line = cmd_queue.get_nowait()
            except queue.Empty:
                line = ""

            if line:
                tokens = line.replace(",", " ").split()
                if not tokens:
                    continue
                cmd = tokens[0].upper()

                if cmd in ("QUIT", "EXIT"):
                    print("\n[INFO] Saving data and shutting down...")
                    node.persist()
                    running = False
                    break

                elif cmd == "SAVE":
                    if len(tokens) == 1:
                        # User typed just 'SAVE' -> flush to disk immediately
                        node.persist()
                        last_action = f"Saved JSON ({len(node.keypoints)} items to {points_path.name})"
                        print(f"[SAVE] Manually flushed {len(node.keypoints)} keypoint(s) to {points_path}.")
                    elif len(tokens) >= 3:
                        # User typed 'SAVE A 0.5'
                        name = tokens[1]
                        try:
                            vis = float(tokens[2])
                            saved = node.save_keypoint(name, vis, streamer)
                            if saved:
                                last_action = f"Saved {name} (ID {saved['id']}, Vis {vis:.2f})"
                        except ValueError:
                            print(f"[ERROR] Invalid visibility value '{tokens[2]}'. Must be a float (e.g. 0.5).")
                    else:
                        print("[ERROR] Usage: SAVE <NAME> <VISIBILITY> (e.g. SAVE A 0.5) or simply SAVE to flush.")

                elif cmd == "GOTO":
                    if len(tokens) < 2:
                        print("[ERROR] Usage: GOTO <ID> (e.g. GOTO 2)")
                    else:
                        try:
                            target_id = int(tokens[1])
                            success = node.navigate_to_id(target_id)
                            if success:
                                last_action = f"Navigating to ID {target_id}..."
                        except ValueError:
                            print(f"[ERROR] Invalid ID '{tokens[1]}'. Must be an integer (e.g. GOTO 2).")

                elif cmd == "LIST":
                    node.list_keypoints()

                elif cmd == "DELETE":
                    if len(tokens) < 2:
                        print("[ERROR] Usage: DELETE <ID>")
                    else:
                        try:
                            del_id = int(tokens[1])
                            node.delete_keypoint(del_id)
                            last_action = f"Deleted Keypoint ID {del_id}"
                        except ValueError:
                            print(f"[ERROR] Invalid ID '{tokens[1]}'.")

                elif cmd == "CANCEL":
                    node.cancel_navigation()
                    last_action = "Navigation canceled"

                elif cmd == "STATUS":
                    node.print_status(streamer)

                elif cmd == "HELP":
                    print(DESCRIPTION_BANNER)

                else:
                    print(f"[WARN] Unknown command '{cmd}'. Available: SAVE <NAME> <VIS>, SAVE, GOTO <ID>, LIST, DELETE <ID>, STATUS, CANCEL, QUIT")

            # Camera display and window interaction
            frame, frame_id, fps = streamer.get_latest_frame()
            if frame is not None and has_display:
                frame_hud = draw_hud(frame, node, fps, last_action)
                cv2.imshow("Stage 2: Camera View & Keypoints", frame_hud)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\n[INFO] 'q' pressed in window. Exiting...")
                    node.persist()
                    running = False
                    break
                elif key == ord("s"):
                    # Manual snapshot
                    ts = int(time.time())
                    snap_p = node.snapshots_dir / f"manual_snapshot_{ts}.jpg"
                    if streamer.save_snapshot(str(snap_p)):
                        print(f"[Snapshot] Saved manual snapshot: {snap_p}")
                        last_action = f"Snapshot saved ({snap_p.name})"
                # Optional teleop hotkeys when OpenCV window is focused
                elif key == ord("w"):
                    node.teleop_twist(0.15, 0.0)
                elif key == ord("s"):
                    node.teleop_twist(-0.15, 0.0)
                elif key == ord("a"):
                    node.teleop_twist(0.0, 0.4)
                elif key == ord("d"):
                    node.teleop_twist(0.0, -0.4)
                elif key == 32:  # Spacebar
                    node.teleop_twist(0.0, 0.0)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        stop_event.set()
        streamer.stop()
        if has_display:
            cv2.destroyAllWindows()
        if node and ROS_AVAILABLE:
            if node.nav_in_progress:
                node.cancel_navigation()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[INFO] Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()

