"""Main entry point for Windows Mission Control (Stage 2).

Launches the GUI dashboard, initializes stereo camera grabbing, connects to
the VM robot controller and SSH size server, and runs autonomous perception workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tkinter as tk

# Ensure Stage2/stage2_system is in sys.path
sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from shared.keypoints import KeypointStore, MapInfo
from shared.protocol import (
    DEFAULT_LAPTOP_PING_PORT,
    DEFAULT_SSH_SIZE_PORT,
    DEFAULT_VM_CMD_PORT,
)
from laptop.camera import CameraManager
from laptop.commands import CommandDispatcher
from laptop.console import ConsoleRedirector
from laptop.gui import MissionControlGUI
from laptop.llm import llm_api
from laptop.map_view import MapRenderer
from laptop.mission import MissionManager
from laptop.robot_link import RobotLink
from laptop.size_link import SizeLink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurtleBot4 Stage 2 Windows Mission Control")

    # Network / Remote Services
    parser.add_argument(
        "--vm",
        type=str,
        default="auto",
        help="IP address of Ubuntu VM running robot_controller.py, or 'auto' (default: auto).",
    )
    parser.add_argument(
        "--vm-port",
        type=int,
        default=DEFAULT_VM_CMD_PORT,
        help=f"HTTP command port on VM (default: {DEFAULT_VM_CMD_PORT}).",
    )
    parser.add_argument(
        "--ping-port",
        type=int,
        default=DEFAULT_LAPTOP_PING_PORT,
        help=f"Local HTTP port to receive VM status pings (default: {DEFAULT_LAPTOP_PING_PORT}).",
    )
    parser.add_argument(
        "--ssh-url",
        type=str,
        default=f"http://127.0.0.1:{DEFAULT_SSH_SIZE_PORT}",
        help=f"URL of SSH GPU Size Server (default: http://127.0.0.1:{DEFAULT_SSH_SIZE_PORT}).",
    )

    # Keypoints and Map files (Read strictly in-place, never copied or overwritten)
    stage2_root = sys_root.parent
    default_points = stage2_root / "vm_map_tools" / "maps" / "stage2_keypoints.json"
    default_map = stage2_root / "vm_map_tools" / "maps" / "final_room.yaml"

    parser.add_argument(
        "--points",
        type=Path,
        default=default_points,
        help=f"Path to stage2_keypoints.json (default: {default_points}).",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=default_map,
        help=f"Path to map YAML file (default: {default_map}).",
    )

    # Stereo Camera Arguments
    parser.add_argument("--left-cam", type=int, default=0, help="Device index of left webcam (default: 0).")
    parser.add_argument("--right-cam", type=int, default=2, help="Device index of right webcam (default: 2).")
    parser.add_argument("--single-cam", type=int, default=None, help="Device index for single camera mode.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720).")
    parser.add_argument("--fps", type=float, default=30.0, help="Target capture FPS (default: 30.0).")

    # Simulation & Offline flags
    parser.add_argument("--mock-camera", action="store_true", help="Use synthetic test pattern feed.")
    parser.add_argument("--mock-robot", action="store_true", help="Run with internal mock robot.")
    parser.add_argument("--mock-llm", action="store_true", help="Use offline canned LLM responses.")
    parser.add_argument("--mock-size", action="store_true", help="Use offline simulated size estimation.")
    parser.add_argument("--no-gui", action="store_true", help="Run headless in terminal mode without Tkinter GUI.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Initialize Console Redirector
    log_dir = sys_root / "runs"
    console = ConsoleRedirector(log_dir=log_dir)
    console.start()

    print("=" * 70)
    print("  STAGE 2 AUTONOMOUS MISSION CONTROL (WINDOWS)")
    print(f"  VM Target: {args.vm}:{args.vm_port} | SSH Size Server: {args.ssh_url}")
    print(f"  Keypoints: {args.points.name} | Map: {args.map.name}")
    print("=" * 70)

    # Configure LLM mode
    if args.mock_llm:
        print("[Setup] Mock LLM enabled (offline testing).")
        llm_api.set_mock_mode(True)

    # 2. Keypoints and Map Store (In-place read-only)
    kps = KeypointStore(file_path=args.points)
    print(f"[Setup] Loaded {len(kps.get_all())} keypoint(s) from {args.points}.")

    map_info = MapInfo(yaml_path=args.map)
    map_renderer = MapRenderer(map_info=map_info)

    # 3. Initialize Camera
    camera = CameraManager(
        left_cam=args.left_cam,
        right_cam=args.right_cam,
        single_cam=args.single_cam,
        width=args.width,
        height=args.height,
        fps=args.fps,
        is_mock=args.mock_camera,
    )
    camera.start()

    # 4. Initialize Robot Link & Size Link
    robot = RobotLink(vm_ip=args.vm, vm_port=args.vm_port, ping_port=args.ping_port)
    robot.start()

    size_link = SizeLink(server_url=args.ssh_url, is_mock=args.mock_size)

    # 5. Initialize Mission Manager & Command Dispatcher
    mission = MissionManager(
        camera=camera,
        robot=robot,
        size_link=size_link,
        keypoints=kps,
        runs_dir=log_dir,
    )
    dispatcher = CommandDispatcher(mission=mission)

    # 6. Launch Tkinter GUI or Headless Terminal loop
    if not args.no_gui:
        root = tk.Tk()
        gui = MissionControlGUI(root, mission, dispatcher, map_renderer)
        try:
            root.mainloop()
        finally:
            mission.stop()
            camera.stop()
            robot.stop()
            console.stop()
    else:
        print("[Setup] Running in headless terminal mode. Enter commands (e.g. 'Level 1', 'Help', 'Quit'):")
        try:
            while True:
                line = input()
                if line.strip().upper() in ("QUIT", "EXIT"):
                    break
                out = dispatcher.execute(line)
                if out:
                    print(out)
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            mission.stop()
            camera.stop()
            robot.stop()
            console.stop()


if __name__ == "__main__":
    main()

