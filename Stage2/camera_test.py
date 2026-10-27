#!/usr/bin/env python3
"""Stage 2: Ubuntu Camera Test Script.

Adapted for Linux / Ubuntu / VMware environments.
Supports single-camera mode or stereo pairs. Uses V4L2 with MJPG compression
for smooth, artifact-free video in VMware virtual machines.

Usage:
    # Auto-detect available cameras (defaults to 640x480 for smooth VMware USB streaming):
    python3 camera_test.py

    # Custom resolution (e.g. 1280x720 @ 15 FPS):
    python3 camera_test.py --cameras 0 --width 1280 --height 720 --fps 15

    # 1080p full resolution:
    python3 camera_test.py --cameras 0 --width 1920 --height 1080 --fps 15
"""

from __future__ import annotations

import argparse
import glob
import os
import platform
import sys
import threading
import time
import cv2
import numpy as np


DESCRIPTION = """
================================================================================
Stage 2 Ubuntu Camera Test Utility
================================================================================
This utility tests video capture from USB cameras on Linux/Ubuntu.
It works with a SINGLE camera or STEREO pairs.

Controls:
  - 's': Save a snapshot from all active cameras
  - 'q' or ESC: Quit test
================================================================================
"""

VMWARE_HELP = """
--------------------------------------------------------------------------------
[VMware Camera Setup Guide]
No active video capture devices were detected inside this Linux virtual machine.
By default, VMware leaves USB devices connected to your host laptop (Windows/macOS).

To attach your USB camera to this Ubuntu VM:
  1. In the VMware top menu:
     VM -> Removable Devices -> Select your USB Camera -> Connect (Disconnect from host)
     (Or right-click the webcam icon in the bottom-right corner of the VMware window).

  2. If your camera is NOT listed under Removable Devices:
     - Shut down the VM.
     - Open: VM -> Settings -> USB Controller.
     - Change "USB compatibility" from USB 2.0 to USB 3.1 (or USB 3.0).
     - Check "Show all USB input devices".
     - Click OK, boot the VM, and reconnect the camera.

  3. Grant user permission (run once in terminal):
     sudo usermod -aG video $USER
     (or: sudo chmod 666 /dev/video*)

  4. Verify with:
     ls -l /dev/video*
--------------------------------------------------------------------------------
"""


class SuppressStderr:
    """Temporarily silence C++ driver probe warnings on non-capture video nodes."""

    def __enter__(self):
        try:
            self.devnull = os.open(os.devnull, os.O_RDWR)
            self.orig_stderr = os.dup(2)
            os.dup2(self.devnull, 2)
        except Exception:
            self.devnull = None
            self.orig_stderr = None

    def __exit__(self, *args):
        if self.devnull is not None and self.orig_stderr is not None:
            try:
                os.dup2(self.orig_stderr, 2)
                os.close(self.orig_stderr)
                os.close(self.devnull)
            except Exception:
                pass


def detect_camera_indices() -> list[int]:
    """Auto-detect available video capture device indices on Linux using MJPG probe."""
    video_devices = sorted(glob.glob("/dev/video*"))
    if not video_devices:
        return []

    available: list[int] = []
    with SuppressStderr():
        for dev in video_devices:
            basename = os.path.basename(dev)
            try:
                idx = int(basename.replace("video", ""))
            except ValueError:
                continue

            backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # Set resolution first, then MJPG to avoid buffer negotiation glitches
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available.append(idx)
                cap.release()

    return available


def open_camera(
    idx: int,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> cv2.VideoCapture | None:
    dev_path = f"/dev/video{idx}"
    if platform.system() == "Linux" and not os.path.exists(dev_path):
        print(f"[FAIL] Device node '{dev_path}' does not exist.")
        print(f"       The camera is not connected to this VM. See VMware setup guide above.")
        return None

    if platform.system() == "Linux" and not os.access(dev_path, os.R_OK):
        print(f"[FAIL] Permission denied on '{dev_path}'.")
        print(f"       Run: sudo usermod -aG video $USER  (or: sudo chmod 666 {dev_path})")
        return None

    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap = cv2.VideoCapture(idx)

    if not cap.isOpened():
        print(f"[FAIL] Could not open camera {idx}.")
        return None

    # In DirectShow & V4L2 UVC, set resolution first, then MJPG FOURCC, then FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Warm-up reads to settle auto-exposure and frame buffer
    warmup_ok = False
    for _ in range(6):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            warmup_ok = True
            break
        time.sleep(0.04)

    if not warmup_ok:
        # Fallback to default format if MJPG not negotiated
        print(f"[INFO] Camera {idx} did not negotiate MJPG at {width}x{height}. Trying default format...")
        cap.release()
        cap = cv2.VideoCapture(idx)
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                warmup_ok = True
                break
            time.sleep(0.04)

    if not warmup_ok or not cap.isOpened():
        print(f"[FAIL] Camera {idx} opened but could not retrieve frames.")
        cap.release()
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"[OK] Camera {idx} initialized: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    return cap


def main() -> None:
    print(DESCRIPTION)
    parser = argparse.ArgumentParser(description="Test camera capture on Ubuntu / VMware.")
    parser.add_argument(
        "--indices",
        "--cameras",
        "-c",
        dest="indices",
        type=int,
        nargs="+",
        default=None,
        help="Camera device index/indices (e.g. 0 for single camera, or 0 2 for stereo). Default: auto-detect.",
    )
    parser.add_argument("--http", type=int, default=None, help="Listen as HTTP receiver on this port (e.g. 8080) for frames from laptop_camera_sender.py")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640; try 1280 or 1920)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480; try 720 or 1080)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target capture FPS (default: 30.0; try 15.0)")
    args = parser.parse_args()

    # Network HTTP receiver mode
    if args.http:
        import http.server, socketserver
        print(f"[INFO] Running in Network Receiver mode on port {args.http} (/api/push_frame)...")
        latest_network_frame = [None]
        lock = threading.Lock()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/api/push_frame":
                    length = int(self.headers.get("Content-Length", 0))
                    if length > 0:
                        payload = self.rfile.read(length)
                        arr = np.frombuffer(payload, dtype=np.uint8)
                        f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if f is not None:
                            with lock:
                                latest_network_frame[0] = f
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"success":true}')
                else:
                    self.send_response(404)
                    self.end_headers()
            def log_message(self, *a): pass

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        port = args.http
        srv = None
        for p in [port, port + 1, 8088, 5000]:
            try:
                srv = Server(("0.0.0.0", p), Handler)
                port = p
                break
            except OSError:
                continue

        if srv is None:
            print(f"[ERROR] Could not bind HTTP receiver to port {args.http} or alternate ports.")
            return

        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[INFO] Waiting for stream from laptop_camera_sender.py on port {port} (/api/push_frame)...")
        has_display = "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ
        try:
            while True:
                with lock:
                    f = latest_network_frame[0].copy() if latest_network_frame[0] is not None else None
                if f is not None and has_display:
                    cv2.imshow(f"Network Stream (Port {args.http})", f)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
                time.sleep(0.01)
        finally:
            srv.shutdown()
            cv2.destroyAllWindows()
            print("[INFO] Receiver closed.")
        return

    indices = args.indices
    if indices is None:
        print("[INFO] Scanning for connected video devices (/dev/video*)...")
        indices = detect_camera_indices()
        if not indices:
            print("[ERROR] No active video capture devices detected in /dev/video*.")
            print(VMWARE_HELP)
            sys.exit(1)
        elif len(indices) == 1:
            print(f"[INFO] Detected 1 camera: Index {indices[0]} (Single-camera mode).")
        else:
            print(f"[INFO] Detected {len(indices)} camera(s): Indices {indices}.")

    print(f"[INFO] Attempting to open camera(s): {indices}")
    cameras: dict[int, cv2.VideoCapture] = {}
    for idx in indices:
        cap = open_camera(idx, width=args.width, height=args.height, fps=args.fps)
        if cap is not None:
            cameras[idx] = cap

    if not cameras:
        print("\n[ERROR] No cameras could be opened.")
        if platform.system() == "Linux" and not glob.glob("/dev/video*"):
            print(VMWARE_HELP)
        sys.exit(1)

    mode_str = "Single-camera" if len(cameras) == 1 else f"Stereo/Multi-camera ({len(cameras)} cameras)"
    print(f"\n[INFO] Successfully opened {mode_str}! Live preview running.")
    print("Press 's' in video window to save snapshot, 'q' to quit.\n")

    has_display = "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ
    last_time = time.perf_counter()
    fps_smooth = 30.0

    try:
        while True:
            t0 = time.perf_counter()
            frames: list[np.ndarray] = []
            for idx, cap in list(cameras.items()):
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Smooth FPS calculation
                    dt = t0 - last_time
                    if dt > 0.001:
                        current_fps = 1.0 / dt
                        fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps
                    last_time = t0

                    h, w = frame.shape[:2]
                    info_text = f"Cam {idx} | {w}x{h} | {fps_smooth:.1f} FPS"
                    cv2.putText(
                        frame,
                        info_text,
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    frames.append(frame)

            if frames:
                composite = np.hstack(frames) if len(frames) > 1 else frames[0]
                if has_display:
                    window_title = f"Stage 2 Camera View ({'Single' if len(frames) == 1 else 'Stereo'})"
                    try:
                        cv2.imshow(window_title, composite)
                    except cv2.error:
                        has_display = False
                        print("\n[INFO] GUI display server unavailable; running in headless stream mode.")

            if has_display:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    ts = int(time.time())
                    for idx, cap in cameras.items():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            fname = f"stage2_snapshot_cam_{idx}_{ts}.jpg"
                            cv2.imwrite(fname, frame)
                            print(f"[SNAPSHOT] Saved {fname}")
            else:
                time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        for cap in cameras.values():
            cap.release()
        if has_display:
            cv2.destroyAllWindows()
        print("\n[INFO] Cameras released. Test completed.")


if __name__ == "__main__":
    main()
