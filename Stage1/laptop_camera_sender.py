#!/usr/bin/env python3
"""Laptop Camera Sender for Binocular Stereo Perception (Stage 1).

Run this script LOCALLY on your laptop where the USB binocular cameras are plugged in.
It captures synchronized frames from both cameras, compresses them, and streams them
directly to the GPU server running Stage1/harness.py.

Default protocol is HTTP Push directly to port 8080 (the exact same port as the Web UI).
This requires ZERO extra port configuration in VS Code!

Usage (Recommended - uses port 8080 which is already forwarded):
    python laptop_camera_sender.py --server 127.0.0.1 --port 8080 --left-cam 0 --right-cam 2

Or with TCP socket:
    python laptop_camera_sender.py --protocol tcp --server 127.0.0.1 --port 5555 --left-cam 0 --right-cam 2
"""

from __future__ import annotations
import argparse
import http.client
import platform
import socket
import struct
import sys
import threading
import time
import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=["http", "tcp"],
        default="http",
        help="Streaming protocol: 'http' (recommended, reuses port 8080) or 'tcp' (default: http).",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1",
        help="Server IP address or hostname running harness.py (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Target port (default: 8080 for HTTP, 5555 for TCP).",
    )
    parser.add_argument(
        "--left-cam",
        type=int,
        default=0,
        help="Device index of left camera (default: 0).",
    )
    parser.add_argument(
        "--right-cam",
        type=int,
        default=2,
        help="Device index of right camera (default: 2).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Capture frame width (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Capture frame height (default: 1080).",
    )
    parser.add_argument(
        "--stream-scale",
        type=float,
        default=0.5,
        help="Downscale factor before transmitting (e.g. 0.5 for 960x540, 1.0 for 1080p). Default: 0.5.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG compression quality (1-100, default: 80).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Network request timeout in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--fps-limit",
        type=float,
        default=30.0,
        help="Maximum capture FPS limit (default: 30.0).",
    )
    return parser.parse_args()


def get_backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def open_camera(idx: int, width: int, height: int, backend: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera device {idx}. Check connections and permissions."
        )

    # Configure MJPG format for high-speed USB 1080p capture if supported
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class CameraGrabber:
    """Threaded camera grabber that decouples hardware acquisition from network transmission."""

    def __init__(self, cam_l: cv2.VideoCapture, cam_r: cv2.VideoCapture) -> None:
        self.cam_l = cam_l
        self.cam_r = cam_r
        self._latest_pair: tuple[np.ndarray, np.ndarray] | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            self.cam_l.grab()
            self.cam_r.grab()
            ret_l, frame_l = self.cam_l.retrieve()
            ret_r, frame_r = self.cam_r.retrieve()
            if ret_l and ret_r and frame_l is not None and frame_r is not None:
                with self._lock:
                    self._latest_pair = (frame_l, frame_r)
            else:
                time.sleep(0.005)

    def get_latest_pair(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            if self._latest_pair is None:
                return None
            return self._latest_pair[0].copy(), self._latest_pair[1].copy()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


def stream_http(args: argparse.Namespace, grabber: CameraGrabber) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0
    scale = max(0.1, min(1.0, float(args.stream_scale)))

    headers = {
        "Content-Type": "image/jpeg",
        "Connection": "keep-alive",
    }

    while True:
        try:
            print(f"[LaptopSender] Connecting via HTTP to http://{args.server}:{args.port}/api/push_frame...")
            conn = http.client.HTTPConnection(args.server, args.port, timeout=args.timeout)
            conn.connect()
            print(
                f"[LaptopSender] Connected! Streaming stereo frames via HTTP port {args.port} "
                f"(Scale: {scale:.2f}x | Quality: {args.jpeg_quality})..."
            )

            frame_count = 0
            fps_timer = time.perf_counter()

            while True:
                loop_start = time.perf_counter()

                pair = grabber.get_latest_pair()
                if pair is None:
                    time.sleep(0.005)
                    continue
                frame_l, frame_r = pair

                # Combine side-by-side: [Left | Right]
                composite = np.hstack((frame_l, frame_r))

                # Apply stream scaling to save network bandwidth if requested
                if scale < 0.999:
                    ch, cw = composite.shape[:2]
                    target_w = int(cw * scale)
                    target_h = int(ch * scale)
                    composite = cv2.resize(composite, (target_w, target_h), interpolation=cv2.INTER_AREA)

                success, encoded_jpg = cv2.imencode(".jpg", composite, encode_param)
                if not success:
                    continue

                payload = encoded_jpg.tobytes()
                conn.request("POST", "/api/push_frame", body=payload, headers=headers)
                res = conn.getresponse()
                res.read()  # read response body to keep connection reusable

                frame_count += 1
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    stream_fps = frame_count / (now - fps_timer)
                    kb_per_frame = len(payload) / 1024.0
                    sys.stdout.write(
                        f"\r[HTTP Streaming] {stream_fps:.1f} FPS | Frame size: {kb_per_frame:.1f} KB | Rate: {stream_fps * kb_per_frame / 1024.0:.2f} MB/s"
                    )
                    sys.stdout.flush()
                    frame_count = 0
                    fps_timer = now

                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        except (ConnectionRefusedError, BrokenPipeError, ConnectionResetError, http.client.HTTPException, socket.timeout) as e:
            print(f"\n[LaptopSender] HTTP connection error ({e}). Retrying in 2 seconds...")
            time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n[LaptopSender] Stopped by user.")
            break
        finally:
            try:
                conn.close()
            except Exception:
                pass


def stream_tcp(args: argparse.Namespace, grabber: CameraGrabber) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0
    scale = max(0.1, min(1.0, float(args.stream_scale)))

    while True:
        try:
            print(f"[LaptopSender] Connecting via TCP to {args.server}:{args.port}...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((args.server, args.port))
            print(
                f"[LaptopSender] Connected! Streaming stereo frames via TCP port {args.port} "
                f"(Scale: {scale:.2f}x | Quality: {args.jpeg_quality})..."
            )

            frame_count = 0
            fps_timer = time.perf_counter()

            while True:
                loop_start = time.perf_counter()

                pair = grabber.get_latest_pair()
                if pair is None:
                    time.sleep(0.005)
                    continue
                frame_l, frame_r = pair

                composite = np.hstack((frame_l, frame_r))
                if scale < 0.999:
                    ch, cw = composite.shape[:2]
                    target_w = int(cw * scale)
                    target_h = int(ch * scale)
                    composite = cv2.resize(composite, (target_w, target_h), interpolation=cv2.INTER_AREA)

                success, encoded_jpg = cv2.imencode(".jpg", composite, encode_param)
                if not success:
                    continue

                payload = encoded_jpg.tobytes()
                header = struct.pack(">Q", len(payload))
                client_socket.sendall(header + payload)

                frame_count += 1
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    stream_fps = frame_count / (now - fps_timer)
                    kb_per_frame = len(payload) / 1024.0
                    sys.stdout.write(
                        f"\r[TCP Streaming] {stream_fps:.1f} FPS | Frame size: {kb_per_frame:.1f} KB | Rate: {stream_fps * kb_per_frame / 1024.0:.2f} MB/s"
                    )
                    sys.stdout.flush()
                    frame_count = 0
                    fps_timer = now

                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        except (ConnectionRefusedError, BrokenPipeError, ConnectionResetError, socket.timeout) as e:
            print(f"\n[LaptopSender] TCP connection error ({e}). Retrying in 2 seconds...")
            time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n[LaptopSender] Stopped by user.")
            break
        finally:
            try:
                client_socket.close()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    backend = get_backend()
    print(f"[LaptopSender] OS: {platform.system()} | Camera backend: {backend}")
    print(
        f"[LaptopSender] Opening Left Camera (ID {args.left_cam}) and Right Camera (ID {args.right_cam})..."
    )

    cam_l = open_camera(args.left_cam, args.width, args.height, backend)
    cam_r = open_camera(args.right_cam, args.width, args.height, backend)

    actual_w = int(cam_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cam_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[LaptopSender] Left Camera resolution: {actual_w}x{actual_h}")

    grabber = CameraGrabber(cam_l, cam_r)
    try:
        if args.protocol == "http":
            stream_http(args, grabber)
        else:
            stream_tcp(args, grabber)
    finally:
        grabber.stop()
        cam_l.release()
        cam_r.release()
        print("[LaptopSender] Cameras released.")


if __name__ == "__main__":
    main()
