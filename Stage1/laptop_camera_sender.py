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
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG compression quality (1-100, default: 80).",
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


def stream_http(args: argparse.Namespace, cam_l: cv2.VideoCapture, cam_r: cv2.VideoCapture) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0

    headers = {
        "Content-Type": "image/jpeg",
        "Connection": "keep-alive",
    }

    while True:
        try:
            print(f"[LaptopSender] Connecting via HTTP to http://{args.server}:{args.port}/api/push_frame...")
            conn = http.client.HTTPConnection(args.server, args.port, timeout=4.0)
            conn.connect()
            print(f"[LaptopSender] Connected! Streaming stereo frames directly via HTTP port {args.port}...")

            frame_count = 0
            fps_timer = time.perf_counter()

            while True:
                loop_start = time.perf_counter()

                # Synchronized shutter grab
                cam_l.grab()
                cam_r.grab()
                ret_l, frame_l = cam_l.retrieve()
                ret_r, frame_r = cam_r.retrieve()

                if not ret_l or not ret_r or frame_l is None or frame_r is None:
                    time.sleep(0.01)
                    continue

                # Combine side-by-side: [Left | Right]
                composite = np.hstack((frame_l, frame_r))
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


def stream_tcp(args: argparse.Namespace, cam_l: cv2.VideoCapture, cam_r: cv2.VideoCapture) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0

    while True:
        try:
            print(f"[LaptopSender] Connecting via TCP to {args.server}:{args.port}...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((args.server, args.port))
            print(f"[LaptopSender] Connected! Streaming stereo frames via TCP port {args.port}...")

            frame_count = 0
            fps_timer = time.perf_counter()

            while True:
                loop_start = time.perf_counter()

                cam_l.grab()
                cam_r.grab()
                ret_l, frame_l = cam_l.retrieve()
                ret_r, frame_r = cam_r.retrieve()

                if not ret_l or not ret_r or frame_l is None or frame_r is None:
                    time.sleep(0.01)
                    continue

                composite = np.hstack((frame_l, frame_r))
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

    try:
        if args.protocol == "http":
            stream_http(args, cam_l, cam_r)
        else:
            stream_tcp(args, cam_l, cam_r)
    finally:
        cam_l.release()
        cam_r.release()
        print("[LaptopSender] Cameras released.")


if __name__ == "__main__":
    main()
