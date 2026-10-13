#!/usr/bin/env python3
"""Laptop Camera Sender for Binocular Stereo Perception (Stage 1).

Run this script LOCALLY on your laptop where the USB binocular cameras are plugged in.
It captures synchronized frames from both cameras, compresses them, and streams them
over a TCP socket to the GPU server running Stage1/harness.py.

Usage:
    python laptop_camera_sender.py --server <SERVER_IP> --port 5555 --left-cam 0 --right-cam 1

If using SSH port forwarding, you can forward port 5555 from server to laptop:
    ssh -R 5555:localhost:5555 user@server_ip
    then run:
    python laptop_camera_sender.py --server 127.0.0.1 --port 5555
"""

from __future__ import annotations
import argparse
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
        "--server",
        type=str,
        default="127.0.0.1",
        help="Server IP address or hostname running harness.py (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="TCP port to stream frames to (default: 5555).",
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
        default=1,
        help="Device index of right camera (default: 1).",
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
        default=85,
        help="JPEG compression quality (1-100, default: 85).",
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
        # Fallback to default backend
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {idx}. Check connections and permissions.")

    # Configure MJPG format for high-speed USB 1080p capture if supported
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main() -> None:
    args = parse_args()
    backend = get_backend()
    print(f"[LaptopSender] OS: {platform.system()} | Camera backend: {backend}")
    print(f"[LaptopSender] Opening Left Camera (ID {args.left_cam}) and Right Camera (ID {args.right_cam})...")

    cam_l = open_camera(args.left_cam, args.width, args.height, backend)
    cam_r = open_camera(args.right_cam, args.width, args.height, backend)

    actual_w = int(cam_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cam_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[LaptopSender] Left Camera resolution: {actual_w}x{actual_h}")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0

    while True:
        try:
            print(f"[LaptopSender] Connecting to {args.server}:{args.port}...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((args.server, args.port))
            print(f"[LaptopSender] Connected successfully! Streaming stereo frames...")

            last_time = time.perf_counter()
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
                    print("[LaptopSender] Warning: frame grab dropped.")
                    time.sleep(0.01)
                    continue

                # Combine side-by-side: [Left | Right]
                composite = np.hstack((frame_l, frame_r))
                success, encoded_jpg = cv2.imencode(".jpg", composite, encode_param)
                if not success:
                    continue

                payload = encoded_jpg.tobytes()
                # 8-byte big-endian header with payload length
                header = struct.pack(">Q", len(payload))
                client_socket.sendall(header + payload)

                frame_count += 1
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    stream_fps = frame_count / (now - fps_timer)
                    kb_per_frame = len(payload) / 1024.0
                    sys.stdout.write(
                        f"\r[Streaming] {stream_fps:.1f} FPS | Frame size: {kb_per_frame:.1f} KB | Rate: {stream_fps * kb_per_frame / 1024.0:.2f} MB/s"
                    )
                    sys.stdout.flush()
                    frame_count = 0
                    fps_timer = now

                # Rate limiting
                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        except (ConnectionRefusedError, BrokenPipeError, ConnectionResetError, socket.timeout) as e:
            print(f"\n[LaptopSender] Connection lost ({e}). Retrying in 2 seconds...")
            time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n[LaptopSender] Stopped by user.")
            break
        finally:
            try:
                client_socket.close()
            except Exception:
                pass

    cam_l.release()
    cam_r.release()
    print("[LaptopSender] Cameras released.")


if __name__ == "__main__":
    main()

