#!/usr/bin/env python3
"""Laptop Camera Sender for Binocular Stereo Perception (Stage 1).

Run this script LOCALLY on your laptop where the USB binocular cameras are plugged in.
It captures synchronized frames from both cameras, compresses them using tuned JPEG,
and streams them asynchronously to the GPU server running Stage1/harness.py.

Default protocol is HTTP Push directly to port 8080 (the exact same port as the Web UI).
This requires ZERO extra port configuration in VS Code!

Usage (Recommended - uses port 8080 which is already forwarded):
    python laptop_camera_sender.py --server 127.0.0.1 --port 8080 --left-cam 0 --right-cam 2

High-speed preset for bandwidth-constrained links (~30 KB / frame, 20-30 FPS):
    python laptop_camera_sender.py --stream-scale 0.35 --jpeg-quality 60

Or with TCP socket (port 5555):
    python laptop_camera_sender.py --protocol tcp --server 127.0.0.1 --port 5555 --left-cam 0 --right-cam 2
"""

from __future__ import annotations
import argparse
import http.client
import platform
import queue
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
        help="Downscale factor before transmitting: 0.5 (960x540 per eye, ~55 KB), 0.35 (672x378 per eye, ~30 KB for low-bandwidth links), 1.0 (1080p). Default: 0.5.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=60,
        help="JPEG compression quality (1-100, default: 60). Quality 60 cuts frame size by ~55%% with <0.35px disparity error.",
    )
    parser.add_argument(
        "--http-conns",
        type=int,
        default=2,
        help="Number of concurrent persistent HTTP connections for pipelined streaming (default: 2). Overlaps network RTT latency.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Network request timeout in seconds (default: 5.0).",
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

    # In DirectShow, resolution must be set FIRST, then FOURCC=MJPG, then FPS=30.
    # Setting FOURCC before resolution causes DirectShow to reset to uncompressed YUY2!
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30.0)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc_chars = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(
        f"[LaptopSender] Camera {idx} initialized: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS (FOURCC: '{fcc_chars}')"
    )
    if actual_fps < 10.0 or fcc_chars.upper() == "YUY2":
        print(
            f"[LaptopSender] ⚠️ Camera {idx} negotiated low FPS ({actual_fps:.1f}) or uncompressed YUY2. "
            f"If USB bandwidth is limited, try passing '--width 1280 --height 720'."
        )
    return cap


class CameraGrabber:
    """Threaded camera grabber that decouples hardware acquisition from network transmission."""

    def __init__(self, cam_l: cv2.VideoCapture, cam_r: cv2.VideoCapture) -> None:
        self.cam_l = cam_l
        self.cam_r = cam_r
        self._latest_pair: tuple[np.ndarray, np.ndarray] | None = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = True
        self.grab_count = 0
        self.grab_fps = 0.0
        self._fps_timer = time.perf_counter()
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
                    self._frame_id += 1
                    self.grab_count += 1
                    self._latest_pair = (frame_l, frame_r)
                    now = time.perf_counter()
                    dt = now - self._fps_timer
                    if dt >= 1.0:
                        self.grab_fps = self.grab_count / dt
                        self.grab_count = 0
                        self._fps_timer = now
            else:
                time.sleep(0.005)

    def get_latest_pair(self) -> tuple[np.ndarray, np.ndarray, int] | None:
        with self._lock:
            if self._latest_pair is None:
                return None
            return self._latest_pair[0].copy(), self._latest_pair[1].copy(), self._frame_id

    def get_grab_fps(self) -> float:
        with self._lock:
            return self.grab_fps

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


class HttpSendWorkerPool:
    """Asynchronous multi-connection HTTP worker pool.
    
    Uses multiple persistent HTTP connections to overlap network Round-Trip Time (RTT).
    While connection A is waiting for HTTP 200 response, connection B is already transmitting
    the next frame, breaking the stop-and-wait latency ceiling.
    """

    def __init__(self, server: str, port: int, num_conns: int = 2, timeout: float = 5.0) -> None:
        self.server = server
        self.port = port
        self.num_conns = max(1, num_conns)
        self.timeout = timeout
        self.queue: queue.Queue[tuple[bytes, float, int]] = queue.Queue(maxsize=self.num_conns)
        self.running = True
        self.sent_count = 0
        self.sent_bytes = 0
        self.current_fps = 0.0
        self.last_payload_kb = 0.0
        self._rtt_accum = 0.0
        self._rtt_samples = 0
        self._lock = threading.Lock()
        self.threads = [
            threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            for i in range(self.num_conns)
        ]
        for t in self.threads:
            t.start()

    def submit_frame(self, payload: bytes, timestamp: float, seq: int) -> None:
        try:
            self.queue.put_nowait((payload, timestamp, seq))
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((payload, timestamp, seq))
            except queue.Full:
                pass

    def _worker_loop(self, worker_id: int) -> None:
        headers = {
            "Content-Type": "image/jpeg",
            "Connection": "keep-alive",
        }
        conn: http.client.HTTPConnection | None = None
        while self.running:
            try:
                payload, timestamp, seq = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if conn is None:
                    conn = http.client.HTTPConnection(self.server, self.port, timeout=self.timeout)
                    conn.connect()

                req_headers = dict(headers)
                req_headers["X-Capture-Timestamp"] = f"{timestamp:.4f}"
                with self._lock:
                    fps_val = self.current_fps
                req_headers["X-Sender-FPS"] = f"{fps_val:.1f}"
                req_headers["X-Frame-Sequence"] = str(seq)

                t0 = time.perf_counter()
                conn.request("POST", "/api/push_frame", body=payload, headers=req_headers)
                res = conn.getresponse()
                res.read()
                rtt = (time.perf_counter() - t0) * 1000.0

                with self._lock:
                    self.sent_count += 1
                    self.sent_bytes += len(payload)
                    self.last_payload_kb = len(payload) / 1024.0
                    self._rtt_accum += rtt
                    self._rtt_samples += 1
            except (ConnectionRefusedError, BrokenPipeError, ConnectionResetError, http.client.HTTPException, socket.timeout, OSError):
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                time.sleep(1.0)

        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def get_and_reset_stats(self, dt: float) -> tuple[float, float, float, float]:
        with self._lock:
            self.current_fps = self.sent_count / dt if dt > 0 else 0.0
            rate_mb = (self.sent_bytes / (1024.0 * 1024.0)) / dt if dt > 0 else 0.0
            avg_rtt = (self._rtt_accum / self._rtt_samples) if self._rtt_samples > 0 else 0.0
            kb = self.last_payload_kb
            self.sent_count = 0
            self.sent_bytes = 0
            self._rtt_accum = 0.0
            self._rtt_samples = 0
            return self.current_fps, rate_mb, avg_rtt, kb

    def stop(self) -> None:
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)


class TcpSendWorker:
    """Asynchronous background worker that transmits length-prefixed frames over TCP."""

    def __init__(self, server: str, port: int, timeout: float = 5.0) -> None:
        self.server = server
        self.port = port
        self.timeout = timeout
        self.queue: queue.Queue[tuple[bytes, float, int]] = queue.Queue(maxsize=1)
        self.running = True
        self.sent_count = 0
        self.sent_bytes = 0
        self.current_fps = 0.0
        self.last_payload_kb = 0.0
        self._socket: socket.socket | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit_frame(self, payload: bytes, timestamp: float, seq: int) -> None:
        try:
            self.queue.put_nowait((payload, timestamp, seq))
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((payload, timestamp, seq))
            except queue.Full:
                pass

    def _get_socket(self) -> socket.socket:
        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.server, self.port))
            print(f"[LaptopSender] TCP socket connected to {self.server}:{self.port}")
        return self._socket

    def _close_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _run(self) -> None:
        while self.running:
            try:
                payload, timestamp, seq = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                s = self._get_socket()
                header = struct.pack(">Q", len(payload))
                s.sendall(header + payload)

                self.sent_count += 1
                self.sent_bytes += len(payload)
                self.last_payload_kb = len(payload) / 1024.0
            except (ConnectionRefusedError, BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                self._close_socket()
                time.sleep(1.0)

    def stop(self) -> None:
        self.running = False
        self._close_socket()
        self.thread.join(timeout=1.0)


def stream_http(args: argparse.Namespace, grabber: CameraGrabber) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0
    scale = max(0.1, min(1.0, float(args.stream_scale)))
    num_conns = max(1, int(args.http_conns))

    target_w_eye = int(round(args.width * scale))
    target_h_eye = int(round(args.height * scale))
    print(
        f"[LaptopSender] Configuration: Scale {scale:.2f}x ({target_w_eye}x{target_h_eye} per eye, "
        f"{target_w_eye * 2}x{target_h_eye} composite) | JPEG Quality: {args.jpeg_quality} | HTTP Conns: {num_conns}"
    )
    print(f"[LaptopSender] Starting pipelined HTTP sender to http://{args.server}:{args.port}...")

    worker_pool = HttpSendWorkerPool(args.server, args.port, num_conns=num_conns, timeout=args.timeout)
    last_seq = -1
    fps_timer = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            pair_data = grabber.get_latest_pair()
            if pair_data is None:
                time.sleep(0.003)
                continue

            frame_l, frame_r, seq = pair_data
            if seq == last_seq:
                time.sleep(0.003)
                continue
            last_seq = seq

            # Combine side-by-side: [Left | Right]
            composite = np.hstack((frame_l, frame_r))

            if scale < 0.999:
                ch, cw = composite.shape[:2]
                target_w = int(round(cw * scale))
                target_h = int(round(ch * scale))
                composite = cv2.resize(composite, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            success, encoded_jpg = cv2.imencode(".jpg", composite, encode_param)
            if not success:
                continue

            payload = encoded_jpg.tobytes()
            worker_pool.submit_frame(payload, time.time(), seq)

            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                dt = now - fps_timer
                net_fps, rate_mb, avg_rtt, kb = worker_pool.get_and_reset_stats(dt)
                cam_fps = grabber.get_grab_fps()
                rtt_str = f"{avg_rtt:.0f} ms" if avg_rtt > 0 else "-- ms"
                sys.stdout.write(
                    f"\r[Stream] Camera: {cam_fps:.1f} FPS | Network: {net_fps:.1f} FPS | RTT: {rtt_str} | Frame: {kb:.1f} KB | Rate: {rate_mb:.2f} MB/s"
                )
                sys.stdout.flush()
                fps_timer = now

            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        print("\n[LaptopSender] Stopped by user.")
    finally:
        worker_pool.stop()


def stream_tcp(args: argparse.Namespace, grabber: CameraGrabber) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0
    scale = max(0.1, min(1.0, float(args.stream_scale)))

    target_w_eye = int(round(args.width * scale))
    target_h_eye = int(round(args.height * scale))
    print(
        f"[LaptopSender] Configuration: Scale {scale:.2f}x ({target_w_eye}x{target_h_eye} per eye, "
        f"{target_w_eye * 2}x{target_h_eye} composite) | JPEG Quality: {args.jpeg_quality}"
    )
    print(f"[LaptopSender] Starting asynchronous TCP sender to {args.server}:{args.port}...")

    worker = TcpSendWorker(args.server, args.port, timeout=args.timeout)
    last_seq = -1
    fps_timer = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            pair_data = grabber.get_latest_pair()
            if pair_data is None:
                time.sleep(0.003)
                continue

            frame_l, frame_r, seq = pair_data
            if seq == last_seq:
                time.sleep(0.003)
                continue
            last_seq = seq

            composite = np.hstack((frame_l, frame_r))
            if scale < 0.999:
                ch, cw = composite.shape[:2]
                target_w = int(round(cw * scale))
                target_h = int(round(ch * scale))
                composite = cv2.resize(composite, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            success, encoded_jpg = cv2.imencode(".jpg", composite, encode_param)
            if not success:
                continue

            payload = encoded_jpg.tobytes()
            worker.submit_frame(payload, time.time(), seq)

            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                dt = now - fps_timer
                net_fps = worker.sent_count / dt
                rate_mb = (worker.sent_bytes / (1024.0 * 1024.0)) / dt
                cam_fps = grabber.get_grab_fps()
                sys.stdout.write(
                    f"\r[Stream] Camera: {cam_fps:.1f} FPS | Network: {net_fps:.1f} FPS | Frame: {worker.last_payload_kb:.1f} KB | Rate: {rate_mb:.2f} MB/s"
                )
                sys.stdout.flush()
                worker.sent_count = 0
                worker.sent_bytes = 0
                fps_timer = now

            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        print("\n[LaptopSender] Stopped by user.")
    finally:
        worker.stop()


def main() -> None:
    args = parse_args()
    backend = get_backend()
    print(f"[LaptopSender] OS: {platform.system()} | Camera backend: {backend}")
    print(
        f"[LaptopSender] Opening Left Camera (ID {args.left_cam}) and Right Camera (ID {args.right_cam})..."
    )

    cam_l = open_camera(args.left_cam, args.width, args.height, backend)
    cam_r = open_camera(args.right_cam, args.width, args.height, backend)

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
