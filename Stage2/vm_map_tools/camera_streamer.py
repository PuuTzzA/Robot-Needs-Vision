#!/usr/bin/env python3
"""Stage 2: Background Camera Streamer for Ubuntu / VMware.

Supports:
  1. Direct USB capture via V4L2 (/dev/video*).
  2. Network HTTP Push receiver (/api/push_frame on port 8080) for streaming
     from a Windows host via laptop_camera_sender.py (avoids VMware USB packet drops).
  3. Synthetic test pattern fallback for headless/offline environments.
"""

from __future__ import annotations

import glob
import http.server
import os
import platform
import socketserver
import threading
import time
import cv2
import numpy as np


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


class HttpFrameReceiver(http.server.BaseHTTPRequestHandler):
    """Handles POST /api/push_frame from laptop_camera_sender.py."""

    def do_POST(self):
        if self.path == "/api/push_frame":
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                payload = self.rfile.read(length)
                np_arr = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.server.streamer._on_network_frame(frame)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"success": true}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class ThreadedHttpServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class CameraStreamer:
    """Threaded camera grabber supporting direct V4L2 USB and network HTTP streaming."""

    def __init__(
        self,
        camera_indices: list[int] | None = None,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        allow_mock: bool = True,
        http_port: int | None = 8080,
    ) -> None:
        self.requested_indices = camera_indices
        self.width = width
        self.height = height
        self.fps = fps
        self.allow_mock = allow_mock
        self.http_port = http_port

        self.caps: dict[int, cv2.VideoCapture] = {}
        self.active_indices: list[int] = []
        self._latest_frame: np.ndarray | None = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.is_mock = False
        self.is_network = False
        self.grab_fps = 0.0
        self._fps_count = 0
        self._fps_timer = time.perf_counter()
        self._http_server: ThreadedHttpServer | None = None
        self._http_thread: threading.Thread | None = None

        if self.http_port:
            self._start_http_receiver()

        self._initialize_hardware()

    def _start_http_receiver(self) -> None:
        try:
            server = ThreadedHttpServer(("0.0.0.0", self.http_port), HttpFrameReceiver)
            server.streamer = self
            self._http_server = server
            self._http_thread = threading.Thread(target=server.serve_forever, daemon=True)
            self._http_thread.start()
            print(f"[CameraStreamer] Network HTTP frame receiver listening on port {self.http_port} (/api/push_frame).")
        except Exception as e:
            print(f"[CameraStreamer] [INFO] HTTP port {self.http_port} not bound ({e}). Direct USB only.")

    def _on_network_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest_frame = frame
            self._frame_id += 1
            self._fps_count += 1
            self.is_network = True
            self.is_mock = False

    def _initialize_hardware(self) -> None:
        indices = self.requested_indices
        if indices is None:
            indices = self._discover_indices()

        if not indices:
            if self.http_port:
                print(f"[CameraStreamer] No direct USB camera found. Waiting for network stream or mock...")
                self.is_mock = True
                return
            elif self.allow_mock:
                print("[CameraStreamer] [WARN] No physical camera found in /dev/video*. Using synthetic test pattern feed.")
                self.is_mock = True
                return

        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY

        for idx in indices:
            dev_path = f"/dev/video{idx}"
            if platform.system() == "Linux" and not os.path.exists(dev_path):
                continue

            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                cap = cv2.VideoCapture(idx)

            if not cap.isOpened():
                continue

            # Set resolution first, then MJPG to avoid buffer negotiation glitches
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Warm-up check
            ok = False
            for _ in range(5):
                ret, f = cap.read()
                if ret and f is not None and f.size > 0:
                    ok = True
                    break
                time.sleep(0.04)

            if not ok:
                cap.release()
                cap = cv2.VideoCapture(idx)
                for _ in range(5):
                    ret, f = cap.read()
                    if ret and f is not None and f.size > 0:
                        ok = True
                        break
                    time.sleep(0.04)

            if ok and cap.isOpened():
                self.caps[idx] = cap
                self.active_indices.append(idx)
                mode_str = "Single-camera" if len(self.active_indices) == 1 else "Stereo"
                print(f"[CameraStreamer] Camera device {idx} initialized ({mode_str} @ {self.width}x{self.height}).")

        if not self.caps and not self.http_port:
            if self.allow_mock:
                self.is_mock = True

    @staticmethod
    def _discover_indices() -> list[int]:
        """Find video devices that are actually capable of frame capture using MJPG."""
        devs = sorted(glob.glob("/dev/video*"))
        if not devs:
            return []

        indices: list[int] = []
        with SuppressStderr():
            for dev in devs:
                try:
                    idx = int(os.path.basename(dev).replace("video", ""))
                except ValueError:
                    continue

                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        indices.append(idx)
                    cap.release()
        return indices

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _generate_mock_frame(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = 40
        frame[:, :, 1] = 45
        frame[:, :, 2] = 50

        t = time.time()
        cx = int((self.width / 2) + np.sin(t * 2.0) * (self.width / 3))
        cy = int((self.height / 2) + np.cos(t * 2.0) * (self.height / 4))
        cv2.circle(frame, (cx, cy), 40, (0, 200, 255), -1)
        cv2.circle(frame, (cx, cy), 15, (255, 255, 255), -1)

        cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (80, 80, 80), 1)
        cv2.line(frame, (0, self.height // 2), (self.width, self.height // 2), (80, 80, 80), 1)

        cv2.putText(
            frame,
            "[MOCK / WAITING] Camera Streamer Active",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        msg = f"Ready on port {self.http_port} (HTTP Push) or /dev/video0" if self.http_port else "Ready"
        cv2.putText(
            frame,
            msg,
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / self.fps if self.fps > 0 else 0.033
        while self._running:
            start_t = time.perf_counter()

            if self.is_network:
                # Network frames are populated asynchronously by HttpFrameReceiver
                time.sleep(0.01)
            elif self.is_mock:
                frame = self._generate_mock_frame()
                with self._lock:
                    self._latest_frame = frame
                    self._frame_id += 1
                    self._fps_count += 1
            else:
                frames: list[np.ndarray] = []
                for idx in self.active_indices:
                    cap = self.caps[idx]
                    ret, f = cap.read()
                    if ret and f is not None:
                        frames.append(f)

                if frames:
                    composite = np.hstack(frames) if len(frames) > 1 else frames[0]
                    with self._lock:
                        self._latest_frame = composite
                        self._frame_id += 1
                        self._fps_count += 1

            now = time.perf_counter()
            dt = now - self._fps_timer
            if dt >= 1.0:
                with self._lock:
                    self.grab_fps = self._fps_count / dt
                    self._fps_count = 0
                    self._fps_timer = now

            elapsed = time.perf_counter() - start_t
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def get_latest_frame(self) -> tuple[np.ndarray | None, int, float]:
        """Returns (latest_frame, frame_id, fps)."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0, 0.0
            return self._latest_frame.copy(), self._frame_id, self.grab_fps

    def save_snapshot(self, path: str) -> bool:
        """Saves current frame to the specified image path."""
        frame, _, _ = self.get_latest_frame()
        if frame is None:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return cv2.imwrite(path, frame)

    def stop(self) -> None:
        self._running = False
        if self._http_server:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()
        self.active_indices.clear()
