"""Camera capture module for Windows Mission Control.

Supports:
- Stereo USB webcams (left + right device indices)
- Single USB webcam
- Synthetic Mock Camera (offline simulation)
"""

from __future__ import annotations

import math
import platform
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


def get_backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def open_camera(
    idx: int,
    width: int = 1280,
    height: int = 720,
    backend: Optional[int] = None,
    retries: int = 4,
    retry_delay: float = 0.4,
) -> cv2.VideoCapture:
    """Open camera with retries and format negotiation."""
    if backend is None:
        backend = get_backend()

    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY and platform.system() != "Windows":
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 30.0)

            # Warmup
            warmup_ok = False
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    warmup_ok = True
                    break
                time.sleep(0.04)

            if warmup_ok:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
                print(f"[Camera] Device {idx} initialized ({actual_w}x{actual_h} @ {actual_fps:.1f} FPS).")
                return cap
            cap.release()
        time.sleep(retry_delay)

    raise RuntimeError(f"Could not open camera device {idx} after {retries} attempts.")


class CameraManager:
    """Threaded camera capture supporting stereo pair, single camera, or mock frames."""

    def __init__(
        self,
        left_cam: Optional[int] = 0,
        right_cam: Optional[int] = 2,
        single_cam: Optional[int] = None,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        is_mock: bool = False,
    ) -> None:
        self.left_idx = left_cam
        self.right_idx = right_cam
        self.single_idx = single_cam
        self.width = width
        self.height = height
        self.fps = fps
        self.is_mock = is_mock

        self.cap_l: Optional[cv2.VideoCapture] = None
        self.cap_r: Optional[cv2.VideoCapture] = None

        self._latest_left: Optional[np.ndarray] = None
        self._latest_right: Optional[np.ndarray] = None
        self._frame_id = 0
        self._last_frame_at = 0.0
        self._fps = 0.0
        self._fps_count = 0
        self._fps_timer = time.perf_counter()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return

        if not self.is_mock:
            try:
                backend = get_backend()
                if self.single_idx is not None:
                    print(f"[Camera] Initializing single camera (ID {self.single_idx})...")
                    self.cap_l = open_camera(self.single_idx, self.width, self.height, backend)
                    self.cap_r = None
                else:
                    print(f"[Camera] Initializing stereo pair (Left {self.left_idx}, Right {self.right_idx})...")
                    self.cap_l = open_camera(self.left_idx, self.width, self.height, backend)
                    time.sleep(0.3)
                    self.cap_r = open_camera(self.right_idx, self.width, self.height, backend)
            except Exception as err:
                print(f"[Camera] [WARN] Physical camera initialization failed: {err}")
                print("[Camera] Falling back to Mock Camera feed.")
                self.is_mock = True
                if self.cap_l:
                    self.cap_l.release()
                    self.cap_l = None
                if self.cap_r:
                    self.cap_r.release()
                    self.cap_r = None

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _generate_mock_frame(self, side: str) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = 30
        frame[:, :, 1] = 35
        frame[:, :, 2] = 40

        t = time.time()
        cx = int((self.width / 2) + np.sin(t * 1.5) * (self.width / 4))
        cy = int((self.height / 2) + np.cos(t * 1.5) * (self.height / 5))

        # Stereo parallax offset
        offset_x = -15 if side == "left" else 15
        cx_eff = max(30, min(self.width - 30, cx + offset_x))

        cv2.circle(frame, (cx_eff, cy), 50, (0, 180, 255), -1)
        cv2.circle(frame, (cx_eff, cy), 18, (255, 255, 255), -1)

        cv2.putText(
            frame,
            f"[MOCK CAMERA] {side.upper()} VIEW",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            time.strftime("%H:%M:%S"),
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _capture_loop(self) -> None:
        interval = 1.0 / self.fps if self.fps > 0 else 0.033

        while self._running:
            t0 = time.perf_counter()

            if self.is_mock:
                left_f = self._generate_mock_frame("left")
                right_f = self._generate_mock_frame("right")
            elif self.cap_r is not None and self.cap_l is not None:
                # Synchronized grab
                self.cap_l.grab()
                self.cap_r.grab()
                ret_l, left_f = self.cap_l.retrieve()
                ret_r, right_f = self.cap_r.retrieve()
                if not (ret_l and ret_r and left_f is not None and right_f is not None):
                    time.sleep(0.005)
                    continue
            elif self.cap_l is not None:
                ret, left_f = self.cap_l.read()
                right_f = left_f.copy() if (ret and left_f is not None) else None
                if not ret or left_f is None:
                    time.sleep(0.005)
                    continue
            else:
                left_f = self._generate_mock_frame("left")
                right_f = self._generate_mock_frame("right")

            now = time.perf_counter()
            with self._lock:
                self._latest_left = left_f
                self._latest_right = right_f
                self._frame_id += 1
                self._last_frame_at = time.monotonic()
                self._fps_count += 1
                dt = now - self._fps_timer
                if dt >= 1.0:
                    self._fps = self._fps_count / dt
                    self._fps_count = 0
                    self._fps_timer = now

            elapsed = time.perf_counter() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def get_latest_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Return (left_frame, right_frame, fps)."""
        with self._lock:
            l = self._latest_left.copy() if self._latest_left is not None else None
            r = self._latest_right.copy() if self._latest_right is not None else None
            return l, r, self._fps

    def snapshot_left_frame(self, max_age_s: float = 2.0) -> Optional[np.ndarray]:
        """Copy a recent single/left frame for manual analysis; reject stopped/stale feeds."""
        with self._lock:
            if (not self._running or self._latest_left is None
                    or time.monotonic() - self._last_frame_at > max_age_s):
                return None
            return self._latest_left.copy()

    def snapshot_left_jpeg(self, quality: int = 85) -> Optional[bytes]:
        """Return JPEG bytes of current left frame."""
        with self._lock:
            if self._latest_left is None:
                return None
            frame = self._latest_left.copy()

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def snapshot_stereo_composite_png(self) -> Optional[bytes]:
        """Return lossless PNG bytes of side-by-side composite [Left | Right] for size server."""
        with self._lock:
            if self._latest_left is None or self._latest_right is None:
                return None
            composite = np.hstack((self._latest_left, self._latest_right))

        ok, buf = cv2.imencode(".png", composite)
        return buf.tobytes() if ok else None

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap_l:
            self.cap_l.release()
            self.cap_l = None
        if self.cap_r:
            self.cap_r.release()
            self.cap_r = None

