"""Camera capture module for Windows Mission Control.

Supports:
- Stereo USB webcams (left + right device indices) with decoupled capture threads
- Automatic freeze/dropout & DirectShow zombie-buffer watchdog + background reopen
- Post-settle fresh-frame barriers so shaking/stale frames are never captured
- Single USB webcam
- Synthetic Mock Camera (offline simulation)
"""

from __future__ import annotations

from dataclasses import dataclass
import platform
import threading
import time
from typing import Any, Optional, Tuple
import cv2
import numpy as np


_OPEN_LOCK = threading.Lock()


def get_backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def _release_capture_async(cap: Optional[cv2.VideoCapture]) -> None:
    """Release an OpenCV VideoCapture on a daemon thread so hung DirectShow drivers cannot block."""
    if cap is None:
        return

    def _closer() -> None:
        try:
            cap.release()
        except Exception:
            pass

    threading.Thread(target=_closer, daemon=True).start()


def open_camera(
    idx: int,
    width: int = 1280,
    height: int = 720,
    backend: Optional[int] = None,
    retries: int = 4,
    retry_delay: float = 0.4,
    fps: float = 15.0,
) -> cv2.VideoCapture:
    """Open camera with retries, MJPG bandwidth negotiation, and buffer minimization."""
    if backend is None:
        backend = get_backend()

    mjpg_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    for _attempt in range(1, retries + 1):
        with _OPEN_LOCK:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY and platform.system() != "Windows":
                cap = cv2.VideoCapture(idx)

            if cap.isOpened():
                # Minimize internal driver queue so reads always reflect recent exposures
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)

                # Negotiate MJPG both before and after resolution so DirectShow avoids YUY2 fallback
                cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
                if fps > 0:
                    cap.set(cv2.CAP_PROP_FPS, float(fps))

                # Warmup & flush initial buffered frames
                valid_reads = 0
                for _ in range(6):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        valid_reads += 1
                        if valid_reads >= 2:
                            break
                    time.sleep(0.03)

                if valid_reads >= 1:
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
                    print(
                        f"[Camera] Device {idx} initialized ({actual_w}x{actual_h} @ {actual_fps:.1f} FPS, target {fps:.1f} FPS)."
                    )
                    return cap
                cap.release()

        time.sleep(retry_delay)

    raise RuntimeError(f"Could not open camera device {idx} after {retries} attempts.")


@dataclass
class _SideTelemetry:
    """Per-camera stream telemetry and watchdog state."""
    side: str
    device_idx: Optional[int]
    seq: int = 0
    last_good_mono: float = 0.0
    last_unique_mono: float = 0.0
    last_sig: Optional[np.ndarray] = None
    consecutive_failures: int = 0
    fps: float = 0.0
    fps_count: int = 0
    fps_timer: float = 0.0
    state: str = "init"  # init | mock | ok | reconnecting | stalled
    worker_gen: int = 0
    last_reopen_mono: float = 0.0
    reopen_count: int = 0


def _frame_signature(frame: np.ndarray) -> np.ndarray:
    """Fast subsampled byte signature used to detect frozen DirectShow sample buffers."""
    step_y = max(1, frame.shape[0] // 36)
    step_x = max(1, frame.shape[1] // 64)
    return frame[::step_y, ::step_x].copy()


class CameraManager:
    """Threaded camera capture supporting stereo pair, single camera, or mock frames."""

    STALL_TIMEOUT_S = 1.6
    ZOMBIE_DUP_TIMEOUT_S = 2.5
    REOPEN_COOLDOWN_S = 3.5
    MAX_CONSECUTIVE_FAILURES = 8

    def __init__(
        self,
        left_cam: Optional[int] = 0,
        right_cam: Optional[int] = 2,
        single_cam: Optional[int] = None,
        width: int = 1280,
        height: int = 720,
        fps: float = 15.0,
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
        self._frame_cond = threading.Condition(self._lock)
        self._reopen_guard = threading.Lock()
        self._running = False

        # Threads
        self._thread: Optional[threading.Thread] = None  # Mock loop thread
        self._left_thread: Optional[threading.Thread] = None
        self._right_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

        eff_left_idx = single_cam if single_cam is not None else left_cam
        eff_right_idx = None if single_cam is not None else right_cam
        self._left_tel = _SideTelemetry(side="left", device_idx=eff_left_idx)
        self._right_tel = _SideTelemetry(side="right", device_idx=eff_right_idx)

    def start(self) -> None:
        if self._running:
            return

        eff_left_idx = self.single_idx if self.single_idx is not None else self.left_idx
        eff_right_idx = None if self.single_idx is not None else self.right_idx
        self._left_tel = _SideTelemetry(side="left", device_idx=eff_left_idx)
        self._right_tel = _SideTelemetry(side="right", device_idx=eff_right_idx)

        if not self.is_mock:
            try:
                backend = get_backend()
                if self.single_idx is not None:
                    print(f"[Camera] Initializing single camera (ID {self.single_idx})...")
                    self.cap_l = open_camera(
                        self.single_idx, self.width, self.height, backend, fps=self.fps
                    )
                    self.cap_r = None
                else:
                    print(f"[Camera] Initializing stereo pair (Left {self.left_idx}, Right {self.right_idx})...")
                    self.cap_l = open_camera(
                        int(self.left_idx or 0), self.width, self.height, backend, fps=self.fps
                    )
                    time.sleep(0.35)
                    self.cap_r = open_camera(
                        int(self.right_idx if self.right_idx is not None else 2),
                        self.width,
                        self.height,
                        backend,
                        fps=self.fps,
                    )
            except Exception as err:
                print(f"[Camera] [WARN] Physical camera initialization failed: {err}")
                print("[Camera] Falling back to Mock Camera feed.")
                self.is_mock = True
                _release_capture_async(self.cap_l)
                _release_capture_async(self.cap_r)
                self.cap_l = None
                self.cap_r = None

        self._running = True
        now_mono = time.monotonic()
        now_perf = time.perf_counter()

        if self.is_mock:
            self._left_tel.state = "mock"
            self._right_tel.state = "mock"
            self._thread = threading.Thread(target=self._mock_capture_loop, daemon=True)
            self._thread.start()
        else:
            self._left_tel.last_good_mono = now_mono
            self._left_tel.last_unique_mono = now_mono
            self._left_tel.fps_timer = now_perf
            self._left_tel.state = "ok"
            self._spawn_side_worker("left")

            if self.single_idx is None and self._right_tel.device_idx is not None:
                self._right_tel.last_good_mono = now_mono
                self._right_tel.last_unique_mono = now_mono
                self._right_tel.fps_timer = now_perf
                self._right_tel.state = "ok"
                self._spawn_side_worker("right")
            else:
                self._right_tel.state = "ok"

            self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._watchdog_thread.start()

    def _spawn_side_worker(self, side: str) -> None:
        with self._lock:
            tel = self._left_tel if side == "left" else self._right_tel
            tel.worker_gen += 1
            gen = tel.worker_gen
            cap = self.cap_l if side == "left" else self.cap_r

        t = threading.Thread(
            target=self._physical_side_loop,
            args=(side, gen, cap),
            daemon=True,
            name=f"CameraWorker-{side}-{gen}",
        )
        if side == "left":
            self._left_thread = t
        else:
            self._right_thread = t
        t.start()

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

    def _mock_capture_loop(self) -> None:
        interval = 1.0 / self.fps if self.fps > 0 else 0.066

        while self._running:
            t0 = time.perf_counter()
            left_f = self._generate_mock_frame("left")
            right_f = self._generate_mock_frame("right")

            now_perf = time.perf_counter()
            now_mono = time.monotonic()
            with self._frame_cond:
                self._latest_left = left_f
                self._latest_right = right_f
                self._frame_id += 1
                self._last_frame_at = now_mono
                self._left_tel.seq += 1
                self._right_tel.seq += 1
                self._left_tel.last_good_mono = now_mono
                self._right_tel.last_good_mono = now_mono
                self._fps_count += 1
                dt = now_perf - self._fps_timer
                if dt >= 1.0:
                    self._fps = self._fps_count / dt
                    self._left_tel.fps = self._fps
                    self._right_tel.fps = self._fps
                    self._fps_count = 0
                    self._fps_timer = now_perf
                self._frame_cond.notify_all()

            elapsed = time.perf_counter() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _physical_side_loop(
        self,
        side: str,
        gen: int,
        cap: Optional[cv2.VideoCapture],
    ) -> None:
        interval = 1.0 / self.fps if self.fps > 0 else 0.066

        while self._running:
            with self._lock:
                tel = self._left_tel if side == "left" else self._right_tel
                if tel.worker_gen != gen:
                    return

            if cap is None or not cap.isOpened():
                with self._lock:
                    tel = self._left_tel if side == "left" else self._right_tel
                    if tel.worker_gen == gen:
                        tel.consecutive_failures += 1
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            try:
                ret, frame = cap.read()
            except Exception:
                ret, frame = False, None

            if not (ret and frame is not None and frame.size > 0):
                with self._lock:
                    tel = self._left_tel if side == "left" else self._right_tel
                    if tel.worker_gen != gen:
                        return
                    tel.consecutive_failures += 1
                time.sleep(0.015)
                continue

            sig = _frame_signature(frame)
            now_mono = time.monotonic()
            now_perf = time.perf_counter()

            with self._frame_cond:
                tel = self._left_tel if side == "left" else self._right_tel
                if tel.worker_gen != gen or not self._running:
                    return

                tel.consecutive_failures = 0
                tel.last_good_mono = now_mono
                tel.state = "ok"

                # Detect whether frame content updated (ignore solid uniform test/lens-cap frames)
                if tel.last_sig is None or int(sig.min()) == int(sig.max()) or not np.array_equal(sig, tel.last_sig):
                    tel.last_sig = sig
                    tel.last_unique_mono = now_mono

                tel.seq += 1
                tel.fps_count += 1
                dt_side = now_perf - tel.fps_timer
                if dt_side >= 1.0:
                    tel.fps = tel.fps_count / dt_side
                    tel.fps_count = 0
                    tel.fps_timer = now_perf

                if side == "left":
                    self._latest_left = frame
                    self._frame_id += 1
                    self._last_frame_at = now_mono
                    if self.single_idx is not None:
                        self._latest_right = frame.copy()
                        self._right_tel.seq += 1
                        self._right_tel.last_good_mono = now_mono
                        self._right_tel.last_unique_mono = now_mono
                        self._right_tel.fps = tel.fps
                        self._right_tel.state = "ok"
                else:
                    self._latest_right = frame

                self._fps = self._left_tel.fps
                self._frame_cond.notify_all()

            elapsed = time.perf_counter() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _watchdog_loop(self) -> None:
        """Monitors physical camera streams and automatically reopens frozen/dropped USB devices."""
        while self._running and not self.is_mock:
            time.sleep(0.3)
            if not self._running or self.is_mock:
                break

            sides = ["left"]
            if self.single_idx is None and self._right_tel.device_idx is not None:
                sides.append("right")

            for side in sides:
                reason = self._check_side_stall_reason(side)
                if reason:
                    self._reopen_side(side, reason=reason, force=False)

    def _check_side_stall_reason(self, side: str) -> Optional[str]:
        now_mono = time.monotonic()
        with self._lock:
            tel = self._left_tel if side == "left" else self._right_tel
            if tel.device_idx is None or tel.state == "reconnecting":
                return None

            # Decay FPS to 0.0 immediately if stalled
            if tel.last_good_mono > 0 and (now_mono - tel.last_good_mono) > self.STALL_TIMEOUT_S:
                tel.fps = 0.0
                if side == "left":
                    self._fps = 0.0

            if (now_mono - tel.last_reopen_mono) < self.REOPEN_COOLDOWN_S:
                return None

            if tel.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                return f"{tel.consecutive_failures} consecutive read failures"

            if tel.last_good_mono > 0 and (now_mono - tel.last_good_mono) > self.STALL_TIMEOUT_S:
                age = now_mono - tel.last_good_mono
                return f"no frame for {age:.1f}s"

            if tel.last_unique_mono > 0 and (now_mono - tel.last_unique_mono) > self.ZOMBIE_DUP_TIMEOUT_S:
                dup_age = now_mono - tel.last_unique_mono
                return f"frozen duplicate buffer for {dup_age:.1f}s"

        return None

    def _reopen_side(self, side: str, reason: str, force: bool = False) -> bool:
        """Detach stalled worker, release old handle asynchronously, and open a fresh stream."""
        with self._reopen_guard:
            if not self._running or self.is_mock:
                return False

            now_mono = time.monotonic()
            with self._lock:
                tel = self._left_tel if side == "left" else self._right_tel
                if tel.device_idx is None:
                    return False
                if not force and (now_mono - tel.last_reopen_mono) < self.REOPEN_COOLDOWN_S:
                    return False

                idx = tel.device_idx
                tel.state = "reconnecting"
                tel.fps = 0.0
                if side == "left":
                    self._fps = 0.0
                tel.last_reopen_mono = now_mono
                tel.worker_gen += 1  # Immediately invalidate old worker thread

                old_cap = self.cap_l if side == "left" else self.cap_r
                if side == "left":
                    self.cap_l = None
                else:
                    self.cap_r = None

            print(f"[Camera] [WARN] {side.title()} camera (ID {idx}) stalled ({reason}). Reopening USB stream...")
            _release_capture_async(old_cap)
            time.sleep(0.25)

            try:
                new_cap = open_camera(
                    idx=idx,
                    width=self.width,
                    height=self.height,
                    backend=get_backend(),
                    retries=2,
                    retry_delay=0.3,
                    fps=self.fps,
                )
            except Exception as err:
                print(f"[Camera] [ERROR] Reopen failed for {side} camera (ID {idx}): {err}")
                with self._lock:
                    tel = self._left_tel if side == "left" else self._right_tel
                    tel.state = "stalled"
                return False

            reopened_mono = time.monotonic()
            reopened_perf = time.perf_counter()
            with self._lock:
                if not self._running:
                    _release_capture_async(new_cap)
                    return False
                tel = self._left_tel if side == "left" else self._right_tel
                if side == "left":
                    self.cap_l = new_cap
                else:
                    self.cap_r = new_cap
                tel.consecutive_failures = 0
                tel.last_good_mono = reopened_mono
                tel.last_unique_mono = reopened_mono
                tel.last_sig = None
                tel.fps_count = 0
                tel.fps_timer = reopened_perf
                tel.reopen_count += 1
                tel.state = "ok"

            self._spawn_side_worker(side)
            print(f"[Camera] [OK] {side.title()} camera (ID {idx}) recovered (reopen #{tel.reopen_count}).")
            return True

    def reset_cameras(self) -> str:
        """Manually force-reopen physical camera streams (used by 'CamReset' command or capture recovery)."""
        if self.is_mock:
            return "Camera is running in mock mode (no USB stream to reset)."
        if not self._running:
            return "CameraManager is stopped."

        results = []
        ok_l = self._reopen_side("left", reason="manual reset", force=True)
        results.append(f"Left={'OK' if ok_l else 'FAIL'}")

        if self.single_idx is None and self._right_tel.device_idx is not None:
            time.sleep(0.2)
            ok_r = self._reopen_side("right", reason="manual reset", force=True)
            results.append(f"Right={'OK' if ok_r else 'FAIL'}")

        status_joined = ", ".join(results)
        return f"Camera reset complete ({status_joined})"

    def get_camera_health(self) -> dict[str, Any]:
        """Return live per-camera FPS and health states for GUI status headers."""
        now_mono = time.monotonic()
        with self._lock:
            if self.is_mock:
                return {
                    "left_fps": self._fps,
                    "right_fps": self._fps,
                    "left_state": "mock",
                    "right_state": "mock",
                    "reopen_count": 0,
                }

            l_fps = (
                0.0
                if (self._left_tel.last_good_mono > 0 and (now_mono - self._left_tel.last_good_mono) > self.STALL_TIMEOUT_S)
                else self._left_tel.fps
            )
            r_fps = (
                l_fps
                if self.single_idx is not None
                else (
                    0.0
                    if (self._right_tel.last_good_mono > 0 and (now_mono - self._right_tel.last_good_mono) > self.STALL_TIMEOUT_S)
                    else self._right_tel.fps
                )
            )
            return {
                "left_fps": l_fps,
                "right_fps": r_fps,
                "left_state": self._left_tel.state,
                "right_state": self._left_tel.state if self.single_idx is not None else self._right_tel.state,
                "reopen_count": self._left_tel.reopen_count + self._right_tel.reopen_count,
            }

    def get_latest_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Return (left_frame, right_frame, fps)."""
        now_mono = time.monotonic()
        with self._lock:
            l = self._latest_left.copy() if self._latest_left is not None else None
            r = self._latest_right.copy() if self._latest_right is not None else None
            fps_val = self._fps
            if not self.is_mock and self._last_frame_at > 0 and (now_mono - self._last_frame_at) > self.STALL_TIMEOUT_S:
                fps_val = 0.0
            return l, r, fps_val

    def snapshot_left_frame(self, max_age_s: float = 2.0) -> Optional[np.ndarray]:
        """Copy a recent single/left frame for manual analysis; reject stopped/stale feeds."""
        with self._lock:
            if (
                not self._running
                or self._latest_left is None
                or time.monotonic() - self._last_frame_at > max_age_s
            ):
                return None
            return self._latest_left.copy()

    def snapshot_left_jpeg(self, quality: int = 85, max_age_s: Optional[float] = 2.5) -> Optional[bytes]:
        """Return JPEG bytes of current left frame (rejecting stale frames when max_age_s is set)."""
        with self._lock:
            if self._latest_left is None:
                return None
            if (
                max_age_s is not None
                and not self.is_mock
                and self._last_frame_at > 0
                and (time.monotonic() - self._last_frame_at) > max_age_s
            ):
                return None
            frame = self._latest_left.copy()

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def snapshot_stereo_composite_png(self, max_age_s: Optional[float] = 2.5) -> Optional[bytes]:
        """Return lossless PNG bytes of side-by-side composite [Left | Right] for size server."""
        with self._lock:
            if self._latest_left is None or self._latest_right is None:
                return None
            if max_age_s is not None and not self.is_mock:
                now_mono = time.monotonic()
                if self._last_frame_at > 0 and (now_mono - self._last_frame_at) > max_age_s:
                    return None
                if (
                    self.single_idx is None
                    and self._right_tel.last_good_mono > 0
                    and (now_mono - self._right_tel.last_good_mono) > max_age_s
                ):
                    return None
            composite = np.hstack((self._latest_left, self._latest_right))

        ok, buf = cv2.imencode(".png", composite)
        return buf.tobytes() if ok else None

    def capture_fresh_left(
        self,
        min_after_mono: Optional[float] = None,
        timeout_s: float = 4.0,
        quality: int = 90,
    ) -> Tuple[Optional[bytes], Optional[np.ndarray]]:
        """Wait for a genuinely fresh left frame acquired at or after `min_after_mono`.

        Automatically triggers a USB stream reopen if the camera froze right before capture.
        """
        target_mono = min_after_mono if min_after_mono is not None else time.monotonic()
        deadline = time.monotonic() + max(0.2, timeout_s)
        reopen_triggered = False
        wait_started = time.monotonic()

        with self._lock:
            start_frame_id = self._frame_id

        while True:
            frame: Optional[np.ndarray] = None
            with self._frame_cond:
                has_worker = bool(
                    (self._thread and self._thread.is_alive())
                    or (self._left_thread and self._left_thread.is_alive())
                )
                if not has_worker and self._latest_left is not None and self._last_frame_at >= target_mono:
                    frame = self._latest_left.copy()
                elif (
                    self._latest_left is not None
                    and self._last_frame_at >= target_mono
                    and (self._frame_id > start_frame_id or self.is_mock)
                ):
                    frame = self._latest_left.copy()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        self._frame_cond.wait(timeout=min(0.15, remaining))

            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if ok:
                    return buf.tobytes(), frame
                return None, None

            now_mono = time.monotonic()
            if now_mono >= deadline:
                break

            # If physical left camera hasn't delivered a fresh frame within 1.2s, force reopen immediately
            if (
                not self.is_mock
                and not reopen_triggered
                and (now_mono - wait_started) >= 1.2
                and self._running
            ):
                reopen_triggered = True
                self._reopen_side("left", reason="capture_fresh_left timeout", force=True)

        return None, None

    def capture_fresh_stereo(
        self,
        min_after_mono: Optional[float] = None,
        timeout_s: float = 4.5,
        quality: int = 90,
    ) -> Tuple[Optional[bytes], Optional[np.ndarray], Optional[bytes]]:
        """Wait for fresh post-settle Left and Right frames, returning (left_jpeg, left_bgr, stereo_png).

        If Right camera fails recovery while Left succeeds, still returns (left_jpeg, left_bgr, None)
        so visual inspection can proceed without losing the waypoint photo.
        """
        target_mono = min_after_mono if min_after_mono is not None else time.monotonic()
        deadline = time.monotonic() + max(0.2, timeout_s)
        reopen_left_done = False
        reopen_right_done = False
        wait_started = time.monotonic()

        with self._lock:
            start_l_seq = self._left_tel.seq
            start_r_seq = self._right_tel.seq

        best_left: Optional[np.ndarray] = None

        while True:
            left_copy: Optional[np.ndarray] = None
            right_copy: Optional[np.ndarray] = None

            with self._frame_cond:
                l_ready = (
                    self._latest_left is not None
                    and self._last_frame_at >= target_mono
                    and (self._left_tel.seq > start_l_seq or self.is_mock)
                )
                r_ready = (
                    self._latest_right is not None
                    and (
                        self.is_mock
                        or self.single_idx is not None
                        or (
                            self._right_tel.last_good_mono >= target_mono
                            and self._right_tel.seq > start_r_seq
                        )
                    )
                )

                if l_ready and best_left is None and self._latest_left is not None:
                    best_left = self._latest_left.copy()

                if l_ready and r_ready and self._latest_left is not None and self._latest_right is not None:
                    left_copy = self._latest_left.copy()
                    right_copy = self._latest_right.copy()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        self._frame_cond.wait(timeout=min(0.15, remaining))

            if left_copy is not None and right_copy is not None:
                if left_copy.shape[:2] != right_copy.shape[:2]:
                    right_copy = cv2.resize(right_copy, (left_copy.shape[1], left_copy.shape[0]))
                ok_jpg, jpg_buf = cv2.imencode(".jpg", left_copy, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                composite = np.hstack((left_copy, right_copy))
                ok_png, png_buf = cv2.imencode(".png", composite)
                return (
                    jpg_buf.tobytes() if ok_jpg else None,
                    left_copy if ok_jpg else None,
                    png_buf.tobytes() if ok_png else None,
                )

            now_mono = time.monotonic()
            if now_mono >= deadline:
                break

            if not self.is_mock and (now_mono - wait_started) >= 1.2 and self._running:
                if best_left is None and not reopen_left_done:
                    reopen_left_done = True
                    self._reopen_side("left", reason="capture_fresh_stereo left timeout", force=True)
                if self.single_idx is None and self._right_tel.device_idx is not None and not reopen_right_done:
                    with self._lock:
                        r_still_old = self._right_tel.last_good_mono < target_mono or self._right_tel.seq <= start_r_seq
                    if r_still_old:
                        reopen_right_done = True
                        self._reopen_side("right", reason="capture_fresh_stereo right timeout", force=True)

        # Graceful partial fallback: if Left succeeded post-settle, still return Left JPEG
        if best_left is not None:
            ok_jpg, jpg_buf = cv2.imencode(".jpg", best_left, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok_jpg:
                return jpg_buf.tobytes(), best_left, None

        return None, None, None

    def stop(self) -> None:
        self._running = False
        with self._frame_cond:
            self._left_tel.worker_gen += 1
            self._right_tel.worker_gen += 1
            self._frame_cond.notify_all()

        for th in (self._thread, self._left_thread, self._right_thread, self._watchdog_thread):
            if th and th.is_alive():
                th.join(timeout=0.8)

        old_l, old_r = self.cap_l, self.cap_r
        self.cap_l = None
        self.cap_r = None
        _release_capture_async(old_l)
        _release_capture_async(old_r)
