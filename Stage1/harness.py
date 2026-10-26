#!/usr/bin/env python3
"""Stage 1 Master Stereo Vision Harness.

Features:
- Camera input: Network pipe (laptop stream), Mock stereo pair, or Local webcams.
- Calibration & Epipolar Rectification: Loads stereo_calibration_fixed.yml and rectifies scanlines.
- Modular Depth Estimators: Dynamically toggles between CREStereo (Deep DL), HITNet (Fast DL), and StereoSGBM (Traditional).
- Multi-GPU Pipelining: Distributes frames across multiple GPUs via --gpus to eliminate sequential waiting.
- Absolute Depth & Metric Legend: Metric depth calculation Z = (fx * B) / d, colormap with labeled legend.
- Interactive Web Dashboard & Streamer: Accessible on http://localhost:8080 (auto-forwarded via VS Code).
- Cursor Depth Inspection: Real-time metric depth lookup at mouse hover.
- TA Measuring Tape HUD: Median depth aggregation over target ROI, live error vs ground truth.
- Local GUI: Optional native OpenCV window if an X11 DISPLAY is available.
"""

from __future__ import annotations
import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from estimators import (
    BaseStereoEstimator,
    DeepCREStereoEstimator,
    HitnetEstimator,
    TraditionalSGBMEstimator,
)
from utils import (
    NetworkStereoReceiver,
    StereoCalibration,
    attach_depth_legend,
    calculate_compactness,
    colorize_depth,
    colorize_disparity,
    disparity_to_depth,
    sample_roi_depth,
)


def parse_gpu_ids(val: str) -> list[int]:
    """Parses GPU argument into a list of integer GPU IDs.

    Examples:
        '0' -> [0]
        '0,1,2,3' -> [0, 1, 2, 3]
        '4' -> [0, 1, 2, 3]
        '2,3' -> [2, 3]
    """
    val = val.strip()
    if "," in val:
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    elif val.isdigit():
        num = int(val)
        return list(range(num)) if num > 1 else [num]
    return [0]


def _gpu_worker_entry(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    crestereo_path: str,
    hitnet_path: str | None,
    initial_estimator: str,
) -> None:
    """Independent worker process dedicated to a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from estimators.deep_crestereo import _setup_cuda_and_cudnn
    _setup_cuda_and_cudnn()

    estimators: dict[str, BaseStereoEstimator] = {
        "crestereo": DeepCREStereoEstimator(model_path=crestereo_path, device_id=0),
        "sgbm": TraditionalSGBMEstimator(),
        "hitnet": HitnetEstimator(model_path=hitnet_path, device_id=0),
    }

    active_name = initial_estimator if initial_estimator in estimators else "crestereo"
    active_est = estimators[active_name]
    try:
        active_est.initialize()
    except Exception as e:
        print(f"[Worker GPU {gpu_id}] Initializing {active_name} failed: {e}. Falling back to crestereo.")
        active_name = "crestereo"
        active_est = estimators["crestereo"]
        try:
            active_est.initialize()
        except Exception:
            pass

    result_queue.put(("READY", gpu_id, active_name))

    while True:
        try:
            msg = task_queue.get()
        except (KeyboardInterrupt, BrokenPipeError, EOFError):
            break
        if msg is None:
            break

        cmd, payload = msg
        if cmd == "INFER":
            frame_id, left_rect, right_rect, t_rectify_ms, t_net_recv, sender_fps, submit_t = payload
            t0 = time.perf_counter()
            try:
                disp = active_est.estimate_disparity(left_rect, right_rect)
            except Exception as e:
                print(f"[Worker GPU {gpu_id}] Inference error: {e}")
                disp = np.full((left_rect.shape[0], left_rect.shape[1]), np.nan, dtype=np.float32)

            infer_ms = (time.perf_counter() - t0) * 1000.0
            crop_bounds = getattr(active_est, "active_crop_bounds", None)

            result_queue.put((
                "RESULT",
                gpu_id,
                (
                    frame_id,
                    disp,
                    crop_bounds,
                    t_rectify_ms,
                    infer_ms,
                    left_rect,
                    right_rect,
                    t_net_recv,
                    sender_fps,
                    submit_t,
                ),
            ))

        elif cmd == "SWITCH":
            new_name = payload
            err_msg = None
            if new_name in estimators:
                try:
                    estimators[new_name].initialize()
                    active_est = estimators[new_name]
                    active_name = new_name
                except Exception as e:
                    err_msg = str(e)
            else:
                err_msg = f"Unknown estimator: {new_name}"
            result_queue.put(("SWITCH_DONE", gpu_id, (active_name, err_msg)))


class MultiGPUWorkerPool:
    """Manages worker processes distributed across multiple GPUs for parallel frame inference."""

    def __init__(
        self,
        gpu_ids: list[int],
        crestereo_path: str,
        hitnet_path: str | None,
        initial_estimator: str,
    ) -> None:
        self.gpu_ids = gpu_ids
        self.ctx = mp.get_context("spawn")
        self.task_queues: dict[int, mp.Queue] = {gid: self.ctx.Queue() for gid in gpu_ids}
        self.result_queue: mp.Queue = self.ctx.Queue()
        self.workers: list[mp.Process] = []
        self.busy_gpus: dict[int, bool] = {gid: False for gid in gpu_ids}
        self.lock = threading.Lock()
        self.frame_counter = 0

        print(f"[MultiGPU] Initializing worker pool across {len(gpu_ids)} GPUs: {gpu_ids}...")
        for gid in gpu_ids:
            p = self.ctx.Process(
                target=_gpu_worker_entry,
                args=(
                    gid,
                    self.task_queues[gid],
                    self.result_queue,
                    crestereo_path,
                    hitnet_path,
                    initial_estimator,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        ready_workers: set[int] = set()
        while len(ready_workers) < len(gpu_ids):
            msg, gid, _ = self.result_queue.get()
            if msg == "READY":
                ready_workers.add(gid)
                print(f"[MultiGPU] Worker on GPU {gid} initialized ({len(ready_workers)}/{len(gpu_ids)})")

    def submit(
        self,
        left_rect: np.ndarray,
        right_rect: np.ndarray,
        t_rectify_ms: float = 0.0,
        t_net_recv_ms: float = 0.0,
        sender_fps: float = 0.0,
    ) -> bool:
        """Dispatches a rectified stereo frame to an available GPU worker without blocking."""
        with self.lock:
            # Find an idle GPU worker
            idle_gpus = [gid for gid in self.gpu_ids if not self.busy_gpus[gid]]
            if idle_gpus:
                chosen_gpu = idle_gpus[0]
            else:
                # All workers busy; queue up to 1 frame on round-robin worker
                chosen_gpu = self.gpu_ids[self.frame_counter % len(self.gpu_ids)]
                if self.task_queues[chosen_gpu].qsize() >= 1:
                    # Drop stale frame to keep real-time latency minimal
                    return False

            self.busy_gpus[chosen_gpu] = True
            self.frame_counter += 1
            frame_id = self.frame_counter

            self.task_queues[chosen_gpu].put((
                "INFER",
                (frame_id, left_rect, right_rect, t_rectify_ms, t_net_recv_ms, sender_fps, time.perf_counter()),
            ))
            return True

    def mark_idle(self, gpu_id: int) -> None:
        with self.lock:
            self.busy_gpus[gpu_id] = False

    def switch_estimator(self, name: str) -> bool:
        with self.lock:
            for gid in self.gpu_ids:
                self.task_queues[gid].put(("SWITCH", name))

            success = True
            for _ in self.gpu_ids:
                msg, gid, (act_name, err) = self.result_queue.get()
                if err is not None:
                    print(f"[MultiGPU] GPU {gid} failed to switch to {name}: {err}")
                    success = False
            return success

    def shutdown(self) -> None:
        for q in self.task_queues.values():
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.workers:
            p.join(timeout=1.0)


class StereoPipeline:
    """Manages calibration, active estimator, multi-GPU pool, inference loop, and shared metric state."""

    def __init__(
        self,
        calibration: StereoCalibration,
        initial_estimator_name: str = "crestereo",
        model_path: str = "models/crestereo_embedded.onnx",
        hitnet_model_path: str | Path | None = None,
        gpu_ids: list[int] | None = None,
        min_depth: float = 0.3,
        max_depth: float = 15.0,
        resilient_voids: bool = True,
    ) -> None:
        self.calib = calibration
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.resilient_voids = resilient_voids
        self.lock = threading.Lock()
        self.gpu_ids = gpu_ids if gpu_ids is not None and len(gpu_ids) > 0 else [0]
        self.is_multigpu = len(self.gpu_ids) > 1

        stage1_dir = Path(__file__).resolve().parent
        m_path = Path(model_path)
        if not m_path.is_absolute() and not m_path.exists():
            candidate = stage1_dir / model_path
            if candidate.exists():
                model_path = str(candidate)

        # Initialize estimators dictionary (3 estimators: CREStereo, SGBM, HITNet)
        primary_gpu = self.gpu_ids[0]
        self.estimators: dict[str, BaseStereoEstimator] = {
            "crestereo": DeepCREStereoEstimator(model_path=model_path, device_id=primary_gpu),
            "sgbm": TraditionalSGBMEstimator(),
            "hitnet": HitnetEstimator(model_path=hitnet_model_path, device_id=primary_gpu),
        }
        self.active_estimator_name = (
            initial_estimator_name if initial_estimator_name in self.estimators else "crestereo"
        )
        self.active_estimator = self.estimators[self.active_estimator_name]

        # Multi-GPU worker pool or single-GPU local initialization
        self.worker_pool: MultiGPUWorkerPool | None = None
        self._running = True
        self._collector_thread: threading.Thread | None = None

        if self.is_multigpu:
            self.worker_pool = MultiGPUWorkerPool(
                gpu_ids=self.gpu_ids,
                crestereo_path=str(model_path),
                hitnet_path=str(hitnet_model_path) if hitnet_model_path else None,
                initial_estimator=self.active_estimator_name,
            )
            self._collector_thread = threading.Thread(target=self._result_collector_loop, daemon=True)
            self._collector_thread.start()
        else:
            print(f"[Pipeline] Initializing active estimator: {self.active_estimator.name} on GPU {primary_gpu}...")
            self.active_estimator.initialize()

        # State buffers
        self.latest_left_rect: np.ndarray | None = None
        self.latest_disparity: np.ndarray | None = None
        self.latest_depth_m: np.ndarray | None = None

        # Create initial standby placeholder so video feed renders immediately on connection
        placeholder = np.full((270, 1510, 3), 24, dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Stage 1 Stereo Vision - Standby (Waiting for camera feed / sender)",
            (320, 142),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (56, 189, 248),
            2,
            cv2.LINE_AA,
        )
        _, p_buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.latest_composite_jpeg: bytes | None = p_buf.tobytes()
        self.preview_dims: tuple[int, int] = (1510, 270)  # total width, height

        # Metrics & Latency breakdown
        self.fps: float = 0.0
        self.total_ms: float = 0.0
        self.t_rectify_ms: float = 0.0
        self.t_infer_ms: float = 0.0
        self.t_depth_ms: float = 0.0
        self.t_vis_ms: float = 0.0
        self.t_net_recv_ms: float = 0.0
        self.sender_fps: float = 0.0
        self.swap_cameras: bool = False
        self._last_completed_t: float = 0.0

        self.compactness_pct: float = 0.0
        self.target_x: int = 960
        self.target_y: int = 540
        self.target_depth_m: float = float("nan")
        self.target_std_m: float = float("nan")
        self.target_valid_ratio: float = 0.0
        self._incoming_network_pair: tuple[np.ndarray, np.ndarray] | None = None

    def set_depth_bounds(
        self,
        min_depth: float | None = None,
        max_depth: float | None = None,
        resilient_voids: bool | None = None,
    ) -> tuple[float, float, bool]:
        """Dynamically update depth bounds and far-field resilience."""
        with self.lock:
            if min_depth is not None and min_depth > 0:
                self.min_depth = float(min_depth)
            if max_depth is not None and max_depth > self.min_depth:
                self.max_depth = float(max_depth)
            if resilient_voids is not None:
                self.resilient_voids = bool(resilient_voids)
            return self.min_depth, self.max_depth, self.resilient_voids

    def _result_collector_loop(self) -> None:
        """Continuously collects completed inference results from the multi-GPU pool."""
        while self._running:
            if self.worker_pool is None:
                break
            try:
                msg = self.worker_pool.result_queue.get(timeout=0.1)
            except Exception:
                continue

            if msg is None:
                break

            msg_type, gpu_id, payload = msg
            if msg_type == "RESULT":
                self.worker_pool.mark_idle(gpu_id)
                (
                    frame_id,
                    disp,
                    crop_bounds,
                    t_rectify_ms,
                    infer_ms,
                    left_rect,
                    right_rect,
                    t_net_recv,
                    sender_fps,
                    submit_t,
                ) = payload

                self._finish_frame_pipeline(
                    disparity=disp,
                    crop_bounds=crop_bounds,
                    left_rect=left_rect,
                    right_rect=right_rect,
                    t_rectify_ms=t_rectify_ms,
                    t_infer_ms=infer_ms,
                    t_net_recv_ms=t_net_recv,
                    sender_fps=sender_fps,
                    submit_t=submit_t,
                    gpu_id=gpu_id,
                )

    def enqueue_network_pair(
        self,
        left: np.ndarray,
        right: np.ndarray,
        recv_ms: float = 0.0,
        sender_fps: float = 0.0,
    ) -> None:
        with self.lock:
            self._incoming_network_pair = (left, right)
            if recv_ms > 0:
                self.t_net_recv_ms = recv_ms
            if sender_fps > 0:
                self.sender_fps = sender_fps

    def toggle_swap_cameras(self) -> bool:
        with self.lock:
            self.swap_cameras = not self.swap_cameras
            print(f"[Pipeline] Swap cameras set to: {self.swap_cameras}")
            return self.swap_cameras

    def switch_estimator(self, name: str) -> bool:
        with self.lock:
            if name not in self.estimators:
                return False
            if name == self.active_estimator_name:
                return True

            print(f"[Pipeline] Switching estimator to: {name}")
            if self.is_multigpu and self.worker_pool is not None:
                ok = self.worker_pool.switch_estimator(name)
                if not ok:
                    print(f"[Pipeline] Error switching multi-GPU workers to: {name}")
                    return False
            else:
                try:
                    self.estimators[name].initialize()
                except Exception as e:
                    print(f"[Pipeline] Error initializing estimator '{name}': {e}")
                    return False

            self.active_estimator = self.estimators[name]
            self.active_estimator_name = name
            return True

    def _display_to_native_coords(self, display_x: int, display_y: int) -> tuple[int, int] | None:
        if self.latest_depth_m is None:
            return None
        orig_h, orig_w = self.latest_depth_m.shape[:2]
        panel_w = 480
        panel_h = 270

        if 2 * panel_w <= display_x < 3 * panel_w:
            rel_x = display_x - 2 * panel_w
        elif 0 <= display_x < panel_w:
            rel_x = display_x
        elif panel_w <= display_x < 2 * panel_w:
            rel_x = display_x - panel_w
        else:
            return None

        map_x = int(np.clip(rel_x * (orig_w / float(panel_w)), 0, orig_w - 1))
        map_y = int(np.clip(display_y * (orig_h / float(panel_h)), 0, orig_h - 1))
        return map_x, map_y

    def set_target_roi(self, display_x: int, display_y: int) -> None:
        coords = self._display_to_native_coords(display_x, display_y)
        if coords is not None:
            with self.lock:
                self.target_x, self.target_y = coords

    def get_depth_at(self, display_x: int, display_y: int) -> tuple[bool, float]:
        with self.lock:
            if self.latest_depth_m is None:
                return False, 0.0

            coords = self._display_to_native_coords(display_x, display_y)
            if coords is None:
                return False, 0.0

            map_x, map_y = coords
            val = float(self.latest_depth_m[map_y, map_x])
            if np.isfinite(val) and val > 0:
                return True, val

            # Check 3x3 neighborhood if point lands on a single noisy boundary pixel
            patch = self.latest_depth_m[
                max(0, map_y - 1) : map_y + 2, max(0, map_x - 1) : map_x + 2
            ]
            valid_patch = patch[np.isfinite(patch) & (patch > 0)]
            if len(valid_patch) > 0:
                return True, float(np.median(valid_patch))
            return False, 0.0

    def submit_pair(
        self,
        left_raw: np.ndarray,
        right_raw: np.ndarray,
        recv_ms: float = 0.0,
        sender_fps: float = 0.0,
    ) -> bool:
        """Entry point for incoming stereo frames.

        In multi-GPU mode, rectifies scanlines and dispatches to an idle GPU worker without blocking.
        In single-GPU mode, processes sequentially.
        """
        if self.swap_cameras:
            left_raw, right_raw = right_raw, left_raw

        t0 = time.perf_counter()
        left_rect, right_rect = self.calib.rectify_pair(left_raw, right_raw)
        t_rectify_ms = (time.perf_counter() - t0) * 1000.0

        if self.is_multigpu and self.worker_pool is not None:
            # Non-blocking dispatch across GPUs
            return self.worker_pool.submit(
                left_rect,
                right_rect,
                t_rectify_ms=t_rectify_ms,
                t_net_recv_ms=recv_ms,
                sender_fps=sender_fps,
            )
        else:
            # Sequential single-GPU processing
            t_submit = time.perf_counter()
            t1 = time.perf_counter()
            disparity = self.active_estimator.estimate_disparity(left_rect, right_rect)
            t_infer_ms = (time.perf_counter() - t1) * 1000.0
            crop_bounds = getattr(self.active_estimator, "active_crop_bounds", None)
            self._finish_frame_pipeline(
                disparity=disparity,
                crop_bounds=crop_bounds,
                left_rect=left_rect,
                right_rect=right_rect,
                t_rectify_ms=t_rectify_ms,
                t_infer_ms=t_infer_ms,
                t_net_recv_ms=recv_ms,
                sender_fps=sender_fps,
                submit_t=t_submit,
                gpu_id=self.gpu_ids[0],
            )
            return True

    def process_pair(self, left_raw: np.ndarray, right_raw: np.ndarray) -> None:
        """Backwards-compatible synchronous frame processing method."""
        self.submit_pair(left_raw, right_raw)

    def _finish_frame_pipeline(
        self,
        disparity: np.ndarray,
        crop_bounds: tuple[int, int] | None,
        left_rect: np.ndarray,
        right_rect: np.ndarray,
        t_rectify_ms: float,
        t_infer_ms: float,
        t_net_recv_ms: float,
        sender_fps: float,
        submit_t: float,
        gpu_id: int | None = None,
    ) -> None:
        """Executes metric depth conversion, reticle sampling, HUD rendering, and state update."""
        t2 = time.perf_counter()
        focal_px = self.calib.get_focal_length(left_rect.shape[1], left_rect.shape[0])
        with self.lock:
            cur_min_depth = self.min_depth
            cur_max_depth = self.max_depth
            cur_resilient = self.resilient_voids

        depth_m = disparity_to_depth(
            disparity=disparity,
            focal_length_px=focal_px,
            baseline_m=self.calib.baseline_m,
            min_depth=cur_min_depth,
            max_depth=cur_max_depth,
            resilient_voids=cur_resilient,
        )

        # Compactness calculation
        if crop_bounds is not None:
            compactness = calculate_compactness(
                depth_m, crop_x_offset=crop_bounds[0], crop_w=crop_bounds[1]
            )
        else:
            compactness = calculate_compactness(depth_m)

        # Target ROI depth sampling
        orig_h, orig_w = depth_m.shape[:2]
        with self.lock:
            cur_tx = self.target_x
            cur_ty = self.target_y
        tx = max(0, min(orig_w - 1, cur_tx))
        ty = max(0, min(orig_h - 1, cur_ty))
        med_z, std_z, valid_ratio = sample_roi_depth(depth_m, tx, ty, box_radius=15)
        t_depth_ms = (time.perf_counter() - t2) * 1000.0

        # Visualizations
        t3 = time.perf_counter()
        target_panel_w, target_panel_h = 480, 270
        left_vis = cv2.resize(left_rect, (target_panel_w, target_panel_h), interpolation=cv2.INTER_LINEAR)
        disp_colored = colorize_disparity(disparity)
        disp_vis = cv2.resize(disp_colored, (target_panel_w, target_panel_h), interpolation=cv2.INTER_LINEAR)
        depth_colored = colorize_depth(depth_m, cur_min_depth, cur_max_depth)
        depth_vis = cv2.resize(depth_colored, (target_panel_w, target_panel_h), interpolation=cv2.INTER_LINEAR)

        # Draw target reticle
        scale_x = target_panel_w / float(orig_w)
        scale_y = target_panel_h / float(orig_h)
        scaled_tx = int(tx * scale_x)
        scaled_ty = int(ty * scale_y)
        box_r = int(15 * scale_x)

        for vis in (left_vis, depth_vis):
            cv2.rectangle(
                vis,
                (scaled_tx - box_r, scaled_ty - box_r),
                (scaled_tx + box_r, scaled_ty + box_r),
                (0, 255, 0),
                2,
            )
            cv2.circle(vis, (scaled_tx, scaled_ty), 3, (0, 0, 255), -1)

        depth_with_legend = attach_depth_legend(
            depth_vis,
            min_depth=cur_min_depth,
            max_depth=cur_max_depth,
            legend_width=70,
        )

        # Side-by-side composite: [Left | Disparity | Depth + Legend]
        composite = np.hstack((left_vis, disp_vis, depth_with_legend))

        # HUD banner
        h_bar = 24
        banner_overlay = composite[0:h_bar, :].copy()
        cv2.rectangle(composite, (0, 0), (composite.shape[1], h_bar), (15, 23, 42), -1)
        cv2.addWeighted(composite[0:h_bar, :], 0.82, banner_overlay, 0.18, 0, composite[0:h_bar, :])

        t_total_ms = (time.perf_counter() - submit_t) * 1000.0
        now = time.perf_counter()
        if self._last_completed_t > 0:
            instant_fps = 1.0 / max(1e-4, now - self._last_completed_t)
            if self.fps <= 0:
                self.fps = instant_fps
            else:
                self.fps = 0.80 * self.fps + 0.20 * instant_fps
        else:
            self.fps = 1.0 / max(1e-4, time.perf_counter() - submit_t)
        self._last_completed_t = now

        swap_tag = " [SWAPPED]" if self.swap_cameras else ""
        gpu_tag = f" [GPU {gpu_id}]" if gpu_id is not None and len(self.gpu_ids) > 1 else ""
        banner_text = (
            f"{self.active_estimator.name}{gpu_tag}{swap_tag} | {self.fps:.1f} FPS ({t_total_ms:.0f} ms) | "
            f"Compactness: {compactness:.1f}% | Target: {med_z:.2f}m"
        )
        cv2.putText(
            composite,
            banner_text,
            (12, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        _, jpeg_buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = jpeg_buf.tobytes()
        t_vis_ms = (time.perf_counter() - t3) * 1000.0

        with self.lock:
            self.latest_left_rect = left_rect
            self.latest_disparity = disparity
            self.latest_depth_m = depth_m
            self.latest_composite_jpeg = jpeg_bytes
            self.preview_dims = (composite.shape[1], composite.shape[0])
            self.total_ms = t_total_ms
            self.t_rectify_ms = t_rectify_ms
            self.t_infer_ms = t_infer_ms
            self.t_depth_ms = t_depth_ms
            self.t_vis_ms = t_vis_ms
            if t_net_recv_ms > 0:
                self.t_net_recv_ms = t_net_recv_ms
            if sender_fps > 0:
                self.sender_fps = sender_fps
            self.compactness_pct = compactness
            self.target_depth_m = med_z
            self.target_std_m = std_z
            self.target_valid_ratio = valid_ratio

    def stop(self) -> None:
        """Terminates worker pool and background threads cleanly."""
        self._running = False
        if self.worker_pool is not None:
            self.worker_pool.shutdown()


class WebServerHandler(BaseHTTPRequestHandler):
    """Serves dashboard UI, MJPEG video feed, and depth query REST APIs."""

    pipeline: StereoPipeline

    def do_HEAD(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index_file = Path(__file__).parent / "static" / "index.html"
            if index_file.exists():
                content = index_file.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "index.html not found")

        elif path == "/video_feed":
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            while True:
                time.sleep(0.03)
                with self.pipeline.lock:
                    frame = self.pipeline.latest_composite_jpeg
                if frame is None:
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break

        elif path == "/api/depth_at":
            query = parse_qs(parsed.query)
            x = int(query.get("x", [0])[0])
            y = int(query.get("y", [0])[0])
            valid, depth_m = self.pipeline.get_depth_at(x, y)
            data = {"valid": valid, "depth_m": depth_m, "x": x, "y": y}
            self._send_json(data)

        elif path == "/api/metrics":
            with self.pipeline.lock:
                data = {
                    "fps": self.pipeline.fps,
                    "total_ms": self.pipeline.total_ms,
                    "rectify_ms": self.pipeline.t_rectify_ms,
                    "infer_ms": self.pipeline.t_infer_ms,
                    "depth_ms": self.pipeline.t_depth_ms,
                    "vis_ms": self.pipeline.t_vis_ms,
                    "net_recv_ms": self.pipeline.t_net_recv_ms,
                    "sender_fps": self.pipeline.sender_fps,
                    "swap_cameras": self.pipeline.swap_cameras,
                    "estimator": self.pipeline.active_estimator.name,
                    "estimator_key": self.pipeline.active_estimator_name,
                    "compactness": self.pipeline.compactness_pct,
                    "target_depth": self.pipeline.target_depth_m,
                    "target_std": self.pipeline.target_std_m,
                    "target_x": self.pipeline.target_x,
                    "target_y": self.pipeline.target_y,
                    "gpus": self.pipeline.gpu_ids,
                    "min_depth": self.pipeline.min_depth,
                    "max_depth": self.pipeline.max_depth,
                    "resilient_voids": self.pipeline.resilient_voids,
                }
            self._send_json(data)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/switch_estimator":
            query = parse_qs(parsed.query)
            name = query.get("name", ["crestereo"])[0]
            ok = self.pipeline.switch_estimator(name)
            self._send_json({
                "success": ok,
                "estimator": self.pipeline.active_estimator_name,
                "estimator_display": self.pipeline.active_estimator.name,
            })

        elif path == "/api/swap_cameras":
            swapped = self.pipeline.toggle_swap_cameras()
            self._send_json({"success": True, "swap_cameras": swapped})

        elif path == "/api/set_target":
            query = parse_qs(parsed.query)
            x = int(query.get("x", [960])[0])
            y = int(query.get("y", [540])[0])
            self.pipeline.set_target_roi(x, y)
            self._send_json({"success": True, "target_x": x, "target_y": y})

        elif path == "/api/set_depth_range":
            query = parse_qs(parsed.query)
            min_val = float(query.get("min", [self.pipeline.min_depth])[0]) if "min" in query else None
            max_val = float(query.get("max", [self.pipeline.max_depth])[0]) if "max" in query else None
            resilient_val = None
            if "resilient" in query:
                resilient_val = query.get("resilient", ["1"])[0].lower() in ("1", "true", "yes")
            c_min, c_max, c_res = self.pipeline.set_depth_bounds(min_val, max_val, resilient_val)
            self._send_json({
                "success": True,
                "min_depth": c_min,
                "max_depth": c_max,
                "resilient_voids": c_res,
            })

        elif path == "/api/push_frame":
            t_recv_start = time.perf_counter()
            content_length = int(self.headers.get("Content-Length", 0))
            sender_fps_header = self.headers.get("X-Sender-FPS", "0.0")
            try:
                sender_fps = float(sender_fps_header)
            except ValueError:
                sender_fps = 0.0

            if content_length > 0:
                payload = self.rfile.read(content_length)
                np_arr = np.frombuffer(payload, dtype=np.uint8)
                composite = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if composite is not None:
                    h, w = composite.shape[:2]
                    mid_x = w // 2
                    left_frame = composite[:, :mid_x]
                    right_frame = composite[:, mid_x:]
                    t_recv_ms = (time.perf_counter() - t_recv_start) * 1000.0
                    self.pipeline.enqueue_network_pair(
                        left_frame, right_frame, recv_ms=t_recv_ms, sender_fps=sender_fps
                    )
            self._send_json({"success": True})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _send_json(self, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["network", "mock", "webcam"],
        default="mock",
        help="Frame source: 'network' (laptop stream), 'mock' (test pair in loop), 'webcam' (local).",
    )
    stage1_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--estimator",
        choices=["crestereo", "sgbm", "hitnet"],
        default="crestereo",
        help="Initial depth estimator: 'crestereo', 'sgbm', or 'hitnet' (default: crestereo).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU device ID(s) to distribute inference across (e.g. '0', '0,1,2,3', or '4' for 4 GPUs). Default: 0.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.3,
        help="Minimum reliable metric depth in metres (default: 0.3).",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=15.0,
        help="Maximum reliable metric depth in metres (default: 15.0).",
    )
    parser.add_argument(
        "--resilient-voids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable far-field void resilience (clamping small/negative disparities to max_depth) default: True.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=stage1_root / "calibration" / "stereo_calibration_fixed.yml",
        help="Path to YAML calibration file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=stage1_root / "models" / "crestereo_embedded.onnx",
        help="Path to CREStereo ONNX model file.",
    )
    parser.add_argument(
        "--hitnet-model",
        type=Path,
        default=None,
        help="Path to HITNet ONNX model file (defaults to auto-detection in models/).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8080,
        help="HTTP port for interactive web dashboard (default: 8080).",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=5555,
        help="TCP port to listen for laptop camera stream (default: 5555).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Native camera width (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Native camera height (default: 1080).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Also open native OpenCV window if an X11 DISPLAY is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpus)

    print("=" * 70)
    print("  STAGE 1 STEREO VISION HARNESS")
    print(f"  Source: {args.source.upper()} | Initial Estimator: {args.estimator.upper()}")
    print(f"  GPU Worker Devices: {gpu_ids} ({'Multi-GPU Parallel' if len(gpu_ids) > 1 else 'Single-GPU Sequential'})")
    print(f"  Depth Range: [{args.min_depth}m - {args.max_depth}m] | Void Resilience: {args.resilient_voids}")
    print("=" * 70)

    # 1. Load calibration & epipolar rectification maps
    calib = StereoCalibration(
        calibration_path=args.calibration,
        target_size=(args.width, args.height),
    )

    # 2. Initialize pipeline
    pipeline = StereoPipeline(
        calibration=calib,
        initial_estimator_name=args.estimator,
        model_path=str(args.model),
        hitnet_model_path=args.hitnet_model,
        gpu_ids=gpu_ids,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        resilient_voids=args.resilient_voids,
    )

    # 3. Start Web Server in background thread
    WebServerHandler.pipeline = pipeline
    server = ThreadingHTTPServer(("0.0.0.0", args.web_port), WebServerHandler)
    web_thread = threading.Thread(target=server.serve_forever, daemon=True)
    web_thread.start()
    print(f"[WebUI] Interactive dashboard running at http://0.0.0.0:{args.web_port}")
    print(f"[WebUI] In VS Code, open forwarded port {args.web_port} in your browser.")

    # 4. Setup Frame Source
    receiver: NetworkStereoReceiver | None = None
    mock_left: np.ndarray | None = None
    mock_right: np.ndarray | None = None

    if args.source == "network":
        receiver = NetworkStereoReceiver(host="0.0.0.0", port=args.stream_port)
        receiver.start()
        print(f"[Source] Waiting for frames from laptop on port {args.stream_port}...")
    elif args.source == "mock":
        test_dir = Path(__file__).resolve().parent / "test_data" / "cones"
        l_path = test_dir / "im2.png"
        r_path = test_dir / "im6.png"
        if not l_path.exists() or not r_path.exists():
            raise FileNotFoundError(f"Mock images not found at {l_path} / {r_path}")
        raw_l = cv2.imread(str(l_path))
        raw_r = cv2.imread(str(r_path))
        mock_left = cv2.resize(raw_l, (args.width, args.height))
        mock_right = cv2.resize(raw_r, (args.width, args.height))
        print(f"[Source] Mock mode initialized using {l_path.name} & {r_path.name}.")
    elif args.source == "webcam":
        cam_l = cv2.VideoCapture(0)
        cam_r = cv2.VideoCapture(1)
        for cam in (cam_l, cam_r):
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # 5. Main Processing Loop
    try:
        while True:
            left_frame: np.ndarray | None = None
            right_frame: np.ndarray | None = None
            recv_ms = 0.0
            sender_fps = 0.0

            if args.source == "network":
                pair = None
                with pipeline.lock:
                    if pipeline._incoming_network_pair is not None:
                        pair = pipeline._incoming_network_pair
                        pipeline._incoming_network_pair = None
                        recv_ms = pipeline.t_net_recv_ms
                        sender_fps = pipeline.sender_fps
                if pair is None and receiver is not None:
                    pair = receiver.get_latest_pair()
                if pair is None:
                    time.sleep(0.005)
                    continue
                left_frame, right_frame = pair
            elif args.source == "mock":
                left_frame = mock_left
                right_frame = mock_right
                time.sleep(0.03)  # ~30 FPS loop for mock feed
            elif args.source == "webcam":
                cam_l.grab()
                cam_r.grab()
                _, left_frame = cam_l.retrieve()
                _, right_frame = cam_r.retrieve()
                if left_frame is None or right_frame is None:
                    continue

            if left_frame is not None and right_frame is not None:
                pipeline.submit_pair(
                    left_frame,
                    right_frame,
                    recv_ms=recv_ms,
                    sender_fps=sender_fps,
                )

                if args.display:
                    with pipeline.lock:
                        disp_frame = pipeline.latest_composite_jpeg
                    if disp_frame is not None:
                        np_arr = np.frombuffer(disp_frame, dtype=np.uint8)
                        vis = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        cv2.imshow("Stage 1 Stereo Vision", vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        elif key == ord("1"):
                            pipeline.switch_estimator("crestereo")
                        elif key == ord("2"):
                            pipeline.switch_estimator("sgbm")
                        elif key == ord("3"):
                            pipeline.switch_estimator("hitnet")

    except KeyboardInterrupt:
        print("\n[Harness] Terminating...")
    finally:
        pipeline.stop()
        if receiver:
            receiver.stop()
        server.shutdown()
        if args.display:
            cv2.destroyAllWindows()
        print("[Harness] Cleaned up.")


if __name__ == "__main__":
    main()
