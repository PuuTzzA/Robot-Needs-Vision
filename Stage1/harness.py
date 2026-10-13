#!/usr/bin/env python3
"""Stage 1 Master Stereo Vision Harness.

Features:
- Camera input: Network pipe (laptop stream), Mock stereo pair, or Local webcams.
- Calibration & Epipolar Rectification: Loads stereo_calibration_fixed.yml and rectifies scanlines.
- Modular Depth Estimators: Dynamically toggles between CREStereo (Deep DL) and StereoSGBM (Traditional).
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
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from estimators import BaseStereoEstimator, DeepCREStereoEstimator, TraditionalSGBMEstimator
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


class StereoPipeline:
    """Manages calibration, active estimator, inference loop, and shared metric state."""

    def __init__(
        self,
        calibration: StereoCalibration,
        initial_estimator_name: str = "crestereo",
        model_path: str = "models/crestereo_embedded.onnx",
        min_depth: float = 0.3,
        max_depth: float = 5.0,
    ) -> None:
        self.calib = calibration
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.lock = threading.Lock()

        # Initialize estimators
        self.estimators: dict[str, BaseStereoEstimator] = {
            "crestereo": DeepCREStereoEstimator(model_path=model_path),
            "sgbm": TraditionalSGBMEstimator(),
        }
        self.active_estimator_name = (
            initial_estimator_name if initial_estimator_name in self.estimators else "crestereo"
        )
        self.active_estimator = self.estimators[self.active_estimator_name]
        print(f"[Pipeline] Initializing active estimator: {self.active_estimator.name}...")
        self.active_estimator.initialize()

        # State buffers
        self.latest_left_rect: np.ndarray | None = None
        self.latest_disparity: np.ndarray | None = None
        self.latest_depth_m: np.ndarray | None = None
        self.latest_composite_jpeg: bytes | None = None
        self.preview_dims: tuple[int, int] = (1440, 360)  # total width, height

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

        self.compactness_pct: float = 0.0
        self.target_x: int = 960
        self.target_y: int = 540
        self.target_depth_m: float = float("nan")
        self.target_std_m: float = float("nan")
        self.target_valid_ratio: float = 0.0
        self._incoming_network_pair: tuple[np.ndarray, np.ndarray] | None = None

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
            new_est = self.estimators[name]
            new_est.initialize()
            self.active_estimator = new_est
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

    def process_pair(self, left_raw: np.ndarray, right_raw: np.ndarray) -> None:
        start_t = time.perf_counter()

        # Handle camera swap if toggled
        if self.swap_cameras:
            left_raw, right_raw = right_raw, left_raw

        # 1. Epipolar rectification (native resolution via cached calibration maps)
        t0 = time.perf_counter()
        left_rect, right_rect = self.calib.rectify_pair(left_raw, right_raw)
        t_rectify_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Run active stereo depth estimator
        t1 = time.perf_counter()
        disparity = self.active_estimator.estimate_disparity(left_rect, right_rect)
        t_infer_ms = (time.perf_counter() - t1) * 1000.0

        # 3. Compute absolute metric depth Z = (fx * B) / d
        t2 = time.perf_counter()
        focal_px = self.calib.get_focal_length(left_rect.shape[1], left_rect.shape[0])
        depth_m = disparity_to_depth(
            disparity=disparity,
            focal_length_px=focal_px,
            baseline_m=self.calib.baseline_m,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

        # 4. Compute metrics (calculate compactness within active evaluated FOV)
        crop_bounds = getattr(self.active_estimator, "active_crop_bounds", None)
        if crop_bounds is not None:
            compactness = calculate_compactness(
                depth_m, crop_x_offset=crop_bounds[0], crop_w=crop_bounds[1]
            )
        else:
            compactness = calculate_compactness(depth_m)

        # Sample target ROI depth
        orig_h, orig_w = depth_m.shape[:2]
        tx = max(0, min(orig_w - 1, self.target_x))
        ty = max(0, min(orig_h - 1, self.target_y))
        med_z, std_z, valid_ratio = sample_roi_depth(depth_m, tx, ty, box_radius=15)
        t_depth_ms = (time.perf_counter() - t2) * 1000.0

        # 5. Visualizations
        t3 = time.perf_counter()
        target_panel_w, target_panel_h = 480, 270
        left_vis = cv2.resize(left_rect, (target_panel_w, target_panel_h), interpolation=cv2.INTER_LINEAR)
        disp_colored = colorize_disparity(disparity)
        disp_vis = cv2.resize(disp_colored, (target_panel_w, target_panel_h), interpolation=cv2.INTER_LINEAR)
        depth_colored = colorize_depth(depth_m, self.min_depth, self.max_depth)
        depth_vis = cv2.resize(depth_colored, (target_panel_w, target_panel_h), interpolation=cv2.INTER_LINEAR)

        # Draw target reticle on Left View and Depth View
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

        # Attach metric depth color legend
        depth_with_legend = attach_depth_legend(
            depth_vis,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            legend_width=70,
        )

        # Assemble side-by-side composite: [Left | Disparity | Depth + Legend]
        composite = np.hstack((left_vis, disp_vis, depth_with_legend))

        # Add clean, non-overlapping translucent dark HUD banner across the top
        h_bar = 24
        banner_overlay = composite[0:h_bar, :].copy()
        cv2.rectangle(composite, (0, 0), (composite.shape[1], h_bar), (15, 23, 42), -1)
        cv2.addWeighted(composite[0:h_bar, :], 0.82, banner_overlay, 0.18, 0, composite[0:h_bar, :])

        t_elapsed = time.perf_counter() - start_t
        t_total_ms = t_elapsed * 1000.0
        fps = 1.0 / max(1e-4, t_elapsed)

        swap_tag = " [SWAPPED]" if self.swap_cameras else ""
        banner_text = (
            f"{self.active_estimator.name}{swap_tag} | {fps:.1f} FPS ({t_total_ms:.0f} ms) | "
            f"Compactness: {compactness:.1f}% | Target: {med_z:.2f}m"
        )
        # Single crisp white text render with dark pill backing
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

        # Update shared state
        with self.lock:
            self.latest_left_rect = left_rect
            self.latest_disparity = disparity
            self.latest_depth_m = depth_m
            self.latest_composite_jpeg = jpeg_bytes
            self.preview_dims = (composite.shape[1], composite.shape[0])
            self.fps = fps
            self.total_ms = t_total_ms
            self.t_rectify_ms = t_rectify_ms
            self.t_infer_ms = t_infer_ms
            self.t_depth_ms = t_depth_ms
            self.t_vis_ms = t_vis_ms
            self.compactness_pct = compactness
            self.target_depth_m = med_z
            self.target_std_m = std_z
            self.target_valid_ratio = valid_ratio


class WebServerHandler(BaseHTTPRequestHandler):
    """Serves the dashboard UI, MJPEG video feed, and depth query REST APIs."""

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
                    "compactness": self.pipeline.compactness_pct,
                    "target_depth": self.pipeline.target_depth_m,
                    "target_std": self.pipeline.target_std_m,
                    "target_x": self.pipeline.target_x,
                    "target_y": self.pipeline.target_y,
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
            self._send_json({"success": ok, "estimator": name})

        elif path == "/api/swap_cameras":
            swapped = self.pipeline.toggle_swap_cameras()
            self._send_json({"success": True, "swap_cameras": swapped})

        elif path == "/api/set_target":
            query = parse_qs(parsed.query)
            x = int(query.get("x", [960])[0])
            y = int(query.get("y", [540])[0])
            self.pipeline.set_target_roi(x, y)
            self._send_json({"success": True, "target_x": x, "target_y": y})

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
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        # Suppress noisy HTTP request logging
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
        choices=["crestereo", "sgbm"],
        default="crestereo",
        help="Initial depth estimator (default: crestereo).",
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
    print("=" * 70)
    print("  STAGE 1 STEREO VISION HARNESS")
    print(f"  Source: {args.source.upper()} | Initial Estimator: {args.estimator.upper()}")
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

            if args.source == "network":
                pair = None
                with pipeline.lock:
                    if pipeline._incoming_network_pair is not None:
                        pair = pipeline._incoming_network_pair
                        pipeline._incoming_network_pair = None
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
                pipeline.process_pair(left_frame, right_frame)

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

    except KeyboardInterrupt:
        print("\n[Harness] Terminating...")
    finally:
        if receiver:
            receiver.stop()
        server.shutdown()
        if args.display:
            cv2.destroyAllWindows()
        print("[Harness] Cleaned up.")


if __name__ == "__main__":
    main()

