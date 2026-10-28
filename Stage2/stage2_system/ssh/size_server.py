#!/usr/bin/env python3
"""Stage 2 SSH Size Estimation Server (GPU Server).

Receives single stereo pairs over HTTP, executes epipolar rectification,
runs CREStereo/HITNet disparity estimation, predicts YOLO instance segmentations,
and computes 3D metric bounding boxes, width, and height in cm.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import cv2
import numpy as np

# Ensure ssh directory is in path
ssh_dir = Path(__file__).resolve().parent
if str(ssh_dir) not in sys.path:
    sys.path.insert(0, str(ssh_dir))

from protocol import DEFAULT_SSH_SIZE_PORT, DetectionItem, MeasurementResult
from measure import draw_detection_overlay, robust_measurement, select_primary_detection
from utils.calibration_loader import StereoCalibration
from utils.depth_converter import colorize_depth, disparity_to_depth
from estimators.deep_crestereo import DeepCREStereoEstimator


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class SizeServerHandler(BaseHTTPRequestHandler):
    """Processes incoming stereo measurement requests."""

    server_core: SizeServerCore

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            data = {
                "status": "ok",
                "estimator": self.server_core.estimator_name,
                "yolo_model": self.server_core.yolo_model_name,
                "device": self.server_core.device,
                "is_mock": self.server_core.is_mock,
            }
            self._send_json(data)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/measure":
            query = parse_qs(parsed.query)
            point_name = query.get("point", [None])[0]
            hint_class = query.get("hint", [None])[0]

            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self.send_error(400, "Empty payload")
                return

            payload = self.rfile.read(length)
            result = self.server_core.process_measurement(payload, point_name, hint_class)
            self._send_json(result.to_dict())
        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class SizeServerCore:
    """Core stereo and YOLO perception inference pipeline."""

    def __init__(
        self,
        calib_path: Path,
        model_path: Path,
        yolo_path: str = "yolo11x-seg.pt",
        device: str = "cuda:0",
        is_mock: bool = False,
    ) -> None:
        self.calib_path = calib_path
        self.model_path = model_path
        self.yolo_model_name = yolo_path
        self.device = device
        self.is_mock = is_mock
        self.estimator_name = "crestereo"

        self.calibration: Optional[StereoCalibration] = None
        self.estimator: Optional[DeepCREStereoEstimator] = None
        self.yolo_model = None

        if not self.is_mock:
            self._initialize_models()

    def _initialize_models(self) -> None:
        print("[SizeServer] Loading stereo calibration...")
        self.calibration = StereoCalibration(calibration_path=self.calib_path)

        print(f"[SizeServer] Initializing CREStereo estimator on {self.device}...")
        gpu_id = 0
        if "cuda:" in self.device:
            try:
                gpu_id = int(self.device.split(":")[1])
            except Exception:
                gpu_id = 0
        self.estimator = DeepCREStereoEstimator(model_path=str(self.model_path), device_id=gpu_id)
        self.estimator.initialize()

        print(f"[SizeServer] Loading YOLO instance segmentation ({self.yolo_model_name})...")
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.yolo_model_name)
        except Exception as err:
            print(f"[SizeServer] [WARN] Could not load YOLO model: {err}. Running stereo-only fallback.")
            self.yolo_model = None

    def process_measurement(
        self,
        image_bytes: bytes,
        point_name: Optional[str] = None,
        hint_class: Optional[str] = None,
    ) -> MeasurementResult:
        t0 = time.perf_counter()

        if self.is_mock:
            return self._mock_result(point_name, hint_class)

        # 1. Decode composite image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        composite = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if composite is None:
            return MeasurementResult(ok=False, point=point_name, error_message="Could not decode image payload")

        h, w = composite.shape[:2]
        mid = w // 2
        left_raw = composite[:, :mid]
        right_raw = composite[:, mid:]

        # 2. Epipolar Rectification
        t_rect_start = time.perf_counter()
        left_rect, right_rect = self.calibration.rectify_pair(left_raw, right_raw)
        t_rect_ms = (time.perf_counter() - t_rect_start) * 1000.0

        # 3. Disparity Estimation
        t_disp_start = time.perf_counter()
        disparity = self.estimator.estimate_disparity(left_rect, right_rect)
        t_disp_ms = (time.perf_counter() - t_disp_start) * 1000.0

        # 4. Metric Depth
        focal_px = self.calibration.get_focal_length(left_rect.shape[1], left_rect.shape[0])
        depth_m = disparity_to_depth(
            disparity=disparity,
            focal_length_px=focal_px,
            baseline_m=self.calibration.baseline_m,
            min_depth=0.2,
            max_depth=6.0,
            resilient_voids=True,
        )

        # 5. YOLO Detection & Instance Mask Measurement
        t_yolo_start = time.perf_counter()
        detections: list[DetectionItem] = []

        if self.yolo_model:
            results = self.yolo_model.predict(left_rect, device=self.device, verbose=False)
            res = results[0] if results else None

            if res and res.masks and res.boxes:
                masks = res.masks.data.detach().cpu().numpy()
                cx = float(self.calibration.P1[0, 2]) if self.calibration.P1 is not None else (left_rect.shape[1] / 2.0)
                cy = float(self.calibration.P1[1, 2]) if self.calibration.P1 is not None else (left_rect.shape[0] / 2.0)
                fy = float(self.calibration.P1[1, 1]) if self.calibration.P1 is not None else focal_px

                for idx, mask in enumerate(masks):
                    mask_resized = cv2.resize(
                        (mask > 0.5).astype(np.uint8),
                        (left_rect.shape[1], left_rect.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                    dist_m, w_cm, h_cm, valid_pix = robust_measurement(
                        mask=mask_resized,
                        depth=depth_m,
                        fx=focal_px,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                    )
                    cls_id = int(res.boxes.cls[idx].item())
                    cls_name = str(res.names[cls_id])
                    conf = float(res.boxes.conf[idx].item())
                    box = [int(v) for v in res.boxes.xyxy[idx].detach().cpu().numpy()]

                    detections.append(
                        DetectionItem(
                            class_name=cls_name,
                            confidence=round(conf, 2),
                            bbox_px=box,
                            height_cm=h_cm,
                            width_cm=w_cm,
                            distance_m=dist_m,
                            valid_depth_pixels=valid_pix,
                        )
                    )

        t_yolo_ms = (time.perf_counter() - t_yolo_start) * 1000.0

        # Select primary detection
        primary = select_primary_detection(detections, hint_class=hint_class, image_shape=left_rect.shape)

        # 6. Annotate image and depth map
        annotated = left_rect.copy()
        for d in detections:
            is_prim = (primary is not None and d.bbox_px == primary.bbox_px)
            draw_detection_overlay(annotated, d, is_primary=is_prim)

        depth_colored = colorize_depth(depth_m, min_depth=0.2, max_depth=4.0)

        # Encode to JPEG
        ok_a, buf_a = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ok_d, buf_d = cv2.imencode(".jpg", depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 80])

        b64_ann = base64.b64encode(buf_a.tobytes()).decode("ascii") if ok_a else None
        b64_depth = base64.b64encode(buf_d.tobytes()).decode("ascii") if ok_d else None

        t_total_ms = (time.perf_counter() - t0) * 1000.0
        timing = {
            "rectify": round(t_rect_ms, 1),
            "stereo": round(t_disp_ms, 1),
            "yolo": round(t_yolo_ms, 1),
            "total": round(t_total_ms, 1),
        }

        return MeasurementResult(
            ok=True,
            point=point_name,
            primary=primary,
            detections=detections,
            annotated_jpeg_b64=b64_ann,
            depth_jpeg_b64=b64_depth,
            timing_ms=timing,
        )

    def _mock_result(self, point_name: Optional[str], hint_class: Optional[str]) -> MeasurementResult:
        det = DetectionItem(
            class_name=hint_class or "bottle",
            confidence=0.94,
            bbox_px=[400, 200, 600, 650],
            height_cm=28.5,
            width_cm=8.5,
            distance_m=0.88,
            valid_depth_pixels=15000,
        )
        return MeasurementResult(
            ok=True,
            point=point_name,
            primary=det,
            detections=[det],
            timing_ms={"rectify": 5.0, "stereo": 35.0, "yolo": 20.0, "total": 60.0},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 SSH GPU Size Estimation Server")
    default_calib = ssh_dir / "calibration" / "stereo_calibration_fixed.yml"
    default_model = ssh_dir / "models" / "crestereo_embedded.onnx"

    parser.add_argument("--port", type=int, default=DEFAULT_SSH_SIZE_PORT, help=f"Port to listen on (default: {DEFAULT_SSH_SIZE_PORT})")
    parser.add_argument("--calibration", type=Path, default=default_calib, help=f"Path to calibration YAML (default: {default_calib})")
    parser.add_argument("--model", type=Path, default=default_model, help=f"Path to CREStereo ONNX model (default: {default_model})")
    parser.add_argument("--yolo", type=str, default="yolo11x-seg.pt", help="YOLO segmentation model (default: yolo11x-seg.pt)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device (e.g. cuda:0 or cpu)")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without loading weights")
    args = parser.parse_args()

    print("=" * 70)
    print("  STAGE 2 SSH SIZE ESTIMATION SERVER")
    print(f"  Port: {args.port} | Device: {args.device} | Mock: {args.mock}")
    print("=" * 70)

    core = SizeServerCore(
        calib_path=args.calibration,
        model_path=args.model,
        yolo_path=args.yolo,
        device=args.device,
        is_mock=args.mock,
    )

    server = ThreadedHTTPServer(("0.0.0.0", args.port), SizeServerHandler)
    server.server_core = core  # type: ignore
    print(f"[Ready] Size Server listening at http://0.0.0.0:{args.port}...")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SizeServer] Stopping...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

