"""YOLO-first live stereo measurement demo.

The displayed result is an Ultralytics instance-segmentation view. Each
detection label is augmented with the median optical-axis distance (Z) and a
robust visible width/height estimate computed from the stereo depth pixels
inside that object's mask.

The stereo stream is calibrated and rectified before both CREStereo and YOLO
see the left image. The two USB cameras are paired with grab()/retrieve(), so
this prototype assumes a mostly stationary scene.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from stereo_prototype import depth_from_disparity, run_model
from stereo_webcam import (
    center_crop_to_4_3,
    colorize_depth,
    colorize_disparity,
    load_calibration,
    open_camera,
    scale_intrinsics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stereo-model",
        type=Path,
        default=None,
        help="CREStereo ONNX model; defaults to the embedded model when available.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("yolo26n-seg.pt"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("stereo_calibration_fixed.yml"),
    )
    # On the current macOS setup, the calibrated pair is camera 0 (left) and
    # camera 1 (right). Keep both values configurable for another machine.
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--no-crop-to-4-3",
        action="store_true",
        help="Keep the full rectified frame; this stretches it for the 320x240 stereo model.",
    )
    parser.add_argument(
        "--use-coreml",
        action="store_true",
        help="Try CoreML for CREStereo, then fall back to CPU if needed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/stereo_yolo_webcam"),
    )
    return parser.parse_args()


def choose_yolo_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def format_length(value_m: float | None) -> str:
    if value_m is None or not np.isfinite(value_m):
        return "n/a"
    if value_m < 1.0:
        return f"{value_m * 100.0:.1f} cm"
    return f"{value_m:.2f} m"


def robust_measurement(
    mask: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float | None, float | None, float | None, int]:
    """Return median Z and robust visible X/Y extents for one instance mask."""

    valid = mask & np.isfinite(depth) & (depth > 0)
    if int(valid.sum()) < 40:
        return None, None, None, int(valid.sum())

    # Erode the mask slightly to reduce depth contamination along instance
    # boundaries. Keep tiny masks unchanged rather than eroding them away.
    if int(valid.sum()) >= 200:
        eroded = cv2.erode(
            valid.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        if int(eroded.sum()) >= 40:
            valid = eroded

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(np.float64)
    median_z = float(np.median(z))

    # Reject severe foreground/background contamination while retaining
    # moderate depth variation on a real object.
    q_low, q_high = np.percentile(z, [10, 90])
    keep = (z >= q_low) & (z <= q_high)
    if int(keep.sum()) < 20:
        keep = np.ones_like(z, dtype=bool)
    rows = rows[keep].astype(np.float64)
    cols = cols[keep].astype(np.float64)
    z = z[keep]

    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    width = float(np.percentile(x, 95) - np.percentile(x, 5))
    height = float(np.percentile(y, 95) - np.percentile(y, 5))
    return median_z, width, height, int(z.size)


def annotate_measurements(
    result,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    annotated = result.plot(conf=False, labels=False, boxes=True, masks=True)
    measurements: list[dict[str, object]] = []
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return annotated, measurements

    masks = result.masks.data.detach().cpu().numpy()
    for index, mask in enumerate(masks):
        mask = cv2.resize(
            (mask > 0.5).astype(np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        distance, width, height, valid_count = robust_measurement(
            mask, depth, fx, fy, cx, cy
        )
        class_id = int(result.boxes.cls[index].item())
        class_name = str(result.names[class_id])
        confidence = float(result.boxes.conf[index].item())
        box = result.boxes.xyxy[index].detach().cpu().numpy().astype(int)

        label = (
            f"{class_name} {confidence:.2f} | "
            f"Z {format_length(distance)} | "
            f"W {format_length(width)} H {format_length(height)}"
        )
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        text_y = max(24, y1 - 8)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        measurements.append(
            {
                "class": class_name,
                "confidence": confidence,
                "distance_m": distance,
                "width_m": width,
                "height_m": height,
                "valid_depth_pixels": valid_count,
            }
        )
    return annotated, measurements


def save_fusion_result(
    output_dir: Path,
    annotated: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    depth: np.ndarray,
    measurements: list[dict[str, object]],
    max_depth: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(str(output_dir / f"{stamp}_yolo_measurements.png"), annotated)
    cv2.imwrite(str(output_dir / f"{stamp}_left_rectified.png"), left)
    cv2.imwrite(str(output_dir / f"{stamp}_right_rectified.png"), right)
    cv2.imwrite(
        str(output_dir / f"{stamp}_disparity.png"), colorize_disparity(disparity)
    )
    cv2.imwrite(str(output_dir / f"{stamp}_depth.png"), colorize_depth(depth, max_depth))
    np.save(output_dir / f"{stamp}_disparity.npy", disparity)
    np.save(output_dir / f"{stamp}_depth_m.npy", depth)
    (output_dir / f"{stamp}_measurements.txt").write_text(
        "\n".join(str(item) for item in measurements) + "\n",
        encoding="utf-8",
    )
    print(f"Saved YOLO measurement result to {output_dir} ({stamp})")


def main() -> None:
    args = parse_args()
    if args.stereo_model is None:
        args.stereo_model = Path("models/crestereo_embedded.onnx")
    for path in (args.stereo_model, args.yolo_model, args.calibration):
        if not path.exists():
            raise FileNotFoundError(f"Could not find required file: {path}")
    if args.cpu_threads <= 0 or args.confidence <= 0 or args.confidence >= 1:
        raise ValueError("cpu-threads must be positive and confidence must be in (0, 1)")

    K1, D1, K2, D2, R, T_mm = load_calibration(args.calibration)
    baseline_m = float(np.linalg.norm(T_mm)) / 1000.0
    print(f"Calibration: {args.calibration}")
    print(f"Baseline: {baseline_m * 1000.0:.3f} mm")

    cam_left = open_camera(args.left_camera, args.width, args.height)
    cam_right = open_camera(args.right_camera, args.width, args.height)
    yolo_device = choose_yolo_device()
    print(f"Loading YOLO {args.yolo_model} on {yolo_device}...")
    yolo = YOLO(str(args.yolo_model))

    try:
        ok_left, first_left = cam_left.read()
        ok_right, first_right = cam_right.read()
        if not ok_left or not ok_right or first_left is None or first_right is None:
            raise RuntimeError("Both cameras opened, but an initial frame could not be read")
        if first_left.shape[:2] != first_right.shape[:2]:
            raise RuntimeError(
                f"Camera resolutions differ: {first_left.shape[:2]} vs {first_right.shape[:2]}"
            )

        image_height, image_width = first_left.shape[:2]
        sx = image_width / args.width
        sy = image_height / args.height
        K1_capture = scale_intrinsics(K1, sx, sy)
        K2_capture = scale_intrinsics(K2, sx, sy)
        image_size = (image_width, image_height)
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            K1_capture,
            D1,
            K2_capture,
            D2,
            image_size,
            R,
            T_mm,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            newImageSize=image_size,
        )
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            K1_capture, D1, R1, P1, image_size, cv2.CV_32FC1
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            K2_capture, D2, R2, P2, image_size, cv2.CV_32FC1
        )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = args.cpu_threads
        session_options.inter_op_num_threads = 1
        providers: list[object] = ["CPUExecutionProvider"]
        if args.use_coreml and "CoreMLExecutionProvider" in ort.get_available_providers():
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        print(f"Using stereo providers: {providers}")
        session = ort.InferenceSession(
            str(args.stereo_model), session_options, providers=providers
        )
        print(f"Active stereo providers: {session.get_providers()}")

        print("Press q/Esc to quit; press s to save the current YOLO measurement result.")
        while True:
            cam_left.grab()
            cam_right.grab()
            ok_left, frame_left = cam_left.retrieve()
            ok_right, frame_right = cam_right.retrieve()
            if not ok_left or not ok_right or frame_left is None or frame_right is None:
                continue

            rect_left = cv2.remap(frame_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
            rect_right = cv2.remap(frame_right, map_right_x, map_right_y, cv2.INTER_LINEAR)
            crop_left = rect_left
            crop_right = rect_right
            crop_x = 0
            if not args.no_crop_to_4_3:
                crop_left, crop_x = center_crop_to_4_3(rect_left)
                crop_right = rect_right[:, crop_x : crop_x + crop_left.shape[1]]

            stereo_start = time.perf_counter()
            disparity = run_model(session, crop_left, crop_right)
            depth = depth_from_disparity(disparity, float(P1[0, 0]), baseline_m)
            result = yolo.predict(
                source=crop_left,
                conf=args.confidence,
                retina_masks=True,
                device=yolo_device,
                verbose=False,
            )[0]
            # Crop shifts the left principal point but does not change fx/fy.
            fx = float(P1[0, 0])
            fy = float(P1[1, 1])
            cx = float(P1[0, 2] - crop_x)
            cy = float(P1[1, 2])
            annotated, measurements = annotate_measurements(
                result, depth, fx, fy, cx, cy
            )
            elapsed = time.perf_counter() - stereo_start
            cv2.putText(
                annotated,
                f"stereo+YOLO {elapsed:.2f}s | objects {len(measurements)}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLO classification + mask + distance + size", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_fusion_result(
                    args.output_dir,
                    annotated,
                    crop_left,
                    crop_right,
                    disparity,
                    depth,
                    measurements,
                    args.max_depth,
                )
    finally:
        cam_left.release()
        cam_right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
