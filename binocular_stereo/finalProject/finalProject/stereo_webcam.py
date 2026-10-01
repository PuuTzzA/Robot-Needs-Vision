"""Live calibrated stereo capture with rectification and CREStereo.

The calibration file follows the format used by the linked Robot-Needs-Vision
project. Its translation vector is in millimetres, so this script converts the
baseline to metres before computing metric depth.

This is a first live prototype. It uses two ordinary USB camera streams, so
frames are approximately paired with ``grab``/``retrieve`` rather than being
hardware synchronized. The CREStereo checkpoint in this project is fixed at
320x240 (4:3); by default the rectified 16:9 frame is center-cropped to 4:3
before inference to avoid stretching the scene.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from stereo_prototype import (
    colorize_depth,
    colorize_disparity,
    depth_from_disparity,
    run_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="ONNX model; defaults to the embedded model when available.",
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
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum depth shown in the visualization, in metres.",
    )
    parser.add_argument(
        "--no-crop-to-4-3",
        action="store_true",
        help="Keep the full rectified frame; the 320x240 model then stretches 16:9 to 4:3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/stereo_webcam"),
    )
    parser.add_argument(
        "--use-coreml",
        action="store_true",
        help="Try CoreML first; automatically fall back to CPU if it fails.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="ONNX Runtime CPU intra-op threads (4 is a good M4 starting point).",
    )
    return parser.parse_args()


def load_calibration(path: Path) -> tuple[np.ndarray, ...]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {path}")

    matrices = []
    for key in ("K1", "D1", "K2", "D2", "R", "T"):
        matrix = storage.getNode(key).mat()
        if matrix is None:
            storage.release()
            raise ValueError(f"Calibration file is missing '{key}'")
        matrices.append(matrix.astype(np.float64))
    storage.release()

    K1, D1, K2, D2, R, T = matrices
    for name, K in (("K1", K1), ("K2", K2)):
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise ValueError(f"{name} is not a finite 3x3 matrix:\n{K}")
        if not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(
                f"{name} is not a standard OpenCV camera matrix; "
                f"expected last row [0, 0, 1], got {K[2]}"
            )
    if T.shape not in ((3, 1), (1, 3)):
        raise ValueError(f"T must be a 3-vector, got {T.shape}")
    return K1, D1, K2, D2, R, T.reshape(3, 1)


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = K.copy()
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {index}. Check macOS Camera permission "
            "and the camera index."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def center_crop_to_4_3(image: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = image.shape[:2]
    target_width = min(width, int(round(height * 4.0 / 3.0)))
    left = (width - target_width) // 2
    return image[:, left : left + target_width], left


def save_result(
    output_dir: Path,
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    depth: np.ndarray,
    max_depth: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    disparity_color = colorize_disparity(disparity)
    depth_color = colorize_depth(depth, max_depth)
    preview = np.hstack((left, disparity_color, depth_color))
    cv2.imwrite(str(output_dir / "left_rectified.png"), left)
    cv2.imwrite(str(output_dir / "right_rectified.png"), right)
    cv2.imwrite(str(output_dir / "disparity.png"), disparity_color)
    cv2.imwrite(str(output_dir / "depth.png"), depth_color)
    cv2.imwrite(str(output_dir / "preview.png"), preview)
    np.save(output_dir / "disparity.npy", disparity)
    np.save(output_dir / "depth_m.npy", depth)
    print(f"Saved one stereo result to {output_dir}")


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("cpu-threads must be positive")
    if args.model is None:
        args.model = Path("models/crestereo_embedded.onnx")
    if not args.model.exists():
        raise FileNotFoundError(f"Could not find ONNX model: {args.model}")

    K1, D1, K2, D2, R, T_mm = load_calibration(args.calibration)
    baseline_m = float(np.linalg.norm(T_mm)) / 1000.0
    if baseline_m <= 0:
        raise ValueError("Calibration translation has zero baseline")
    print(f"Calibration: {args.calibration}")
    print(f"Baseline: {baseline_m * 1000.0:.3f} mm")

    cam_left = open_camera(args.left_camera, args.width, args.height)
    cam_right = open_camera(args.right_camera, args.width, args.height)

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
        calibration_width, calibration_height = args.width, args.height
        sx = image_width / calibration_width
        sy = image_height / calibration_height
        if abs(sx - sy) > 0.01:
            print(
                "Warning: actual and calibrated aspect ratios differ; "
                "rectification may be inaccurate."
            )
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
        print(f"Camera frames: {image_width}x{image_height}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = args.cpu_threads
        session_options.inter_op_num_threads = 1

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if args.use_coreml and "CoreMLExecutionProvider" in available:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            coreml_cache = args.output_dir / "coreml_cache"
            coreml_cache.mkdir(parents=True, exist_ok=True)
            providers = [
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelCacheDirectory": str(coreml_cache),
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "CPUAndNeuralEngine",
                        "RequireStaticInputShapes": "1",
                    },
                ),
                "CPUExecutionProvider",
            ]
        print(f"ONNX Runtime providers: {available}")
        print(f"Using providers: {providers}")
        try:
            session = ort.InferenceSession(
                str(args.model), session_options, providers=providers
            )
        except Exception as error:
            if not args.use_coreml:
                raise
            print(f"CoreML session failed ({error}); falling back to CPU.")
            session = ort.InferenceSession(
                str(args.model), session_options, providers=["CPUExecutionProvider"]
            )
        print(f"Active ONNX Runtime providers: {session.get_providers()}")

        print("Press q to quit, s to save the current rectified pair and depth result.")
        last_time = time.perf_counter()
        while True:
            # grab() requests both frames before retrieve(), reducing timestamp skew
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

            start = time.perf_counter()
            disparity = run_model(session, crop_left, crop_right)
            depth = depth_from_disparity(disparity, float(P1[0, 0]), baseline_m)
            elapsed = time.perf_counter() - start

            disparity_color = colorize_disparity(disparity)
            depth_color = colorize_depth(depth, args.max_depth)
            display_left = cv2.resize(crop_left, (480, 360))
            display_disparity = cv2.resize(disparity_color, (480, 360))
            display_depth = cv2.resize(depth_color, (480, 360))
            preview = np.hstack((display_left, display_disparity, display_depth))
            cv2.putText(
                preview,
                f"inference {elapsed:.2f}s | baseline {baseline_m * 1000:.1f} mm",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("left rectified | disparity | depth", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_result(
                    args.output_dir,
                    crop_left,
                    crop_right,
                    disparity,
                    depth,
                    args.max_depth,
                )

            now = time.perf_counter()
            if now - last_time >= 1.0:
                valid = np.isfinite(depth) & (depth > 0)
                print(
                    f"Inference {elapsed:.2f}s; valid depth "
                    f"{valid.mean() * 100:.1f}%; "
                    f"median disparity {np.median(disparity[valid]):.2f}px; "
                    f"median depth {np.median(depth[valid]):.3f}m"
                )
                last_time = now
    finally:
        cam_left.release()
        cam_right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
