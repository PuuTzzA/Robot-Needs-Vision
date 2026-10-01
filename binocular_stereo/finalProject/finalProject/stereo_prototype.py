"""Minimal offline CREStereo prototype.

This script intentionally works on one saved, rectified stereo pair first. It
does not capture cameras or perform calibration. The model produces disparity;
the baseline and focal length convert disparity to metric depth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument(
        "--baseline",
        required=True,
        type=float,
        help="Camera baseline in metres.",
    )
    parser.add_argument(
        "--focal-length",
        required=True,
        type=float,
        help="Rectified focal length in pixels (usually fx from calibration).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/stereo_prototype"),
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum depth shown in the visualization, in metres.",
    )
    parser.add_argument(
        "--use-coreml",
        action="store_true",
        help="Try ONNX Runtime's CoreML provider when it is installed.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a preview window after inference.",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def model_dimension(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"The model has a dynamic {name} dimension. "
            "This prototype expects a fixed-size CREStereo ONNX model."
        )
    return int(value)


def prepare_input(image: np.ndarray, height: int, width: int) -> np.ndarray:
    # This CREStereo export follows the public example used by the supplied
    # camera project: RGB images, normalized to [0, 1].  OpenCV captures BGR,
    # so convert the channel order before creating the NCHW tensor.
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = resized.transpose(2, 0, 1)[None]
    return (tensor.astype(np.float32) / 255.0).copy()


def run_model(
    session: ort.InferenceSession,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    inputs = session.get_inputs()
    if len(inputs) not in (2, 4):
        raise ValueError(
            f"Expected a 2-input or 4-input CREStereo model, "
            f"found {len(inputs)} inputs"
        )

    # Different CREStereo exports order the full-resolution and half-
    # resolution inputs differently. Use each input's name and shape instead
    # of assuming a particular ordering.
    feed = {}
    for index, item in enumerate(inputs):
        height = model_dimension(item.shape[2], "height")
        width = model_dimension(item.shape[3], "width")
        name = item.name.lower()
        if "left" in name:
            source = left
        elif "right" in name:
            source = right
        else:
            source = left if index % 2 == 0 else right
        feed[item.name] = prepare_input(source, height, width)

    output = session.run(None, feed)[0]
    disparity = np.squeeze(output).astype(np.float32)
    if disparity.ndim != 2:
        raise ValueError(f"Expected a 2D disparity output, got {disparity.shape}")

    original_height, original_width = left.shape[:2]
    output_height, output_width = disparity.shape
    disparity = cv2.resize(
        disparity,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Disparity is measured in model-input pixels. Resizing it to the original
    # image requires the same horizontal scale factor.
    disparity *= original_width / output_width
    return disparity


def depth_from_disparity(
    disparity: np.ndarray,
    focal_length: float,
    baseline: float,
) -> np.ndarray:
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 1e-6)
    depth[valid] = focal_length * baseline / disparity[valid]
    return depth


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disparity) & (disparity > 0)
    if not np.any(valid):
        return np.zeros((*disparity.shape, 3), dtype=np.uint8)

    low, high = np.percentile(disparity[valid], [2, 98])
    high = max(high, low + 1e-6)
    normalized = np.zeros(disparity.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        255 * (disparity[valid] - low) / (high - low), 0, 255
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)


def colorize_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        255 * (1 - depth[valid] / max_depth), 0, 255
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)


def main() -> None:
    args = parse_args()
    if args.baseline <= 0 or args.focal_length <= 0:
        raise ValueError("baseline and focal-length must be positive")

    left = load_image(args.left)
    right = load_image(args.right)
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(
            f"Left/right image sizes differ: {left.shape[:2]} vs {right.shape[:2]}"
        )

    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if args.use_coreml and "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    print(f"ONNX Runtime providers: {available}")
    print(f"Using providers: {providers}")
    try:
        session = ort.InferenceSession(str(args.model), providers=providers)
    except Exception as error:
        if not args.use_coreml:
            raise
        print(f"CoreML session failed ({error}); falling back to CPU.")
        session = ort.InferenceSession(
            str(args.model), providers=["CPUExecutionProvider"]
        )
    disparity = run_model(session, left, right)
    depth = depth_from_disparity(disparity, args.focal_length, args.baseline)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "disparity.npy", disparity)
    np.save(args.output_dir / "depth_m.npy", depth)

    disparity_color = colorize_disparity(disparity)
    depth_color = colorize_depth(depth, args.max_depth)
    preview = np.hstack((left, disparity_color, depth_color))
    cv2.imwrite(str(args.output_dir / "disparity.png"), disparity_color)
    cv2.imwrite(str(args.output_dir / "depth.png"), depth_color)
    cv2.imwrite(str(args.output_dir / "preview.png"), preview)

    valid_depth = depth[np.isfinite(depth) & (depth > 0) & (depth <= args.max_depth)]
    if valid_depth.size:
        print(
            f"Valid depth: {valid_depth.size}/{depth.size} pixels; "
            f"median={np.median(valid_depth):.3f} m"
        )
    else:
        print("No valid depth values were produced.")
    print(f"Saved outputs to {args.output_dir}")

    if args.show:
        cv2.imshow("left | disparity | depth", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
