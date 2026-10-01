"""Capture one labeled frame from each requested macOS camera index.

Use this before stereo inference to identify the two physical cameras that
were used during calibration.  The generated images make it easy to check
that the selected pair sees the same scene with substantial overlap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Camera indices to probe (default: 0 1 2 3).",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/camera_pair_check"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving camera snapshots to {args.output_dir}")
    for index in args.indices:
        camera = cv2.VideoCapture(index)
        if not camera.isOpened():
            print(f"camera {index}: unavailable")
            camera.release()
            continue
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        ok, frame = camera.read()
        actual = tuple(frame.shape[:2][::-1]) if ok and frame is not None else None
        if ok and frame is not None:
            path = args.output_dir / f"camera_{index}.png"
            cv2.imwrite(str(path), frame)
            print(f"camera {index}: {actual[0]}x{actual[1]} -> {path}")
        else:
            print(f"camera {index}: opened but no frame")
        camera.release()


if __name__ == "__main__":
    main()
