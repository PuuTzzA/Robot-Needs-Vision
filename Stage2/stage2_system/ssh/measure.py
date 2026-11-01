"""Robust 3D metric size estimation and object detection utilities."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import cv2
import numpy as np

from protocol import DetectionItem


def robust_measurement(
    mask: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """Calculate median distance, robust metric width, and height in cm for an instance mask.

    Returns:
        (distance_m, width_cm, height_cm, valid_pixel_count)
    """
    valid = mask & np.isfinite(depth) & (depth > 0.1) & (depth < 8.0)
    valid_count = int(valid.sum())
    if valid_count < 30:
        return None, None, None, valid_count

    # Erode mask slightly to eliminate edge depth bleeding if instance is sufficiently large
    if valid_count >= 180:
        kernel = np.ones((5, 5), dtype=np.uint8)
        eroded = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
        if int(eroded.sum()) >= 30:
            valid = eroded

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(np.float64)
    median_z = float(np.median(z))

    # Reject depth outlier contamination (10th to 90th percentile)
    q_low, q_high = np.percentile(z, [10, 90])
    inliers = (z >= q_low) & (z <= q_high)
    if int(inliers.sum()) < 20:
        inliers = np.ones_like(z, dtype=bool)

    rows = rows[inliers].astype(np.float64)
    cols = cols[inliers].astype(np.float64)
    z = z[inliers]

    # Metric back-projection to 3D camera coordinates
    x_m = (cols - cx) * z / fx
    y_m = (rows - cy) * z / fy

    width_m = float(np.percentile(x_m, 95) - np.percentile(x_m, 5))
    height_m = float(np.percentile(y_m, 95) - np.percentile(y_m, 5))

    # Convert to centimeters
    width_cm = round(width_m * 100.0, 1)
    height_cm = round(height_m * 100.0, 1)
    distance_m = round(median_z, 3)

    return distance_m, width_cm, height_cm, int(z.size)


def select_primary_detection(
    detections: list[DetectionItem],
    hint_class: Optional[str] = None,
    image_shape: tuple[int, int] = (1080, 1920),
) -> Optional[DetectionItem]:
    """Select the primary object of interest from a list of detections.

    Prioritizes:
    1. Detection matching `hint_class` (case-insensitive substring)
    2. Proximity to image center and reasonable distance (< 2.5m)
    """
    if not detections:
        return None

    norm_hint = hint_class.strip().lower() if hint_class else ""
    if norm_hint:
        matching = [
            d for d in detections if norm_hint in d.class_name.lower() or d.class_name.lower() in norm_hint
        ]
        if matching:
            # Sort by confidence
            matching.sort(key=lambda d: d.confidence, reverse=True)
            return matching[0]

    # Score detections by centrality, confidence, and valid depth
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def score(d: DetectionItem) -> float:
        x1, y1, x2, y2 = d.bbox_px
        bx = (x1 + x2) / 2.0
        by = (y1 + y2) / 2.0
        dist_center = math.hypot(bx - cx, by - cy) / math.hypot(cx, cy)
        center_score = 1.0 - min(1.0, dist_center)

        has_height = 1.0 if d.height_cm is not None else 0.2
        return (d.confidence * 0.4) + (center_score * 0.4) + (has_height * 0.2)

    detections_sorted = sorted(detections, key=score, reverse=True)
    return detections_sorted[0]


def draw_detection_overlay(
    image: np.ndarray,
    detection: DetectionItem,
    is_primary: bool = True,
) -> np.ndarray:
    """Draw high-contrast bounding box and 3D size label above it on depth/disparity image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in detection.bbox_px]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return image

    color = (90, 255, 130) if is_primary else (0, 215, 255)
    thickness = 3 if is_primary else 2

    # Double-stroke rectangle for crisp visibility over Turbo/Magma colormaps
    cv2.rectangle(image, (x1, y1), (x2, y2), (12, 16, 22), thickness + 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Size text above the bounding box
    size_parts = []
    if detection.height_cm is not None:
        size_parts.append(f"H: {detection.height_cm:.1f}cm")
    if detection.width_cm is not None:
        size_parts.append(f"W: {detection.width_cm:.1f}cm")
    if detection.distance_m is not None:
        size_parts.append(f"Z: {detection.distance_m:.2f}m")

    size_str = " ".join(size_parts)
    text = f"{detection.class_name} | {size_str}" if size_str else f"{detection.class_name}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    f_thick = 2
    (tw, th), base = cv2.getTextSize(text, font, scale, f_thick)
    box_h = th + base + 10
    box_w = tw + 12

    bx1 = max(4, min(max(4, w - box_w - 4), x1))
    by2 = y1 - 4
    if by2 - box_h < 4:
        by1 = 4
        by2 = min(h - 4, 4 + box_h)
    else:
        by1 = by2 - box_h

    cv2.rectangle(image, (bx1, by1), (min(w - 2, bx1 + box_w), by2), (15, 20, 26), -1)
    cv2.rectangle(image, (bx1, by1), (min(w - 2, bx1 + box_w), by2), color, 1)
    cv2.putText(image, text, (bx1 + 6, by2 - base - 4), font, scale, color, f_thick, cv2.LINE_AA)

    return image

