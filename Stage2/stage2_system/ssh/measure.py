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
    """Draw bounding box and 3D measurement label on image."""
    x1, y1, x2, y2 = detection.bbox_px
    color = (0, 255, 0) if is_primary else (200, 160, 40)
    thickness = 3 if is_primary else 1

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Label text
    h_str = f"H {detection.height_cm:.1f}cm" if detection.height_cm is not None else ""
    w_str = f"W {detection.width_cm:.1f}cm" if detection.width_cm is not None else ""
    z_str = f"Z {detection.distance_m:.2f}m" if detection.distance_m is not None else ""

    text = f"{detection.class_name} ({detection.confidence:.2f})"
    if h_str or w_str or z_str:
        text += f" | {h_str} {w_str} {z_str}".strip()

    # Draw label banner
    text_y = max(28, y1 - 10)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(image, (x1, text_y - th - 6), (x1 + tw + 8, text_y + 4), (15, 20, 25), -1)
    cv2.putText(image, text, (x1 + 4, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    return image

