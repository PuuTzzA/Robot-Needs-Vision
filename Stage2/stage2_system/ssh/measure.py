"""Robust 3D metric size estimation and object detection utilities."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import cv2
import numpy as np

from protocol import DetectionItem


def average_center_distance(
    depth: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    mask: Optional[np.ndarray] = None,
    center_fraction: float = 0.3,
    min_radius: int = 4,
) -> Optional[float]:
    """Calculate distance from camera by averaging over the center pixels of the detected object.

    Args:
        depth: 2D array of metric depth in metres.
        bbox: Bounding box [x1, y1, x2, y2].
        mask: Optional boolean segmentation mask of the detected object.
        center_fraction: Proportion of width/height around object center to sample.
        min_radius: Minimum pixel radius around center to sample.

    Returns:
        Average distance in metres, or None if no valid center depth pixels exist.
    """
    if depth is None or depth.size == 0 or len(bbox) < 4:
        return None

    h, w = depth.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx = max(min_radius, int((x2 - x1) * center_fraction * 0.5))
    ry = max(min_radius, int((y2 - y1) * center_fraction * 0.5))

    y_min, y_max = max(0, cy - ry), min(h, cy + ry + 1)
    x_min, x_max = max(0, cx - rx), min(w, cx + rx + 1)

    roi_depth = depth[y_min:y_max, x_min:x_max]
    valid = np.isfinite(roi_depth) & (roi_depth > 0.1) & (roi_depth < 8.0)

    if mask is not None:
        roi_mask = mask[y_min:y_max, x_min:x_max]
        masked_valid = valid & roi_mask
        if np.any(masked_valid):
            valid = masked_valid

    pixels = roi_depth[valid]
    if pixels.size == 0:
        return None

    return float(np.mean(pixels))


def robust_measurement(
    mask: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    bbox: Optional[list[int] | tuple[int, int, int, int]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """Calculate average center distance, robust metric width, and height in cm for an instance mask.

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

    # Average distance over center pixels of the detected object
    target_box = bbox
    if target_box is None and cols.size > 0 and rows.size > 0:
        target_box = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]

    center_dist = average_center_distance(depth, target_box, mask=valid) if target_box else None
    if center_dist is not None:
        distance_m = round(center_dist, 3)
    else:
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

    # Class label text above the bounding box
    text = detection.class_name.strip().capitalize() if detection.class_name else "Object"

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

