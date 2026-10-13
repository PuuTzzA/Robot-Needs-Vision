"""Depth calculation, metric colorization, legend generation, and measurement metrics."""

from __future__ import annotations
import cv2
import numpy as np


def disparity_to_depth(
    disparity: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    min_disparity: float = 0.5,
    min_depth: float = 0.2,
    max_depth: float = 10.0,
) -> np.ndarray:
    """Convert a 2D disparity map (in pixels) into absolute metric depth (in metres).

    Formula: Z = (fx * B) / d
    """
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity >= min_disparity)
    
    calculated = np.zeros_like(disparity)
    calculated[valid] = (focal_length_px * baseline_m) / disparity[valid]
    
    # Filter out physical depth outliers
    depth_valid = valid & (calculated >= min_depth) & (calculated <= max_depth)
    depth[depth_valid] = calculated[depth_valid]
    return depth


def colorize_disparity(
    disparity: np.ndarray,
    colormap: int = cv2.COLORMAP_MAGMA,
) -> np.ndarray:
    """Colorize a disparity map with invalid pixels shown in black."""
    vis = np.zeros((*disparity.shape, 3), dtype=np.uint8)
    valid = np.isfinite(disparity) & (disparity > 0)
    if not np.any(valid):
        return vis

    low, high = np.percentile(disparity[valid], [2, 98])
    high = max(high, low + 1e-4)

    normalized = np.zeros(disparity.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        255.0 * (disparity[valid] - low) / (high - low), 0, 255
    ).astype(np.uint8)

    colored = cv2.applyColorMap(normalized, colormap)
    vis[valid] = colored[valid]
    return vis


def colorize_depth(
    depth: np.ndarray,
    min_depth: float = 0.4,
    max_depth: float = 4.0,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Colorize absolute depth using Turbo colormap (warm=near, cool=far).
    
    Invalid pixels (NaN or out of bounds) are shown in solid black.
    """
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    if not np.any(valid):
        return vis

    # Map [min_depth, max_depth] linearly to [0, 255]
    # 0 = close (red/warm in Turbo inverted, or blue in standard Turbo)
    # Using standard Turbo: 0 = dark blue (close), 255 = dark red (far),
    # or invert so 0 = red (near), 255 = blue (far).
    # In robotics, warm colors (red/orange) = near, cool colors (blue/cyan) = far is conventional.
    clipped = np.clip(depth[valid], min_depth, max_depth)
    ratio = (clipped - min_depth) / (max_depth - min_depth)
    scaled_uint8 = np.clip(ratio * 255.0, 0, 255).astype(np.uint8)

    colored = cv2.applyColorMap(scaled_uint8, colormap)
    vis[valid] = colored
    return vis


def attach_depth_legend(
    depth_vis: np.ndarray,
    min_depth: float = 0.4,
    max_depth: float = 4.0,
    legend_width: int = 80,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Renders a vertical metric depth colorbar legend and appends it to the right of depth_vis.

    Displays tick marks and distance text (e.g., '0.5m', '1.0m', '2.0m', '4.0m').
    """
    h, w = depth_vis.shape[:2]
    legend_bar = np.full((h, legend_width, 3), 35, dtype=np.uint8)  # dark background

    # Margin inside legend
    bar_x = 12
    bar_w = 20
    top_y = 30
    bot_y = h - 30
    bar_h = bot_y - top_y

    if bar_h > 20:
        # Create gradient from top (near = min_depth) to bottom (far = max_depth)
        gradient = np.linspace(0, 255, bar_h, dtype=np.uint8)[:, None]
        gradient_3ch = np.repeat(gradient, bar_w, axis=1)
        colored_gradient = cv2.applyColorMap(gradient_3ch, colormap)
        legend_bar[top_y:bot_y, bar_x : bar_x + bar_w] = colored_gradient

        # Draw border around the gradient bar
        cv2.rectangle(
            legend_bar,
            (bar_x - 1, top_y - 1),
            (bar_x + bar_w, bot_y),
            (200, 200, 200),
            1,
        )

        # Draw tick marks and metric labels
        tick_values = [min_depth]
        step = (max_depth - min_depth) / 4.0
        for i in range(1, 4):
            tick_values.append(round(min_depth + i * step, 1))
        tick_values.append(max_depth)

        for val in tick_values:
            fraction = (val - min_depth) / (max_depth - min_depth)
            y_pos = int(top_y + fraction * bar_h)

            # Draw tick line
            cv2.line(
                legend_bar,
                (bar_x + bar_w, y_pos),
                (bar_x + bar_w + 4, y_pos),
                (255, 255, 255),
                1,
            )
            # Label
            label = f"{val:.1f}m"
            cv2.putText(
                legend_bar,
                label,
                (bar_x + bar_w + 7, y_pos + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Title at the top
        cv2.putText(
            legend_bar,
            "Depth (m)",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return np.hstack((depth_vis, legend_bar))


def calculate_compactness(depth_or_disparity: np.ndarray) -> float:
    """Compute the compactness (fill rate / density percentage) of valid depth pixels.

    Formula: (valid_pixels / total_pixels) * 100.0
    """
    total = depth_or_disparity.size
    if total == 0:
        return 0.0
    valid = np.count_nonzero(np.isfinite(depth_or_disparity) & (depth_or_disparity > 0))
    return float(valid / total) * 100.0


def sample_roi_depth(
    depth_map: np.ndarray,
    center_x: int,
    center_y: int,
    box_radius: int = 12,
) -> tuple[float, float, float]:
    """Sample depth over a local window centered at (center_x, center_y).

    Returns:
        median_z: Median depth in metres (NaN if no valid pixels).
        std_z: Standard deviation in metres.
        valid_ratio: Proportion of valid pixels in the ROI (0.0 to 1.0).
    """
    h, w = depth_map.shape[:2]
    x1 = max(0, center_x - box_radius)
    x2 = min(w, center_x + box_radius + 1)
    y1 = max(0, center_y - box_radius)
    y2 = min(h, center_y + box_radius + 1)

    roi = depth_map[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    total_pixels = roi.size

    if len(valid) == 0:
        return float("nan"), float("nan"), 0.0

    median_z = float(np.median(valid))
    std_z = float(np.std(valid))
    valid_ratio = float(len(valid) / total_pixels)
    return median_z, std_z, valid_ratio

