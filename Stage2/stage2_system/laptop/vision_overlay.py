"""OpenCV overlay rendering helpers for 'Last Processed Frame' in Mission Control GUI.

Renders crisp, resolution-adaptive classification and stereo GPU measurement
overlays onto captured camera frames so results remain legible when downscaled
to the 440x260 preview canvas.
"""

from __future__ import annotations

import base64
from typing import Optional
import cv2
import numpy as np

from shared.protocol import DetectionItem, MeasurementResult
from .llm.schemas import is_none_category


def _draw_banner_card(
    img: np.ndarray,
    header: str,
    main_line: str,
    accent_bgr: tuple[int, int, int] = (80, 220, 100),
    position: str = "bottom",
) -> np.ndarray:
    """Render a high-contrast semi-transparent banner with two text lines."""
    out = img.copy()
    h, w = out.shape[:2]
    if h < 40 or w < 60:
        return out

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_hdr = max(0.45, min(1.3, h / 780.0))
    scale_main = max(0.55, min(1.65, h / 560.0))
    thick_hdr = max(1, int(round(scale_hdr * 2.0)))
    thick_main = max(2, int(round(scale_main * 2.4)))

    pad_x = max(10, int(w * 0.02))
    pad_y = max(8, int(h * 0.018))
    gap_y = max(6, int(h * 0.012))
    bar_w = max(5, int(w * 0.008))
    max_text_w = max(40, w - bar_w - pad_x * 2)

    (hdr_w, hdr_h), hdr_base = cv2.getTextSize(header, font, scale_hdr, thick_hdr)
    if hdr_w > max_text_w:
        scale_hdr = max(0.35, scale_hdr * (max_text_w / float(hdr_w)))
        thick_hdr = max(1, int(round(scale_hdr * 2.0)))
        (_, hdr_h), hdr_base = cv2.getTextSize(header, font, scale_hdr, thick_hdr)

    (main_w, main_h), main_base = cv2.getTextSize(main_line, font, scale_main, thick_main)
    if main_w > max_text_w:
        scale_main = max(0.40, scale_main * (max_text_w / float(main_w)))
        thick_main = max(1, int(round(scale_main * 2.4)))
        (_, main_h), main_base = cv2.getTextSize(main_line, font, scale_main, thick_main)

    box_h = pad_y * 2 + hdr_h + hdr_base + gap_y + main_h + main_base
    box_h = min(h, box_h)

    if position == "top":
        y0, y1 = 0, box_h
    else:
        y0, y1 = max(0, h - box_h), h

    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0), (w, y1), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.76, out, 0.24, 0, out)

    # Accent bar
    cv2.rectangle(out, (0, y0), (bar_w, y1), accent_bgr, -1)

    text_x = bar_w + pad_x
    hdr_y = y0 + pad_y + hdr_h
    main_y = hdr_y + hdr_base + gap_y + main_h

    cv2.putText(out, header, (text_x, hdr_y), font, scale_hdr, (200, 215, 230), thick_hdr, cv2.LINE_AA)
    cv2.putText(out, main_line, (text_x, main_y), font, scale_main, accent_bgr, thick_main, cv2.LINE_AA)
    return out


def format_height_cm(height_cm: Optional[float]) -> Optional[str]:
    """Format a positive height in centimeters cleanly (e.g. 11.0 -> '11 cm', 17.5 -> '17.5 cm')."""
    if height_cm is None:
        return None
    try:
        val = float(height_cm)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val) or val <= 0:
        return None
    return f"{round(val, 1):g} cm"


def draw_classification_overlay(
    frame: np.ndarray,
    title: str,
    category: str,
    color: Optional[str] = None,
    height_cm: Optional[float] = None,
) -> np.ndarray:
    """Overlay LLM classification (and optional height) result onto a copy of the captured image."""
    raw_cat = (category or "").strip()
    if raw_cat and is_none_category(raw_cat):
        return _draw_banner_card(
            frame,
            header=f"{title} - Classification Result",
            main_line="None (Empty Box)",
            accent_bgr=(60, 200, 255),
            position="bottom",
        )

    clean_cat = raw_cat or "Unknown"
    clean_color = (color or "").strip()
    h_str = format_height_cm(height_cm)

    has_color = bool(clean_color and clean_color.lower() not in ("none", "n/a", "unknown"))
    if has_color and h_str:
        label = f"{clean_cat} ({clean_color}) - {h_str}"
    elif h_str:
        label = f"{clean_cat} - {h_str}"
    elif has_color:
        label = f"{clean_cat} ({clean_color})"
    else:
        label = clean_cat

    header = f"{title} - Classification & Height" if h_str else f"{title} - Classification Result"

    return _draw_banner_card(
        frame,
        header=header,
        main_line=label,
        accent_bgr=(90, 235, 130),
        position="bottom",
    )


def draw_text_result_overlay(
    frame: np.ndarray,
    title: str,
    result_text: str,
) -> np.ndarray:
    """Overlay free-form analysis summary onto a copy of a captured frame."""
    return _draw_banner_card(
        frame,
        header=title,
        main_line=result_text.strip(),
        accent_bgr=(90, 215, 250),
        position="bottom",
    )


def _draw_local_detection_box(img: np.ndarray, det: DetectionItem, is_primary: bool) -> None:
    """Draw high-contrast bounding box and class label directly above the box."""
    if len(det.bbox_px) != 4:
        return
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in det.bbox_px]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return

    color = (90, 255, 130) if is_primary else (0, 215, 255)
    thickness = max(2, int(round(h / 230.0)))
    # Double-stroke outline so bounding box pops cleanly over Turbo/Magma depth colormaps
    cv2.rectangle(img, (x1, y1), (x2, y2), (12, 16, 22), thickness + 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Class label directly above the bounding box
    tag = det.class_name.strip().capitalize() if det.class_name else "Object"

    font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = max(0.5, min(1.05, h / 700.0))
    f_thick = max(1, int(round(f_scale * 2.0)))
    (tw, th), base = cv2.getTextSize(tag, font, f_scale, f_thick)
    pad_h = th + base + 10
    pad_w = tw + 12

    bx1 = max(4, min(max(4, w - pad_w - 4), x1))
    by2 = y1 - 4
    if by2 - pad_h < 4:
        by1 = 4
        by2 = min(h - 4, 4 + pad_h)
    else:
        by1 = by2 - pad_h

    cv2.rectangle(img, (bx1, by1), (min(w - 2, bx1 + pad_w), by2), (15, 18, 24), -1)
    cv2.rectangle(img, (bx1, by1), (min(w - 2, bx1 + pad_w), by2), color, 1)
    cv2.putText(img, tag, (bx1 + 6, by2 - base - 4), font, f_scale, color, f_thick, cv2.LINE_AA)


def _decode_b64_jpeg(b64_str: Optional[str]) -> Optional[np.ndarray]:
    if not b64_str:
        return None
    try:
        jpg_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is not None and decoded.size > 0:
            return decoded
    except Exception:
        pass
    return None


def draw_measurement_overlay(
    raw_frame: np.ndarray,
    meas: MeasurementResult,
    point_name: Optional[str] = None,
) -> np.ndarray:
    """Build annotated disparity map preview frame with bounding box and class label composited over it."""
    depth_img = _decode_b64_jpeg(meas.depth_jpeg_b64)
    ann_img = _decode_b64_jpeg(meas.annotated_jpeg_b64) if depth_img is None else None

    need_local_boxes = False
    if depth_img is not None:
        base_img = depth_img
        need_local_boxes = True
    elif ann_img is not None:
        base_img = ann_img
        need_local_boxes = False
    elif meas.ok and raw_frame.size > 0:
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if raw_frame.ndim == 3 else raw_frame
        base_img = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        need_local_boxes = True
    else:
        base_img = raw_frame.copy()
        need_local_boxes = True

    if need_local_boxes:
        dets = list(meas.detections)
        if not dets and meas.primary is not None:
            dets = [meas.primary]
        for d in dets:
            is_prim = meas.primary is not None and d.bbox_px == meas.primary.bbox_px
            _draw_local_detection_box(base_img, d, is_primary=is_prim)

    if point_name:
        header = f"Point {point_name}: Estimated Depth and Class"
    else:
        header = "Stereo GPU: Estimated Depth and Class"

    if meas.ok and meas.primary is not None:
        p = meas.primary
        cls_title = p.class_name.strip().capitalize() if p.class_name else "Object"
        if p.distance_m is not None:
            summary = f"{cls_title} ({p.distance_m:.1f}m)"
        else:
            summary = cls_title
        accent = (80, 235, 130)
    elif meas.ok:
        summary = "No object detected"
        accent = (60, 200, 255)
    else:
        err = (meas.error_message or "measurement failed").strip()
        if len(err) > 48:
            err = err[:45] + "..."
        summary = f"Failed: {err}"
        accent = (90, 110, 255)

    return _draw_banner_card(
        base_img,
        header=header,
        main_line=summary,
        accent_bgr=accent,
        position="bottom",
    )

