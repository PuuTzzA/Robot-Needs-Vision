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

    (_, hdr_h), hdr_base = cv2.getTextSize(header, font, scale_hdr, thick_hdr)
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
    bar_w = max(5, int(w * 0.008))
    cv2.rectangle(out, (0, y0), (bar_w, y1), accent_bgr, -1)

    text_x = bar_w + pad_x
    hdr_y = y0 + pad_y + hdr_h
    main_y = hdr_y + hdr_base + gap_y + main_h

    cv2.putText(out, header, (text_x, hdr_y), font, scale_hdr, (200, 215, 230), thick_hdr, cv2.LINE_AA)
    cv2.putText(out, main_line, (text_x, main_y), font, scale_main, accent_bgr, thick_main, cv2.LINE_AA)
    return out


def draw_classification_overlay(
    frame: np.ndarray,
    title: str,
    category: str,
    color: Optional[str] = None,
) -> np.ndarray:
    """Overlay LLM classification result onto a copy of the captured image."""
    clean_cat = (category or "Unknown").strip()
    clean_color = (color or "").strip()
    if clean_color and clean_color.lower() not in ("none", "n/a", "unknown"):
        label = f"{clean_cat} ({clean_color})"
    else:
        label = clean_cat

    return _draw_banner_card(
        frame,
        header=f"{title} - Classification Result",
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
    """Draw bounding box and class/height tag when GPU server did not attach annotated JPEG."""
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

    color = (0, 230, 90) if is_primary else (0, 180, 255)
    thickness = max(2, int(round(h / 240.0)))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    h_part = f" {det.height_cm:.1f}cm" if det.height_cm is not None else ""
    tag = f"{det.class_name}{h_part} ({det.confidence:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = max(0.5, min(1.1, h / 720.0))
    f_thick = max(1, int(round(f_scale * 2.0)))
    (tw, th), base = cv2.getTextSize(tag, font, f_scale, f_thick)
    ty1 = max(th + base + 8, y1)
    cv2.rectangle(img, (x1, ty1 - th - base - 8), (min(w, x1 + tw + 12), ty1), color, -1)
    cv2.putText(img, tag, (x1 + 6, ty1 - base - 4), font, f_scale, (15, 18, 22), f_thick, cv2.LINE_AA)


def draw_measurement_overlay(
    raw_frame: np.ndarray,
    meas: MeasurementResult,
    point_name: Optional[str] = None,
) -> np.ndarray:
    """Build annotated preview frame for a stereo GPU MeasurementResult."""
    base_img: Optional[np.ndarray] = None
    if meas.annotated_jpeg_b64:
        try:
            jpg_bytes = base64.b64decode(meas.annotated_jpeg_b64)
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is not None and decoded.size > 0:
                base_img = decoded
        except Exception:
            base_img = None

    if base_img is None:
        base_img = raw_frame.copy()
        dets = list(meas.detections)
        if not dets and meas.primary is not None:
            dets = [meas.primary]
        for d in dets:
            is_prim = meas.primary is not None and d.bbox_px == meas.primary.bbox_px
            _draw_local_detection_box(base_img, d, is_primary=is_prim)

    prefix = f"Point {point_name}" if point_name else "Stereo GPU"
    if meas.ok and meas.primary is not None:
        p = meas.primary
        h_str = f"H: {p.height_cm:.1f} cm" if p.height_cm is not None else "H: n/a"
        w_str = f" | W: {p.width_cm:.1f} cm" if p.width_cm is not None else ""
        summary = f"{p.class_name} ({h_str}{w_str})"
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
        header=f"{prefix} - GPU Size & Class Estimate",
        main_line=summary,
        accent_bgr=accent,
        position="top",
    )

