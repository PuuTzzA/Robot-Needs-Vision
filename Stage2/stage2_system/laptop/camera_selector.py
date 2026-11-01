"""Interactive stereo camera discovery, live annotated preview, and config persistence.

Before Mission Control launches, probes connected USB/integrated cameras, opens a
live OpenCV stream of every available camera annotated with its device ID and
saved LEFT / RIGHT assignment, lets the user press Enter to confirm or type e.g.
`left 0, right 2` to reassign, and persists the choice for future runs.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import sys
import threading
import time
from typing import Optional
import cv2
import numpy as np

from .camera import get_backend

DEFAULT_LEFT_CAM = 0
DEFAULT_RIGHT_CAM = 2
DEFAULT_CONFIG_FILENAME = ".camera_config.json"
PREVIEW_WINDOW_TITLE = "Stage 2 Camera Selection (Live Preview)"


class SuppressStderr:
    """Temporarily silence OpenCV C++ backend warnings when probing unused camera indices."""

    def __enter__(self) -> SuppressStderr:
        self._devnull: Optional[int] = None
        self._orig_stderr: Optional[int] = None
        try:
            self._devnull = os.open(os.devnull, os.O_RDWR)
            self._orig_stderr = os.dup(2)
            os.dup2(self._devnull, 2)
        except Exception:
            self._close_handles()
        return self

    def __exit__(self, *args) -> None:
        if self._orig_stderr is not None:
            try:
                os.dup2(self._orig_stderr, 2)
            except Exception:
                pass
        self._close_handles()

    def _close_handles(self) -> None:
        for fd in (self._orig_stderr, self._devnull):
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
        self._orig_stderr = None
        self._devnull = None


def load_camera_config(
    config_path: Path,
    default_left: int = DEFAULT_LEFT_CAM,
    default_right: int = DEFAULT_RIGHT_CAM,
) -> tuple[int, int]:
    """Load saved (left_cam, right_cam) from JSON config, or return defaults."""
    try:
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                left = data.get("left_cam")
                right = data.get("right_cam")
                if (
                    isinstance(left, int)
                    and not isinstance(left, bool)
                    and isinstance(right, int)
                    and not isinstance(right, bool)
                    and left >= 0
                    and right >= 0
                ):
                    return left, right
    except Exception as err:
        print(f"[CameraSelect] [WARN] Could not read {config_path.name}: {err}")
    return default_left, default_right


def save_camera_config(config_path: Path, left_cam: int, right_cam: int) -> None:
    """Persist (left_cam, right_cam) to JSON config."""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"left_cam": int(left_cam), "right_cam": int(right_cam)}
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except Exception as err:
        print(f"[CameraSelect] [WARN] Could not save camera config to {config_path}: {err}")


def parse_camera_selection(
    raw_input: str,
    current_left: int,
    current_right: int,
    available_ids: Optional[list[int]] = None,
    allow_unavailable: bool = False,
) -> tuple[int, int]:
    """Parse user terminal input into (left_cam, right_cam).

    Supported formats:
      - "" (Enter)             -> accept (current_left, current_right)
      - "left 0, right 2"      -> set left=0, right=2 (commas optional, any order)
      - "left=0 right=2"       -> '=', ':' and 'l'/'r' shorthands supported
      - "left 1" / "right 3"   -> update one side while keeping the other saved ID
      - "0 2" / "0, 2"         -> bare pair interpreted as <left> <right>
      - "swap"                 -> swap current_left and current_right
    """
    text = raw_input.strip()

    if not text:
        left, right = current_left, current_right
    elif text.lower() == "swap":
        left, right = current_right, current_left
    else:
        # Check bare two-integer shorthand first: e.g. "0 2" or "0, 2"
        bare_match = re.fullmatch(r"(\d+)\s*(?:[,;/]|\s)\s*(\d+)", text)
        if bare_match:
            left = int(bare_match.group(1))
            right = int(bare_match.group(2))
        else:
            # Parse explicit left/right clauses: e.g. "left 0, right 2" or "right=2 left=0"
            token_pattern = re.compile(r"\b(left|l|right|r)\s*[:=]?\s*(\d+)\b", re.IGNORECASE)
            matches = list(token_pattern.finditer(text))
            if not matches:
                raise ValueError(
                    "Invalid format. Enter e.g. 'left 0, right 2' (or press Enter to accept saved IDs)."
                )

            # Ensure no unrecognized extra tokens exist (catches typos like 'left 0, rigth 2')
            remainder = token_pattern.sub("", text)
            if re.sub(r"[\s,;]+", "", remainder):
                raise ValueError(
                    f"Unrecognized input '{text}'. Use e.g. 'left 0, right 2' or press Enter."
                )

            left = current_left
            right = current_right
            seen_left = False
            seen_right = False
            for m in matches:
                side_key = m.group(1).lower()
                cam_idx = int(m.group(2))
                if side_key in ("left", "l"):
                    if seen_left:
                        raise ValueError("Duplicate 'left' camera entry in input.")
                    left = cam_idx
                    seen_left = True
                else:
                    if seen_right:
                        raise ValueError("Duplicate 'right' camera entry in input.")
                    right = cam_idx
                    seen_right = True

    if left == right:
        raise ValueError(f"Left and Right camera IDs must be different (both set to {left}).")

    if available_ids and not allow_unavailable:
        missing = [cid for cid in (left, right) if cid not in available_ids]
        if missing:
            missing_str = ", ".join(str(c) for c in missing)
            avail_str = ", ".join(str(c) for c in available_ids)
            raise ValueError(
                f"Camera ID(s) [{missing_str}] not among available cameras [{avail_str}]. "
                f"Please choose from [{avail_str}]."
            )

    return left, right


def open_available_cameras(
    max_index: int = 8,
    preview_width: int = 640,
    preview_height: int = 480,
) -> dict[int, cv2.VideoCapture]:
    """Probe indices 0..max_index-1 and return open VideoCapture streams keyed by camera ID."""
    backend = get_backend()
    cameras: dict[int, cv2.VideoCapture] = {}

    with SuppressStderr():
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY and platform.system() != "Windows":
                cap = cv2.VideoCapture(idx)

            if not cap.isOpened():
                cap.release()
                continue

            # Request 640x480 MJPG so 3-4 USB cameras can stream simultaneously within USB 2.0 bandwidth
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, preview_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preview_height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 30.0)

            warmup_ok = False
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    warmup_ok = True
                    break
                time.sleep(0.03)

            if not warmup_ok:
                # Fallback for integrated/virtual cameras that reject MJPG negotiation
                cap.release()
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    for _ in range(4):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            warmup_ok = True
                            break
                        time.sleep(0.03)

            if warmup_ok and cap.isOpened():
                cameras[idx] = cap
            else:
                cap.release()

    return cameras


def annotate_camera_frame(
    frame: Optional[np.ndarray],
    cam_id: int,
    saved_left: int,
    saved_right: int,
    tile_size: tuple[int, int] = (480, 300),
) -> np.ndarray:
    """Render a single camera frame into a fixed-size tile annotated with ID and LEFT/RIGHT."""
    tw, th = tile_size
    tile = np.full((th, tw, 3), (24, 26, 32), dtype=np.uint8)

    if frame is not None and frame.size > 0:
        if frame.ndim == 2:
            src = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            src = frame
        fh, fw = src.shape[:2]
        scale = min(tw / max(fw, 1), th / max(fh, 1))
        nw = max(1, int(round(fw * scale)))
        nh = max(1, int(round(fh * scale)))
        resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)
        ox = (tw - nw) // 2
        oy = (th - nh) // 2
        tile[oy : oy + nh, ox : ox + nw] = resized
    else:
        cv2.putText(
            tile,
            f"Camera {cam_id}: waiting for frame...",
            (20, th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (160, 165, 175),
            2,
            cv2.LINE_AA,
        )

    is_left = cam_id == saved_left
    is_right = cam_id == saved_right

    if is_left and is_right:
        role_text = " [LEFT & RIGHT]"
        accent_bgr = (0, 220, 255)
    elif is_left:
        role_text = " [LEFT]"
        accent_bgr = (60, 220, 80)
    elif is_right:
        role_text = " [RIGHT]"
        accent_bgr = (255, 190, 0)
    else:
        role_text = ""
        accent_bgr = (140, 145, 155)

    # Top overlay banner
    banner_h = 44
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tw, banner_h), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.78, tile, 0.22, 0, tile)

    label = f"ID: {cam_id}{role_text}"
    cv2.putText(
        tile,
        label,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        accent_bgr,
        2,
        cv2.LINE_AA,
    )

    # Colored frame border highlighting saved LEFT / RIGHT cameras
    border_thickness = 4 if (is_left or is_right) else 1
    cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), accent_bgr, border_thickness)

    return tile


def compose_camera_grid(
    tiles: list[np.ndarray],
    saved_left: int,
    saved_right: int,
) -> np.ndarray:
    """Stack annotated camera tiles into a single preview image with a footer bar."""
    if not tiles:
        empty = np.full((220, 640, 3), (24, 26, 32), dtype=np.uint8)
        cv2.putText(
            empty,
            "No physical cameras detected.",
            (24, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (80, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            empty,
            "Connect cameras and type 'rescan' in terminal, or press Enter to continue.",
            (24, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 205, 215),
            1,
            cv2.LINE_AA,
        )
        return empty

    n = len(tiles)
    cols = n if n <= 3 else min(3, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    th, tw = tiles[0].shape[:2]

    row_imgs: list[np.ndarray] = []
    for r in range(rows):
        row_tiles: list[np.ndarray] = []
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                row_tiles.append(tiles[idx])
            else:
                row_tiles.append(np.full((th, tw, 3), (18, 20, 25), dtype=np.uint8))
        row_imgs.append(np.hstack(row_tiles))

    grid = np.vstack(row_imgs)
    gw = grid.shape[1]

    footer_h = 42
    footer = np.full((footer_h, gw, 3), (15, 17, 22), dtype=np.uint8)
    hint = (
        f"Saved: Left={saved_left}, Right={saved_right}  |  "
        f"In terminal: press Enter to confirm, or type e.g. 'left 0, right 2'"
    )
    cv2.putText(
        footer,
        hint,
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (225, 230, 240),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([grid, footer])


def _release_cameras(cameras: dict[int, cv2.VideoCapture], had_window: bool) -> None:
    had_any = bool(cameras)
    for cap in cameras.values():
        try:
            cap.release()
        except Exception:
            pass
    cameras.clear()

    if had_window:
        try:
            cv2.destroyWindow(PREVIEW_WINDOW_TITLE)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
            for _ in range(3):
                cv2.waitKey(1)
        except Exception:
            pass

    if had_any and platform.system() == "Windows":
        # Allow DirectShow driver handles to settle before opening at 1280x720
        time.sleep(0.35)


def select_stereo_cameras(
    config_path: Path,
    cli_left: Optional[int] = None,
    cli_right: Optional[int] = None,
    max_probe_index: int = 8,
) -> tuple[int, int]:
    """Discover cameras, stream annotated preview, prompt user in terminal, and save choice."""
    saved_left, saved_right = load_camera_config(config_path)
    if cli_left is not None:
        saved_left = cli_left
    if cli_right is not None:
        saved_right = cli_right

    # Non-interactive stdin fallback (e.g., redirected pipe)
    if sys.stdin is None or not sys.stdin.isatty():
        print(
            f"[CameraSelect] Non-interactive stdin; using saved cameras Left={saved_left}, Right={saved_right}."
        )
        return saved_left, saved_right

    print("\n" + "=" * 70)
    print("  STEP 0: STEREO CAMERA DISCOVERY & SELECTION")
    print("  Scanning connected cameras...")
    cameras = open_available_cameras(max_index=max_probe_index)
    available_ids = sorted(cameras.keys())

    def print_status_banner() -> None:
        print("-" * 70)
        if available_ids:
            details = []
            for cid in available_ids:
                w = int(cameras[cid].get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cameras[cid].get(cv2.CAP_PROP_FRAME_HEIGHT))
                tag = ""
                if cid == saved_left and cid == saved_right:
                    tag = " [saved LEFT & RIGHT]"
                elif cid == saved_left:
                    tag = " [saved LEFT]"
                elif cid == saved_right:
                    tag = " [saved RIGHT]"
                details.append(f"ID {cid} ({w}x{h}){tag}")
            print("  Available cameras : " + ", ".join(details))
        else:
            print("  Available cameras : NONE DETECTED")
        print(f"  Saved assignment  : Left = {saved_left}, Right = {saved_right}")
        if len(available_ids) >= 2:
            missing_saved = [c for c in (saved_left, saved_right) if c not in available_ids]
            if missing_saved:
                print(
                    f"  [WARN] Saved camera ID(s) {missing_saved} not currently connected! "
                    f"Please select from {available_ids}."
                )
        elif len(available_ids) == 1:
            print(
                f"  [WARN] Only 1 camera detected (ID {available_ids[0]}). "
                "Plug in the second camera and type 'rescan', or press Enter to continue."
            )
        print("-" * 70)

    print_status_banner()

    input_queue: queue.Queue[Optional[str]] = queue.Queue()

    def prompt_once() -> None:
        prompt_msg = (
            f"  -> Enter 'left <id>, right <id>' (e.g. 'left 0, right 2'), 'swap', 'rescan',\n"
            f"     or press Enter to accept [Left={saved_left}, Right={saved_right}]: "
        )
        sys.stdout.write(prompt_msg)
        sys.stdout.flush()

        def _reader() -> None:
            try:
                line = sys.stdin.readline()
                input_queue.put(line if line != "" else None)
            except Exception:
                input_queue.put(None)

        threading.Thread(target=_reader, daemon=True).start()

    has_display = True
    last_frames: dict[int, Optional[np.ndarray]] = {cid: None for cid in available_ids}
    tile_size = (480, 300) if len(available_ids) <= 2 else (400, 250)

    prompt_once()

    try:
        while True:
            # 1. Grab latest frame from every available camera and update live preview window
            tiles: list[np.ndarray] = []
            for cid in available_ids:
                cap = cameras.get(cid)
                if cap is not None:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        last_frames[cid] = frame
                tiles.append(
                    annotate_camera_frame(
                        last_frames.get(cid),
                        cam_id=cid,
                        saved_left=saved_left,
                        saved_right=saved_right,
                        tile_size=tile_size,
                    )
                )

            if has_display:
                composite = compose_camera_grid(tiles, saved_left=saved_left, saved_right=saved_right)
                try:
                    cv2.imshow(PREVIEW_WINDOW_TITLE, composite)
                    cv2.waitKey(20)
                except cv2.error:
                    has_display = False
                    print("  [INFO] Display window unavailable; continuing with terminal selection only.")
            else:
                time.sleep(0.03)

            # 2. Check if user submitted a line in the terminal
            try:
                raw_line = input_queue.get_nowait()
            except queue.Empty:
                continue

            if raw_line is None:
                # EOF on stdin -> accept current saved IDs
                chosen_left, chosen_right = saved_left, saved_right
                break

            cmd_lower = raw_line.strip().lower()
            if cmd_lower in ("q", "quit", "exit"):
                raise KeyboardInterrupt("Camera selection aborted by user.")

            if cmd_lower == "rescan":
                _release_cameras(cameras, had_window=False)
                print("  Rescanning connected cameras...")
                cameras = open_available_cameras(max_index=max_probe_index)
                available_ids = sorted(cameras.keys())
                last_frames = {cid: None for cid in available_ids}
                tile_size = (480, 300) if len(available_ids) <= 2 else (400, 250)
                print_status_banner()
                prompt_once()
                continue

            try:
                chosen_left, chosen_right = parse_camera_selection(
                    raw_line,
                    current_left=saved_left,
                    current_right=saved_right,
                    available_ids=available_ids,
                    allow_unavailable=(len(available_ids) < 2),
                )
                break
            except ValueError as err:
                print(f"  [!] {err}")
                prompt_once()

    finally:
        _release_cameras(cameras, had_window=has_display)

    save_camera_config(config_path, chosen_left, chosen_right)
    print(f"[CameraSelect] Confirmed & saved stereo pair: Left = {chosen_left}, Right = {chosen_right}")
    print("=" * 70 + "\n")
    return chosen_left, chosen_right

