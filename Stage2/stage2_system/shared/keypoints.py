"""Stage 2 Keypoint Store & Map Coordinate Transformer.

Reads existing keypoint and map files strictly in-place with automatic
mtime-based hot-reloading. Never overwrites or modifies existing files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Optional
import yaml

try:
    from .protocol import TASK_POINT_NAME, VISIBILITY_DEFAULT, VISIBILITY_HIGH
except ImportError:
    from protocol import TASK_POINT_NAME, VISIBILITY_DEFAULT, VISIBILITY_HIGH


@dataclass
class Keypoint:
    """Represents a recorded robot waypoint on the 2D occupancy map."""
    id: int
    name: str
    x: float
    y: float
    yaw_degrees: float = 0.0
    quaternion: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
    visibility: float = 1.0
    timestamp: float = 0.0
    snapshot: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Keypoint:
        x_val = float(data.get("x", data.get("position", {}).get("x", 0.0)))
        y_val = float(data.get("y", data.get("position", {}).get("y", 0.0)))
        yaw_val = float(data.get("yaw_degrees", data.get("rotation", {}).get("yaw_degrees", 0.0)))

        quat = data.get("quaternion") or data.get("rotation", {}).get("quaternion")
        if not quat or not isinstance(quat, dict):
            rad = math.radians(yaw_val) / 2.0
            quat = {"x": 0.0, "y": 0.0, "z": math.sin(rad), "w": math.cos(rad)}

        return cls(
            id=int(data.get("id", 0)),
            name=str(data.get("name", "")).strip().upper(),
            x=round(x_val, 4),
            y=round(y_val, 4),
            yaw_degrees=round(yaw_val, 2),
            quaternion={
                "x": float(quat.get("x", 0.0)),
                "y": float(quat.get("y", 0.0)),
                "z": float(quat.get("z", 0.0)),
                "w": float(quat.get("w", 1.0)),
            },
            visibility=round(float(data.get("visibility", 1.0)), 3),
            timestamp=float(data.get("timestamp", 0.0)),
            snapshot=data.get("snapshot"),
        )


def parse_keypoints_json(text: str) -> tuple[list[Keypoint], str]:
    """Parse JSON keypoints payload into Keypoint objects and return (list, sha256)."""
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    data = json.loads(text)
    entries: list[dict] = []

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("keypoints", [])
        if not entries and "points" in data:
            legacy = data["points"]
            auto_idx = 0
            for name, pt in legacy.items():
                entries.append({
                    "id": auto_idx,
                    "name": name,
                    "x": float(pt.get("x", 0.0)),
                    "y": float(pt.get("y", 0.0)),
                    "yaw_degrees": float(pt.get("yaw_degrees", 0.0)),
                    "visibility": 1.0,
                })
                auto_idx += 1

    keypoints = [Keypoint.from_dict(item) for item in entries]
    return keypoints, sha


class KeypointStore:
    """Thread-safe keypoint manager supporting live file hot-reloading & remote updates."""

    def __init__(self, file_path: Optional[Path | str] = None) -> None:
        self.file_path = Path(file_path).resolve() if file_path else None
        self._keypoints: list[Keypoint] = []
        self._sha256: str = ""
        self._last_mtime: float = -1.0

        if self.file_path and self.file_path.exists():
            self.reload_if_needed()

    @property
    def sha256(self) -> str:
        return self._sha256

    def reload_if_needed(self) -> bool:
        """Check file modification time and reload strictly in read-only mode if modified."""
        if not self.file_path or not self.file_path.exists():
            return False

        try:
            mtime = os.path.getmtime(self.file_path)
            if mtime == self._last_mtime:
                return False

            content = self.file_path.read_text(encoding="utf-8")
            kps, sha = parse_keypoints_json(content)
            self._keypoints = kps
            self._sha256 = sha
            self._last_mtime = mtime
            return True
        except Exception as err:
            print(f"[KeypointStore] [WARN] Could not read keypoints file {self.file_path}: {err}")
            return False

    def update_from_payload(self, data: dict | list | str, sha: str = "") -> bool:
        """Update store from remote VM HTTP response."""
        try:
            if isinstance(data, str):
                kps, calc_sha = parse_keypoints_json(data)
            else:
                json_str = json.dumps(data)
                kps, calc_sha = parse_keypoints_json(json_str)

            self._keypoints = kps
            self._sha256 = sha or calc_sha
            return True
        except Exception as err:
            print(f"[KeypointStore] [WARN] Failed to parse keypoints payload: {err}")
            return False

    def get_all(self) -> list[Keypoint]:
        self.reload_if_needed()
        return list(self._keypoints)

    def get_by_id(self, keypoint_id: int) -> Optional[Keypoint]:
        self.reload_if_needed()
        for kp in self._keypoints:
            if kp.id == keypoint_id:
                return kp
        return None

    def get_by_name(self, name: str) -> list[Keypoint]:
        self.reload_if_needed()
        norm = name.strip().upper()
        return [kp for kp in self._keypoints if kp.name == norm]

    def object_point_names(self) -> list[str]:
        """Return unique sorted names for all non-task object locations (excluding '0')."""
        self.reload_if_needed()
        names = {kp.name for kp in self._keypoints if kp.name and kp.name != TASK_POINT_NAME}
        return sorted(list(names))

    def select_entry(
        self,
        name: str,
        robot_xy: tuple[float, float] = (0.0, 0.0),
        min_visibility: float = VISIBILITY_DEFAULT,
    ) -> Optional[Keypoint]:
        """Select best keypoint entry for a given waypoint name.

        Selection Rule:
        1. Find all entries for `name`.
        2. Filter by `visibility >= min_visibility`.
        3. If matching entries exist, select the one closest to `robot_xy`.
        4. If none pass `min_visibility`, log a warning and fall back to the
           entry with highest visibility.
        """
        entries = self.get_by_name(name)
        if not entries:
            return None

        # Filter by threshold
        eligible = [kp for kp in entries if kp.visibility >= min_visibility]

        rx, ry = robot_xy
        if eligible:
            # Sort by Euclidean distance to current robot pose
            eligible.sort(key=lambda kp: math.hypot(kp.x - rx, kp.y - ry))
            return eligible[0]

        # Fallback to highest visibility
        print(
            f"[KeypointStore] [WARN] No entry for '{name}' with visibility >= {min_visibility:.2f}. "
            f"Falling back to highest available visibility."
        )
        entries.sort(key=lambda kp: kp.visibility, reverse=True)
        return entries[0]

    def order_greedy(
        self,
        names: list[str],
        start_xy: tuple[float, float] = (0.0, 0.0),
        min_visibility: float = VISIBILITY_DEFAULT,
    ) -> list[str]:
        """Greedy nearest-neighbour ordering of point names starting from start_xy."""
        remaining = list(names)
        ordered: list[str] = []
        cur_pos = start_xy

        while remaining:
            best_name = None
            best_dist = float("inf")
            best_next_xy = cur_pos

            for name in remaining:
                entry = self.select_entry(name, robot_xy=cur_pos, min_visibility=min_visibility)
                if entry is None:
                    continue
                d = math.hypot(entry.x - cur_pos[0], entry.y - cur_pos[1])
                if d < best_dist:
                    best_dist = d
                    best_name = name
                    best_next_xy = (entry.x, entry.y)

            if best_name is None:
                # Add whatever remains if no coordinates found
                ordered.extend(remaining)
                break

            ordered.append(best_name)
            remaining.remove(best_name)
            cur_pos = best_next_xy

        return ordered


class MapInfo:
    """Handles ROS 2 Nav2 map metadata and world-to-pixel coordinate conversions."""

    def __init__(
        self,
        yaml_path: Optional[Path | str] = None,
        raw_meta: Optional[dict[str, Any]] = None,
        image_bytes: Optional[bytes] = None,
    ) -> None:
        self.yaml_path = Path(yaml_path).resolve() if yaml_path else None
        self.image_path: Optional[Path] = None
        self.resolution: float = 0.05
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        self.origin_yaw: float = 0.0
        self.width_px: int = 0
        self.height_px: int = 0
        self.image_bytes: Optional[bytes] = image_bytes
        self._last_mtime: float = -1.0

        if raw_meta:
            self._apply_metadata(raw_meta)
        elif self.yaml_path and self.yaml_path.exists():
            self.load()

    def load(self) -> bool:
        """Load map metadata and image in place."""
        if not self.yaml_path or not self.yaml_path.exists():
            return False

        try:
            mtime = os.path.getmtime(self.yaml_path)
            content = self.yaml_path.read_text(encoding="utf-8")
            meta = yaml.safe_load(content)
            self._apply_metadata(meta)

            img_rel = meta.get("image", "map.pgm")
            img_path = (self.yaml_path.parent / img_rel).resolve()
            if img_path.exists():
                self.image_path = img_path
                self.image_bytes = img_path.read_bytes()
                self._inspect_image_dimensions()

            self._last_mtime = mtime
            return True
        except Exception as e:
            print(f"[MapInfo] [WARN] Error loading map from {self.yaml_path}: {e}")
            return False

    def _apply_metadata(self, meta: dict[str, Any]) -> None:
        self.resolution = float(meta.get("resolution", 0.05))
        origin = meta.get("origin", [0.0, 0.0, 0.0])
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        self.origin_yaw = float(origin[2]) if len(origin) > 2 else 0.0

    def _inspect_image_dimensions(self) -> None:
        if not self.image_bytes:
            return
        try:
            import cv2
            import numpy as np
            arr = np.frombuffer(self.image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.height_px, self.width_px = img.shape[:2]
        except Exception:
            pass

    def world_to_pixel(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert real-world metric coordinates (meters) to image pixel coordinates (u, v)."""
        if self.resolution <= 0:
            return 0, 0
        px = int(round((wx - self.origin_x) / self.resolution))
        # Map origin is at bottom-left in ROS map frame
        py = int(round(self.height_px - 1 - ((wy - self.origin_y) / self.resolution))) if self.height_px > 0 else 0
        return px, py

    def pixel_to_world(self, px: int, py: int) -> tuple[float, float]:
        """Convert image pixel coordinates (u, v) to real-world metric coordinates (meters)."""
        wx = self.origin_x + (px * self.resolution)
        wy = self.origin_y + ((self.height_px - 1 - py) * self.resolution) if self.height_px > 0 else 0.0
        return round(wx, 4), round(wy, 4)
