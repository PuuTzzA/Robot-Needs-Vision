"""Protocol definitions and shared dataclasses for Stage 2 System.

Defines network ports, discovery tokens, HTTP endpoints, and message schemas
used across Windows Mission Control, VM Robot Controller, and SSH Size Server.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any, Optional

# -----------------------------------------------------------------------------
# Default Network Configuration & Ports
# -----------------------------------------------------------------------------
DEFAULT_VM_CMD_PORT = 8070       # VM robot_controller HTTP API
DEFAULT_VM_BEACON_PORT = 8079    # VM robot_controller UDP discovery beacon
DEFAULT_LAPTOP_PING_PORT = 8071  # Laptop main.py HTTP status ping receiver
DEFAULT_SSH_SIZE_PORT = 8082     # SSH size_server HTTP API

DISCOVERY_BEACON_REQUEST = b"TB4_ROBOT_DISCOVERY"
DISCOVERY_BEACON_RESPONSE_PREFIX = "TB4_ROBOT_OFFER:"

# Task point name
TASK_POINT_NAME = "0"

# Default visibility thresholds
VISIBILITY_DEFAULT = 0.10
VISIBILITY_HIGH = 0.75

# -----------------------------------------------------------------------------
# Dataclasses & Message Types
# -----------------------------------------------------------------------------

@dataclass
class RobotStatus:
    """Robot status ping sent periodically (e.g. 2 Hz) from VM to Laptop."""
    t: float = field(default_factory=time.time)
    localized: bool = False
    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0
    nav_state: str = "idle"  # idle | navigating | succeeded | aborted | canceled | rejected
    goal_seq: int = 0
    goal_keypoint_id: Optional[int] = None
    goal_name: Optional[str] = None
    nav2_ready: bool = False
    teleop_active: bool = False
    keypoints_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotStatus:
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class GotoRequest:
    """Navigation target request sent from Laptop to VM."""
    keypoint_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GotoRequest:
        return cls(keypoint_id=int(data.get("keypoint_id", -1)))


@dataclass
class TeleopRequest:
    """Short-burst teleop velocity command sent from Laptop to VM."""
    key: str  # 'w', 'a', 's', 'd', 'space'
    duration: float = 0.6
    linear_x: Optional[float] = None
    angular_z: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeleopRequest:
        return cls(
            key=str(data.get("key", "")).lower(),
            duration=float(data.get("duration", 0.6)),
            linear_x=float(data["linear_x"]) if "linear_x" in data and data["linear_x"] is not None else None,
            angular_z=float(data["angular_z"]) if "angular_z" in data and data["angular_z"] is not None else None,
        )


@dataclass
class RegisterRequest:
    """Laptop registration payload so VM knows where to push status pings."""
    ping_port: int = DEFAULT_LAPTOP_PING_PORT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegisterRequest:
        return cls(ping_port=int(data.get("ping_port", DEFAULT_LAPTOP_PING_PORT)))


@dataclass
class DetectionItem:
    """A single object detection from the size server."""
    class_name: str
    confidence: float
    bbox_px: list[int]  # [x1, y1, x2, y2]
    height_cm: Optional[float]
    width_cm: Optional[float]
    distance_m: Optional[float]
    valid_depth_pixels: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectionItem:
        return cls(
            class_name=str(data.get("class_name", data.get("class", "unknown"))),
            confidence=float(data.get("confidence", 0.0)),
            bbox_px=list(data.get("bbox_px", [0, 0, 0, 0])),
            height_cm=float(data["height_cm"]) if data.get("height_cm") is not None else None,
            width_cm=float(data["width_cm"]) if data.get("width_cm") is not None else None,
            distance_m=float(data["distance_m"]) if data.get("distance_m") is not None else None,
            valid_depth_pixels=int(data.get("valid_depth_pixels", 0)),
        )


@dataclass
class MeasurementResult:
    """Result returned by SSH size_server for a single stereo pair."""
    ok: bool
    point: Optional[str] = None
    primary: Optional[DetectionItem] = None
    detections: list[DetectionItem] = field(default_factory=list)
    annotated_jpeg_b64: Optional[str] = None
    depth_jpeg_b64: Optional[str] = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        if self.primary:
            res["primary"] = self.primary.to_dict()
        res["detections"] = [d.to_dict() for d in self.detections]
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeasurementResult:
        primary_raw = data.get("primary")
        primary_item = DetectionItem.from_dict(primary_raw) if primary_raw else None
        dets = [DetectionItem.from_dict(d) for d in data.get("detections", [])]
        return cls(
            ok=bool(data.get("ok", False)),
            point=data.get("point"),
            primary=primary_item,
            detections=dets,
            annotated_jpeg_b64=data.get("annotated_jpeg_b64"),
            depth_jpeg_b64=data.get("depth_jpeg_b64"),
            timing_ms=dict(data.get("timing_ms", {})),
            error_message=data.get("error_message"),
        )

