"""Map renderer overlaying robot pose, heading, and keypoint targets."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shared.keypoints import Keypoint, KeypointStore, MapInfo
from shared.protocol import RobotStatus


class MapRenderer:
    """Renders the 2D occupancy map with robot pose, heading arrow, and keypoint markers."""

    def __init__(self, map_info: Optional[MapInfo] = None) -> None:
        self.map_info = map_info
        self._cached_base_img: Optional[Image.Image] = None

    def set_map_info(self, map_info: MapInfo) -> None:
        self.map_info = map_info
        self._cached_base_img = None

    def _get_base_image(self) -> Image.Image:
        if self._cached_base_img is not None:
            return self._cached_base_img.copy()

        if self.map_info and self.map_info.image_bytes:
            try:
                img = Image.open(io.BytesIO(self.map_info.image_bytes)).convert("RGB")
                self._cached_base_img = img
                return img.copy()
            except Exception:
                pass

        # Fallback: create schematic canvas (600x400)
        img = Image.new("RGB", (600, 400), (35, 38, 45))
        draw = ImageDraw.Draw(img)
        for x in range(0, 600, 50):
            draw.line([(x, 0), (x, 400)], fill=(50, 55, 65), width=1)
        for y in range(0, 400, 50):
            draw.line([(0, y), (600, y)], fill=(50, 55, 65), width=1)
        draw.text((20, 20), "Occupancy Map (Schematic Grid)", fill=(180, 180, 180))
        return img

    def render(
        self,
        keypoints: list[Keypoint],
        robot_status: RobotStatus,
        target_keypoint_id: Optional[int] = None,
        target_size: tuple[int, int] = (450, 320),
    ) -> Image.Image:
        """Render map view scaled to target_size."""
        base = self._get_base_image()
        draw = ImageDraw.Draw(base)

        w, h = base.size

        # Draw Keypoints
        for kp in keypoints:
            if self.map_info and self.map_info.width_px > 0:
                px, py = self.map_info.world_to_pixel(kp.x, kp.y)
            else:
                # Default scale assuming [0..10] meters
                px = int(round((kp.x / 10.0) * w))
                py = int(round(h - (kp.y / 10.0) * h))

            px = max(5, min(w - 5, px))
            py = max(5, min(h - 5, py))

            # Color by visibility (green for high, orange for medium, blue for task '0')
            if kp.name == "0":
                dot_color = (60, 160, 255)  # Task center blue
                r = 6
            elif kp.visibility >= 0.75:
                dot_color = (50, 220, 90)   # High visibility green
                r = 5
            elif kp.visibility >= 0.3:
                dot_color = (255, 180, 40)  # Moderate visibility orange
                r = 4
            else:
                dot_color = (220, 80, 60)   # Low visibility red
                r = 4

            is_target = target_keypoint_id is not None and kp.id == target_keypoint_id
            if is_target:
                # Highlight active target
                draw.ellipse([(px - r - 4, py - r - 4), (px + r + 4, py + r + 4)], outline=(255, 230, 0), width=2)

            draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=dot_color, outline=(255, 255, 255), width=1)
            draw.text((px + r + 2, py - r - 2), kp.name, fill=(255, 255, 255))

        # Draw Robot Pose and Heading
        if robot_status.localized or (robot_status.x != 0.0 or robot_status.y != 0.0):
            if self.map_info and self.map_info.width_px > 0:
                rx, ry = self.map_info.world_to_pixel(robot_status.x, robot_status.y)
            else:
                rx = int(round((robot_status.x / 10.0) * w))
                ry = int(round(h - (robot_status.y / 10.0) * h))

            rx = max(10, min(w - 10, rx))
            ry = max(10, min(h - 10, ry))

            # Robot circle
            r_bot = 8
            draw.ellipse(
                [(rx - r_bot, ry - r_bot), (rx + r_bot, ry + r_bot)],
                fill=(0, 200, 255),
                outline=(255, 255, 255),
                width=2,
            )

            # Heading arrow
            # Note: in map pixel space, py decreases going up, so yaw angle is inverted
            rad = math.radians(robot_status.yaw_deg)
            arrow_len = 18
            ax = rx + int(round(arrow_len * math.cos(rad)))
            ay = ry - int(round(arrow_len * math.sin(rad)))
            draw.line([(rx, ry), (ax, ay)], fill=(255, 240, 0), width=3)

        # Scale to target display size while preserving aspect ratio
        base.thumbnail(target_size, Image.Resampling.BILINEAR)

        # Center on target_size background canvas
        canvas = Image.new("RGB", target_size, (25, 28, 35))
        offset_x = (target_size[0] - base.width) // 2
        offset_y = (target_size[1] - base.height) // 2
        canvas.paste(base, (offset_x, offset_y))

        return canvas

