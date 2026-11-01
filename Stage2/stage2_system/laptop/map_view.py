"""Map renderer overlaying robot pose, heading, and keypoint targets.

Renders a zoomed metric viewport (default 2.5m x 2.5m) centered on the robot's
live position so nearby walls, obstacles, and waypoints are clearly legible.
"""

from __future__ import annotations

import io
import math
from typing import Optional
from PIL import Image, ImageDraw

from shared.keypoints import Keypoint, MapInfo
from shared.protocol import RobotStatus, TASK_POINT_NAME

DEFAULT_VIEW_SIZE_M = 2.5


class MapRenderer:
    """Renders a robot-centered zoomed 2D occupancy map with heading and keypoint markers."""

    def __init__(
        self,
        map_info: Optional[MapInfo] = None,
        view_size_m: float = DEFAULT_VIEW_SIZE_M,
    ) -> None:
        self.map_info = map_info
        self.view_size_m = float(view_size_m)
        self._cached_base_img: Optional[Image.Image] = None

    def set_map_info(self, map_info: MapInfo) -> None:
        self.map_info = map_info
        self._cached_base_img = None

    def _get_full_map_image(self) -> Optional[Image.Image]:
        if self._cached_base_img is not None:
            return self._cached_base_img

        if self.map_info and self.map_info.image_bytes:
            try:
                img = Image.open(io.BytesIO(self.map_info.image_bytes)).convert("RGB")
                self._cached_base_img = img
                return img
            except Exception:
                pass
        return None

    def _select_view_center(
        self,
        keypoints: list[Keypoint],
        robot_status: RobotStatus,
    ) -> tuple[float, float, bool]:
        """Return (center_x_m, center_y_m, has_robot_pose)."""
        has_robot_pose = bool(
            robot_status.localized or (robot_status.x != 0.0 or robot_status.y != 0.0)
        )
        if has_robot_pose:
            return robot_status.x, robot_status.y, True

        # Before localization, center on Task Point "0" if available
        for kp in keypoints:
            if kp.name == TASK_POINT_NAME:
                return kp.x, kp.y, False

        if keypoints:
            avg_x = sum(kp.x for kp in keypoints) / len(keypoints)
            avg_y = sum(kp.y for kp in keypoints) / len(keypoints)
            return avg_x, avg_y, False

        if self.map_info and self.map_info.width_px > 0 and self.map_info.height_px > 0:
            wx, wy = self.map_info.pixel_to_world(
                self.map_info.width_px // 2,
                self.map_info.height_px // 2,
            )
            return wx, wy, False

        return 5.0, 5.0, False

    def _render_cropped_grid(
        self,
        cx_w: float,
        cy_w: float,
        span_m: float,
        side: int,
    ) -> Image.Image:
        """Extract and upscale a `span_m x span_m` metric patch around (cx_w, cy_w)."""
        half_m = span_m / 2.0
        full_img = self._get_full_map_image()

        if (
            full_img is not None
            and self.map_info is not None
            and self.map_info.resolution > 0
            and self.map_info.height_px > 0
        ):
            res = self.map_info.resolution
            ox = self.map_info.origin_x
            oy = self.map_info.origin_y
            h_px = self.map_info.height_px

            # Continuous Pillow EXTENT coordinates (pixel centers at col + 0.5, row + 0.5)
            u_left = ((cx_w - half_m - ox) / res) + 0.5
            u_right = ((cx_w + half_m - ox) / res) + 0.5
            v_top = (h_px - 1 - ((cy_w + half_m - oy) / res)) + 0.5
            v_bottom = (h_px - 1 - ((cy_w - half_m - oy) / res)) + 0.5

            return full_img.transform(
                (side, side),
                Image.Transform.EXTENT,
                (u_left, v_top, u_right, v_bottom),
                resample=Image.Resampling.NEAREST,
                fillcolor=(205, 205, 205),
            )

        # Fallback schematic metric grid (0.5m spacing) aligned to world coordinates
        view = Image.new("RGB", (side, side), (35, 38, 45))
        draw = ImageDraw.Draw(view)
        grid_step_m = 0.5

        x_min = cx_w - half_m
        x_max = cx_w + half_m
        gx = math.ceil(x_min / grid_step_m) * grid_step_m
        while gx <= x_max + 1e-6:
            sx = int(round(((gx - x_min) / span_m) * side))
            draw.line([(sx, 0), (sx, side)], fill=(55, 60, 72), width=1)
            gx += grid_step_m

        y_min = cy_w - half_m
        y_max = cy_w + half_m
        gy = math.ceil(y_min / grid_step_m) * grid_step_m
        while gy <= y_max + 1e-6:
            sy = int(round(((y_max - gy) / span_m) * side))
            draw.line([(0, sy), (side, sy)], fill=(55, 60, 72), width=1)
            gy += grid_step_m

        return view

    def render(
        self,
        keypoints: list[Keypoint],
        robot_status: RobotStatus,
        target_keypoint_id: Optional[int] = None,
        target_size: tuple[int, int] = (450, 320),
        view_size_m: Optional[float] = None,
    ) -> Image.Image:
        """Render robot-centered zoomed map view (`view_size_m x view_size_m`) into target_size."""
        span_m = float(view_size_m if view_size_m is not None else self.view_size_m)
        if span_m <= 0:
            span_m = DEFAULT_VIEW_SIZE_M
        half_m = span_m / 2.0

        side = max(64, min(target_size[0], target_size[1]))
        cx_w, cy_w, has_robot_pose = self._select_view_center(keypoints, robot_status)

        view = self._render_cropped_grid(cx_w, cy_w, span_m, side)
        draw = ImageDraw.Draw(view)

        def world_to_view(wx: float, wy: float) -> tuple[float, float]:
            sx = ((wx - (cx_w - half_m)) / span_m) * side
            sy = (((cy_w + half_m) - wy) / span_m) * side
            return sx, sy

        # 1. Draw Keypoints within the zoomed viewport (plus edge indicator for off-screen active target)
        for kp in keypoints:
            fx, fy = world_to_view(kp.x, kp.y)
            is_target = target_keypoint_id is not None and kp.id == target_keypoint_id

            if kp.name == TASK_POINT_NAME:
                dot_color = (60, 160, 255)   # Task center blue
                r = 6
            elif kp.visibility >= 0.75:
                dot_color = (50, 220, 90)    # High visibility green
                r = 5
            elif kp.visibility >= 0.3:
                dot_color = (255, 180, 40)   # Moderate visibility orange
                r = 5
            else:
                dot_color = (220, 80, 60)    # Low visibility red
                r = 4

            inside = (-r <= fx <= side + r) and (-r <= fy <= side + r)
            if inside:
                px = int(round(fx))
                py = int(round(fy))
                if is_target:
                    draw.ellipse(
                        [(px - r - 4, py - r - 4), (px + r + 4, py + r + 4)],
                        outline=(255, 230, 0),
                        width=2,
                    )
                draw.ellipse(
                    [(px - r, py - r), (px + r, py + r)],
                    fill=dot_color,
                    outline=(255, 255, 255),
                    width=1,
                )
                # Keep label inside viewport edges
                lx = px + r + 3 if px < side - 28 else px - r - 8 * max(1, len(kp.name))
                ly = py - r - 2 if py > 14 else py + r + 2
                draw.rectangle(
                    [(lx - 1, ly - 1), (lx + 6 * max(1, len(kp.name)) + 2, ly + 11)],
                    fill=(20, 24, 30),
                )
                draw.text((lx, ly), kp.name, fill=(255, 255, 255))
            elif is_target:
                # Active navigation target is outside the 2.5m x 2.5m window: show edge direction indicator
                mid = side / 2.0
                dx = fx - mid
                dy = fy - mid
                max_comp = max(abs(dx), abs(dy), 1e-6)
                scale_to_edge = (mid - 12.0) / max_comp
                ex = int(round(mid + dx * scale_to_edge))
                ey = int(round(mid + dy * scale_to_edge))
                draw.ellipse(
                    [(ex - 6, ey - 6), (ex + 6, ey + 6)],
                    fill=(255, 215, 0),
                    outline=(20, 24, 30),
                    width=2,
                )
                lx = max(4, min(side - 26, ex - 6))
                ly = max(4, min(side - 16, ey - 18 if ey > 24 else ey + 8))
                draw.rectangle([(lx - 1, ly - 1), (lx + 6 * max(1, len(kp.name)) + 2, ly + 11)], fill=(20, 24, 30))
                draw.text((lx, ly), kp.name, fill=(255, 230, 0))

        # 2. Draw Robot Pose and Heading at the exact center of the zoomed viewport
        if has_robot_pose:
            rx = int(round(side / 2.0))
            ry = int(round(side / 2.0))

            r_bot = 8
            draw.ellipse(
                [(rx - r_bot, ry - r_bot), (rx + r_bot, ry + r_bot)],
                fill=(0, 200, 255),
                outline=(255, 255, 255),
                width=2,
            )

            # In image space, y decreases going up (North), so sin(yaw) is negated
            rad = math.radians(robot_status.yaw_deg)
            arrow_len = 18
            ax = rx + int(round(arrow_len * math.cos(rad)))
            ay = ry - int(round(arrow_len * math.sin(rad)))
            draw.line([(rx, ry), (ax, ay)], fill=(255, 240, 0), width=3)

        # 3. Viewport border & scale badge
        scale_label = f"{span_m:g}m x {span_m:g}m"
        badge_w = 6 * len(scale_label) + 8
        draw.rectangle([(4, side - 16), (4 + badge_w, side - 3)], fill=(20, 24, 30))
        draw.text((8, side - 15), scale_label, fill=(210, 215, 225))
        draw.rectangle([(0, 0), (side - 1, side - 1)], outline=(75, 85, 99), width=1)

        # Center square viewport on target_size background canvas
        canvas = Image.new("RGB", target_size, (25, 28, 35))
        offset_x = (target_size[0] - side) // 2
        offset_y = (target_size[1] - side) // 2
        canvas.paste(view, (offset_x, offset_y))

        return canvas
