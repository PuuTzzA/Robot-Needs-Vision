"""Map renderer overlaying robot pose, heading, and keypoint targets.

Renders an isotropic metric viewport centered on the robot's live position
filling the entire available map region so nearby walls, obstacles, and waypoints are clearly legible.
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
    """Renders a robot-centered 2D occupancy map with heading and keypoint markers."""

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
        half_x_m: float,
        half_y_m: float,
        target_w: int,
        target_h: int,
        ppm: float,
    ) -> Image.Image:
        """Extract and upscale a metric patch around (cx_w, cy_w) to fill target_w x target_h."""
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
            u_left = ((cx_w - half_x_m - ox) / res) + 0.5
            u_right = ((cx_w + half_x_m - ox) / res) + 0.5
            v_top = (h_px - 1 - ((cy_w + half_y_m - oy) / res)) + 0.5
            v_bottom = (h_px - 1 - ((cy_w - half_y_m - oy) / res)) + 0.5

            return full_img.transform(
                (target_w, target_h),
                Image.Transform.EXTENT,
                (u_left, v_top, u_right, v_bottom),
                resample=Image.Resampling.NEAREST,
                fillcolor=(205, 205, 205),
            )

        # Fallback schematic metric grid (0.5m spacing) aligned to world coordinates
        view = Image.new("RGB", (target_w, target_h), (35, 38, 45))
        draw = ImageDraw.Draw(view)
        grid_step_m = 0.5

        x_min = cx_w - half_x_m
        x_max = cx_w + half_x_m
        gx = math.ceil(x_min / grid_step_m) * grid_step_m
        while gx <= x_max + 1e-6:
            sx = int(round((gx - x_min) * ppm))
            draw.line([(sx, 0), (sx, target_h)], fill=(55, 60, 72), width=1)
            gx += grid_step_m

        y_min = cy_w - half_y_m
        y_max = cy_w + half_y_m
        gy = math.ceil(y_min / grid_step_m) * grid_step_m
        while gy <= y_max + 1e-6:
            sy = int(round((y_max - gy) * ppm))
            draw.line([(0, sy), (target_w, sy)], fill=(55, 60, 72), width=1)
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
        """Render robot-centered map view filling the entire target_size region."""
        span_m = float(view_size_m if view_size_m is not None else self.view_size_m)
        if span_m <= 0:
            span_m = DEFAULT_VIEW_SIZE_M

        target_w = max(64, int(target_size[0]))
        target_h = max(64, int(target_size[1]))

        # Maintain isotropic metric scale: 1m real world = same pixels along X and Y
        ppm = max(1e-3, min(target_w, target_h) / span_m)
        span_x_m = target_w / ppm
        span_y_m = target_h / ppm
        half_x_m = span_x_m / 2.0
        half_y_m = span_y_m / 2.0

        cx_w, cy_w, has_robot_pose = self._select_view_center(keypoints, robot_status)

        view = self._render_cropped_grid(cx_w, cy_w, half_x_m, half_y_m, target_w, target_h, ppm)
        draw = ImageDraw.Draw(view)

        def world_to_view(wx: float, wy: float) -> tuple[float, float]:
            sx = (wx - (cx_w - half_x_m)) * ppm
            sy = ((cy_w + half_y_m) - wy) * ppm
            return sx, sy

        # 1. Draw Keypoints within the viewport (plus edge indicator for off-screen active target)
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

            inside = (-r <= fx <= target_w + r) and (-r <= fy <= target_h + r)
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
                lx = px + r + 3 if px < target_w - 28 else px - r - 8 * max(1, len(kp.name))
                ly = py - r - 2 if py > 14 else py + r + 2
                draw.rectangle(
                    [(lx - 1, ly - 1), (lx + 6 * max(1, len(kp.name)) + 2, ly + 11)],
                    fill=(20, 24, 30),
                )
                draw.text((lx, ly), kp.name, fill=(255, 255, 255))
            elif is_target:
                # Active navigation target is outside the window: show edge direction indicator
                mid_x = target_w / 2.0
                mid_y = target_h / 2.0
                dx = fx - mid_x
                dy = fy - mid_y
                lim_x = max(1.0, mid_x - 14.0)
                lim_y = max(1.0, mid_y - 14.0)
                scale_x = lim_x / abs(dx) if abs(dx) > 1e-6 else float("inf")
                scale_y = lim_y / abs(dy) if abs(dy) > 1e-6 else float("inf")
                scale = min(scale_x, scale_y)
                ex = int(round(mid_x + dx * scale))
                ey = int(round(mid_y + dy * scale))
                draw.ellipse(
                    [(ex - 6, ey - 6), (ex + 6, ey + 6)],
                    fill=(255, 215, 0),
                    outline=(20, 24, 30),
                    width=2,
                )
                lx = max(4, min(target_w - 26, ex - 6))
                ly = max(4, min(target_h - 16, ey - 18 if ey > 24 else ey + 8))
                draw.rectangle([(lx - 1, ly - 1), (lx + 6 * max(1, len(kp.name)) + 2, ly + 11)], fill=(20, 24, 30))
                draw.text((lx, ly), kp.name, fill=(255, 230, 0))

        # 2. Draw Robot Pose and Heading at the exact center of the zoomed viewport
        if has_robot_pose:
            rx = int(round(target_w / 2.0))
            ry = int(round(target_h / 2.0))

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
        scale_label = f"{span_x_m:.1f}m x {span_y_m:.1f}m"
        badge_w = 6 * len(scale_label) + 8
        draw.rectangle([(4, target_h - 16), (4 + badge_w, target_h - 3)], fill=(20, 24, 30))
        draw.text((8, target_h - 15), scale_label, fill=(210, 215, 225))
        draw.rectangle([(0, 0), (target_w - 1, target_h - 1)], outline=(75, 85, 99), width=1)

        return view
