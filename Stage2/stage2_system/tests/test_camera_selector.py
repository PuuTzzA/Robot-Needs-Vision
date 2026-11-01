"""Unit tests for camera selector, config persistence, and 2.5m x 2.5m zoomed MapRenderer."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from laptop.camera_selector import (
    annotate_camera_frame,
    compose_camera_grid,
    load_camera_config,
    parse_camera_selection,
    save_camera_config,
)
from laptop.map_view import MapRenderer
from shared.keypoints import Keypoint, MapInfo
from shared.protocol import RobotStatus


class TestCameraSelectorConfigAndParsing(unittest.TestCase):
    def test_config_roundtrip_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / ".camera_config.json"
            # Missing file -> defaults (0, 2)
            self.assertEqual(load_camera_config(cfg), (0, 2))

            save_camera_config(cfg, 1, 3)
            self.assertEqual(load_camera_config(cfg), (1, 3))

            # Malformed JSON -> fallback safely
            cfg.write_text("{not valid json", encoding="utf-8")
            self.assertEqual(load_camera_config(cfg), (0, 2))

            # Boolean / negative values rejected -> fallback safely
            cfg.write_text('{"left_cam": true, "right_cam": -1}', encoding="utf-8")
            self.assertEqual(load_camera_config(cfg), (0, 2))

    def test_parse_enter_and_explicit_pairs(self):
        avail = [0, 1, 2, 3]
        self.assertEqual(parse_camera_selection("", 0, 2, avail), (0, 2))
        self.assertEqual(parse_camera_selection("   \n", 1, 3, avail), (1, 3))
        self.assertEqual(parse_camera_selection("left 0, right 2", 1, 3, avail), (0, 2))
        self.assertEqual(parse_camera_selection("left 1 right 3", 0, 2, avail), (1, 3))
        self.assertEqual(parse_camera_selection("right 2, left 0", 1, 3, avail), (0, 2))
        self.assertEqual(parse_camera_selection("LEFT=3, RIGHT:1", 0, 2, avail), (3, 1))
        self.assertEqual(parse_camera_selection("l 2, r 0", 0, 1, avail), (2, 0))
        self.assertEqual(parse_camera_selection("0 2", 1, 3, avail), (0, 2))
        self.assertEqual(parse_camera_selection("1, 3", 0, 2, avail), (1, 3))
        self.assertEqual(parse_camera_selection("swap", 0, 2, avail), (2, 0))

    def test_parse_single_side_override(self):
        avail = [0, 1, 2, 3]
        self.assertEqual(parse_camera_selection("left 1", 0, 2, avail), (1, 2))
        self.assertEqual(parse_camera_selection("right 3", 0, 2, avail), (0, 3))

    def test_parse_rejects_invalid_inputs(self):
        avail = [0, 1, 2]
        # Same ID for both left and right
        with self.assertRaisesRegex(ValueError, "different"):
            parse_camera_selection("left 1, right 1", 0, 2, avail)

        # Camera ID not in available_ids
        with self.assertRaisesRegex(ValueError, "not among available"):
            parse_camera_selection("left 0, right 4", 0, 2, avail)

        # Saved camera ID missing when user presses Enter with >=2 cameras available
        with self.assertRaisesRegex(ValueError, "not among available"):
            parse_camera_selection("", 0, 5, avail)

        # Misspelled side should NOT silently succeed with partial assignment
        with self.assertRaisesRegex(ValueError, "Unrecognized|Invalid"):
            parse_camera_selection("left 0, rigth 2", 1, 1, avail)

    def test_annotate_and_grid_handle_mixed_resolutions(self):
        f_4_3 = np.full((480, 640, 3), 90, dtype=np.uint8)
        f_16_9 = np.full((720, 1280, 3), 140, dtype=np.uint8)

        t0 = annotate_camera_frame(f_4_3, cam_id=0, saved_left=0, saved_right=2, tile_size=(400, 250))
        t1 = annotate_camera_frame(f_16_9, cam_id=1, saved_left=0, saved_right=2, tile_size=(400, 250))
        t2 = annotate_camera_frame(None, cam_id=2, saved_left=0, saved_right=2, tile_size=(400, 250))

        # Source arrays are not mutated
        self.assertTrue(np.all(f_4_3 == 90))
        self.assertEqual(t0.shape, (250, 400, 3))
        self.assertEqual(t1.shape, (250, 400, 3))
        self.assertEqual(t2.shape, (250, 400, 3))

        grid = compose_camera_grid([t0, t1, t2], saved_left=0, saved_right=2)
        self.assertEqual(grid.shape[1], 1200)
        self.assertGreater(grid.shape[0], 250)


class TestZoomedMapRenderer(unittest.TestCase):
    def test_zoomed_2_5m_crops_and_centers_on_robot(self):
        # Build a synthetic 200x200 occupancy map at 0.05 m/px (10.0m x 10.0m, origin [0, 0, 0])
        # Fill white (250) everywhere except:
        #   1) Near obstacle at world (5.50m, 5.00m) -> 0.50m East of robot at (5.00m, 5.00m) (inside 2.5m view)
        #   2) Far obstacle at world (8.00m, 5.00m)  -> 3.00m East of robot (outside 2.5m view)
        arr = np.full((200, 200, 3), 250, dtype=np.uint8)

        # World (5.50, 5.00) -> px = 5.50/0.05 = 110, py = 199 - 5.00/0.05 = 99
        arr[98:101, 109:112] = (10, 10, 10)
        # World (8.00, 5.00) -> px = 8.00/0.05 = 160, py = 99
        arr[95:105, 155:165] = (200, 0, 0)

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")

        map_info = MapInfo(
            raw_meta={"resolution": 0.05, "origin": [0.0, 0.0, 0.0]},
            image_bytes=buf.getvalue(),
        )
        map_info._inspect_image_dimensions()

        renderer = MapRenderer(map_info=map_info, view_size_m=2.5)
        status = RobotStatus(localized=True, x=5.0, y=5.0, yaw_deg=0.0)
        out = renderer.render(keypoints=[], robot_status=status, target_size=(440, 260))
        self.assertEqual(out.size, (440, 260))

        out_np = np.array(out)
        # Viewport is 260x260 centered horizontally at offset_x = (440 - 260)//2 = 90
        # Robot is centered at x = 90 + 130 = 220, y = 130 (yellow heading arrow at origin, cyan body behind it)
        self.assertEqual(tuple(int(c) for c in out_np[130, 220]), (255, 240, 0))
        self.assertEqual(tuple(int(c) for c in out_np[130, 216]), (0, 200, 255))

        # Near obstacle is +0.50m East in a 2.5m span -> +0.20 * 260 = +52 px right of center -> x = 220 + 52 = 272, y = 130
        near_patch = out_np[126:134, 268:276]
        self.assertTrue(np.any(near_patch < 30), "Obstacle 0.5m East of robot should appear at +52px in 2.5m view")

        # Far red obstacle (200, 0, 0) at +3.0m East is outside [-1.25m, +1.25m] and must NOT appear anywhere
        red_mask = (out_np[:, :, 0] > 180) & (out_np[:, :, 1] < 30) & (out_np[:, :, 2] < 30)
        self.assertFalse(np.any(red_mask), "Obstacle 3.0m away must be cropped out of the 2.5m x 2.5m view")

    def test_near_edge_robot_pose_pads_cleanly_without_crash(self):
        renderer = MapRenderer(map_info=None, view_size_m=2.5)
        kps = [
            Keypoint(id=0, name="0", x=0.2, y=0.2, visibility=1.0),
            Keypoint(id=1, name="B", x=9.5, y=9.5, visibility=0.8),
        ]
        status = RobotStatus(localized=True, x=0.1, y=0.1, yaw_deg=90.0)
        out = renderer.render(keypoints=kps, robot_status=status, target_keypoint_id=1, target_size=(440, 260))
        self.assertEqual(out.size, (440, 260))

    def test_real_stage2_map_and_controller_default_paths(self):
        real_maps_dir = sys_root.parent / "vm_map_tools" / "maps"
        map_yaml = real_maps_dir / "final_room.yaml"
        if map_yaml.exists():
            mi = MapInfo(yaml_path=map_yaml)
            self.assertGreater(mi.width_px, 0)
            renderer = MapRenderer(map_info=mi)
            out = renderer.render(
                keypoints=[Keypoint(id=0, name="0", x=5.42, y=5.43, visibility=1.0)],
                robot_status=RobotStatus(localized=True, x=5.42, y=5.43, yaw_deg=141.0),
                target_size=(440, 260),
            )
            self.assertEqual(out.size, (440, 260))

    def test_map_rectangular_fills_target_size(self):
        """Verifies that map rendering fills the entire rectangular space (e.g. 560x376) without borders."""
        renderer = MapRenderer(map_info=None, view_size_m=2.5)
        status = RobotStatus(localized=True, x=2.0, y=3.0, yaw_deg=45.0)
        target_size = (560, 376)
        out = renderer.render(keypoints=[], robot_status=status, target_size=target_size)
        self.assertEqual(out.size, target_size)

        out_np = np.array(out)
        # Robot center must be at exact center of target_size: (560 // 2 = 280, 376 // 2 = 188)
        cx = target_size[0] // 2
        cy = target_size[1] // 2
        # Yellow heading arrow originates at (cx, cy), cyan body surrounds it
        self.assertEqual(tuple(int(c) for c in out_np[cy, cx]), (255, 240, 0))


if __name__ == "__main__":
    unittest.main()
