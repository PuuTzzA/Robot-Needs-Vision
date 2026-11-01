"""Unit tests for Stage 2 GUI visual improvements: borderless rendering, map rectangular fill, and result wrapping."""

from pathlib import Path
import sys
import tkinter as tk
import unittest
import numpy as np

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from laptop.gui import MissionControlGUI
from laptop.map_view import MapRenderer
from laptop.mission import MissionManager
from laptop.commands import CommandDispatcher
from laptop.camera import CameraManager
from laptop.robot_link import RobotLink
from laptop.size_link import SizeLink
from shared.keypoints import Keypoint, KeypointStore
from shared.protocol import RobotStatus


class TestGuiVisuals(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.geometry("1480x880")
        self.camera = CameraManager(is_mock=True)
        self.robot = RobotLink(vm_ip="127.0.0.1")
        self.size = SizeLink(is_mock=True)
        self.kps = KeypointStore()
        self.mission = MissionManager(self.camera, self.robot, self.size, self.kps)
        self.dispatcher = CommandDispatcher(self.mission)
        self.renderer = MapRenderer()
        self.gui = MissionControlGUI(self.root, self.mission, self.dispatcher, self.renderer)
        self.root.update()

    def tearDown(self):
        self.gui._on_close()

    def test_image_labels_have_zero_borders(self):
        """Verifies that the 4 image container labels have no border or highlight padding."""
        for label in [
            self.gui.canvas_left_cam,
            self.gui.canvas_right_cam,
            self.gui.canvas_proc,
            self.gui.canvas_map,
        ]:
            self.assertEqual(int(label.cget("bd")), 0)
            self.assertEqual(int(label.cget("highlightthickness")), 0)
            self.assertEqual(int(label.cget("padx")), 0)
            self.assertEqual(int(label.cget("pady")), 0)

    def test_dynamic_frame_render_to_label(self):
        """Verifies that _render_frame_to_label scales to widget's actual dimensions."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Force a known size on the label
        self.gui.canvas_left_cam.config(width=500, height=300)
        self.root.update()

        self.gui._render_frame_to_label(frame, self.gui.canvas_left_cam, "_img_left")
        img = getattr(self.gui, "_img_left")
        self.assertIsNotNone(img)
        self.assertEqual(img.width(), max(64, self.gui.canvas_left_cam.winfo_width()))
        self.assertEqual(img.height(), max(64, self.gui.canvas_left_cam.winfo_height()))

    def test_result_text_wrapping_and_adaptive_font(self):
        """Verifies that long result strings wrap and adopt compact font."""
        short_res = "Cup at Point B"
        long_res = (
            "Answer: The tallest object in the venue is the green bottle located at "
            "Point B with a measured height of 31.5 cm, followed by the cup at Point C with 14.2 cm."
        )

        # 1. Short result
        self.mission.current_level = 1
        self.mission.set_result(1, short_res)
        self.gui._update_loop()
        self.root.update()
        self.assertIn(short_res, self.gui.lbl_result.cget("text"))
        self.assertGreater(self.gui.lbl_result.cget("wraplength"), 200)

        # 2. Long result triggers compact font (len > 80)
        self.mission.current_level = 3
        self.mission.set_result(3, long_res)
        self.gui._update_loop()
        self.root.update()
        self.assertIn(long_res, self.gui.lbl_result.cget("text"))
        self.assertEqual(self.gui.lbl_result.cget("font"), str(self.gui.font_result_compact))
        self.assertGreater(self.gui.lbl_result.cget("wraplength"), 200)

    def test_map_fills_arbitrary_rectangular_target_size(self):
        """Verifies that the map view renders directly into rectangular target sizes."""
        sizes_to_test = [(500, 300), (560, 376), (640, 400)]
        for w, h in sizes_to_test:
            out = self.renderer.render(
                keypoints=[Keypoint(id=0, name="0", x=0.0, y=0.0, visibility=1.0)],
                robot_status=RobotStatus(localized=True, x=0.0, y=0.0, yaw_deg=90.0),
                target_size=(w, h),
            )
            self.assertEqual(out.size, (w, h))

    def test_failed_navigation_displayed_in_ui(self):
        """Verifies VM failed navigation status and mission retry status are visibly shown in GUI."""
        self.robot.active_vm_ip = "127.0.0.1"
        self.robot._last_ping_time = __import__("time").time()
        self.robot.latest_status = RobotStatus(
            localized=True,
            x=1.0,
            y=1.0,
            yaw_deg=0.0,
            nav_state="failed",
            nav_message="Navigation stuck (no movement for 10s)",
            goal_seq=7,
            goal_name="A",
        )
        self.gui._update_loop()
        self.root.update()

        self.assertIn("FAILED", self.gui.lbl_robot_conn.cget("text"))
        self.assertIn("A", self.gui.lbl_robot_conn.cget("text"))
        self.assertIn("Failed to get to A", self.gui.lbl_status.cget("text"))
        self.assertEqual(self.gui.lbl_status.cget("fg"), "#991b1b")

        # Also check mission mid-queue retry message
        self.mission.set_status("Failed to get to A, continuing with B")
        self.gui._update_loop()
        self.root.update()
        self.assertEqual(self.gui.lbl_status.cget("text"), "Status: Failed to get to A, continuing with B")
        self.assertEqual(self.gui.lbl_status.cget("fg"), "#991b1b")

    def test_measurement_overlay_uses_disparity_depth_map_with_size_above_bbox(self):
        """Verifies Level 3 measurement overlay draws bbox + size over disparity/depth map instead of RGB."""
        from laptop.vision_overlay import draw_measurement_overlay
        import cv2

        # Create a pure black RGB frame; mock SizeLink produces a vivid non-black Turbo depth map
        black_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        ok_png, png_buf = cv2.imencode(".png", np.hstack((black_rgb, black_rgb)))
        self.assertTrue(ok_png)

        meas = self.size.measure(png_buf.tobytes(), point_name="B")
        self.assertTrue(meas.ok)
        self.assertIsNotNone(meas.depth_jpeg_b64)

        rendered = draw_measurement_overlay(black_rgb, meas, point_name="B")
        self.assertEqual(rendered.shape, black_rgb.shape)
        # Upper-left background outside bbox/banner would be (0,0,0) if RGB were used, but non-zero on depth map
        self.assertGreater(float(rendered[20:80, 20:120].mean()), 15.0)


if __name__ == "__main__":
    unittest.main()
