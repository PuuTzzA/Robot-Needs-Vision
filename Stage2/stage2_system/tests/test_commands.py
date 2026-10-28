"""Tests for command parsing and dispatch."""

from pathlib import Path
import sys
import unittest

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from laptop.camera import CameraManager
from laptop.commands import CommandDispatcher
from laptop.mission import MissionManager
from laptop.robot_link import RobotLink
from laptop.size_link import SizeLink
from shared.keypoints import KeypointStore


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.cam = CameraManager(is_mock=True)
        self.robot = RobotLink(vm_ip="127.0.0.1")
        self.size = SizeLink(is_mock=True)
        self.kps = KeypointStore()
        self.kps.update_from_payload({
            "keypoints": [
                {"id": 0, "name": "0", "x": 0.0, "y": 0.0, "visibility": 1.0},
                {"id": 1, "name": "B", "x": 2.0, "y": 3.0, "visibility": 0.8},
            ]
        })
        self.mission = MissionManager(self.cam, self.robot, self.size, self.kps)
        self.dispatcher = CommandDispatcher(self.mission)

    def test_help_and_unknown(self):
        res = self.dispatcher.execute("Help")
        self.assertIn("AVAILABLE MISSION CONTROL COMMANDS", res)

        bad = self.dispatcher.execute("FOOBAR")
        self.assertIn("Unknown command", bad)

    def test_level_command_syntax(self):
        err = self.dispatcher.execute("Level")
        self.assertIn("Usage", err)

        bad_lvl = self.dispatcher.execute("Level 9")
        self.assertIn("Usage", bad_lvl)

    def test_points_command(self):
        pts = self.dispatcher.execute("Points")
        self.assertIn("B", pts)
        self.assertIn("0", pts)

    def test_teleop_command(self):
        res = self.dispatcher.execute("W")
        self.assertIn("Teleop", res)

    def test_ping_mock(self):
        res = self.dispatcher.execute("Ping SSH")
        self.assertIn("Ping SSH", res)


if __name__ == "__main__":
    unittest.main()

