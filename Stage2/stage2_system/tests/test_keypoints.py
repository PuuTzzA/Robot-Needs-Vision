"""Unit tests for KeypointStore and selection logic."""

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from shared.keypoints import Keypoint, KeypointStore, parse_keypoints_json
from shared.protocol import VISIBILITY_DEFAULT, VISIBILITY_HIGH


class TestKeypointStore(unittest.TestCase):
    def setUp(self):
        self.sample_data = {
            "keypoints": [
                {
                    "id": 0,
                    "name": "0",
                    "x": 5.0,
                    "y": 5.0,
                    "yaw_degrees": 0.0,
                    "visibility": 0.95,
                },
                {
                    "id": 1,
                    "name": "B",
                    "x": 2.0,
                    "y": 2.0,
                    "yaw_degrees": 45.0,
                    "visibility": 0.20,  # low vis, close to (1, 1)
                },
                {
                    "id": 2,
                    "name": "B",
                    "x": 2.5,
                    "y": 2.5,
                    "yaw_degrees": 45.0,
                    "visibility": 0.85,  # high vis, slightly further
                },
                {
                    "id": 3,
                    "name": "C",
                    "x": 8.0,
                    "y": 1.0,
                    "yaw_degrees": 90.0,
                    "visibility": 0.50,  # only entry for C
                },
            ]
        }

    def test_parse_json(self):
        text = json.dumps(self.sample_data)
        kps, sha = parse_keypoints_json(text)
        self.assertEqual(len(kps), 4)
        self.assertTrue(len(sha) > 0)
        self.assertEqual(kps[0].name, "0")
        self.assertEqual(kps[1].name, "B")

    def test_object_point_names(self):
        text = json.dumps(self.sample_data)
        store = KeypointStore()
        store.update_from_payload(self.sample_data)
        names = store.object_point_names()
        self.assertEqual(names, ["B", "C"])
        self.assertNotIn("0", names)

    def test_select_entry_visibility_rule(self):
        store = KeypointStore()
        store.update_from_payload(self.sample_data)

        # 1. When min_visibility is 0.1, both B entries qualify.
        # Robot is at (1.0, 1.0). Entry 1 at (2.0, 2.0) is closer than Entry 2 at (2.5, 2.5).
        selected_low = store.select_entry("B", robot_xy=(1.0, 1.0), min_visibility=0.10)
        self.assertIsNotNone(selected_low)
        self.assertEqual(selected_low.id, 1)

        # 2. When min_visibility is 0.75 (e.g. Level 3 with height requirement),
        # only Entry 2 (vis 0.85) qualifies.
        selected_high = store.select_entry("B", robot_xy=(1.0, 1.0), min_visibility=0.75)
        self.assertIsNotNone(selected_high)
        self.assertEqual(selected_high.id, 2)

        # 3. For C, visibility is 0.50. When asking for min_visibility 0.75,
        # none qualify -> fallback to highest visibility available (Entry 3).
        selected_fallback = store.select_entry("C", robot_xy=(0.0, 0.0), min_visibility=0.75)
        self.assertIsNotNone(selected_fallback)
        self.assertEqual(selected_fallback.id, 3)

    def test_greedy_order(self):
        store = KeypointStore()
        store.update_from_payload(self.sample_data)
        order = store.order_greedy(["C", "B"], start_xy=(0.0, 0.0))
        # Start (0, 0) is closer to B (2, 2) than C (8, 1)
        self.assertEqual(order, ["B", "C"])

    def test_mtime_hot_reload_and_no_write(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(self.sample_data, f)
            temp_path = Path(f.name)

        try:
            store = KeypointStore(temp_path)
            self.assertEqual(len(store.get_all()), 4)
            orig_mtime = temp_path.stat().st_mtime

            # Query store - must NOT change file or write anything
            _ = store.select_entry("B")
            self.assertEqual(temp_path.stat().st_mtime, orig_mtime)

            # Now modify file externally (as if camera_keypoints.py recorded a new keypoint)
            time.sleep(0.05)
            new_data = dict(self.sample_data)
            new_data["keypoints"].append({
                "id": 4,
                "name": "D",
                "x": 3.0,
                "y": 7.0,
                "yaw_degrees": 180.0,
                "visibility": 0.8,
            })
            temp_path.write_text(json.dumps(new_data), encoding="utf-8")

            # Store should detect change on next query
            names = store.object_point_names()
            self.assertIn("D", names)
            self.assertEqual(len(store.get_all()), 5)
        finally:
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
