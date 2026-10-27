#!/usr/bin/env python3
"""Unit tests for Stage 2 Keypoint Collector storage and schema logic."""

import json
from pathlib import Path
import tempfile
import unittest


class MockKeypointCollector:
    """Standalone logic runner testing keypoint persistence and schema without live ROS hardware."""

    def __init__(self, points_path: Path):
        self.points_path = points_path
        self.keypoints: list[dict] = []
        self.next_id = 0
        self._load_keypoints()

    def _load_keypoints(self) -> None:
        if not self.points_path.exists():
            return
        data = json.loads(self.points_path.read_text())
        if isinstance(data, list):
            self.keypoints = data
        elif isinstance(data, dict):
            self.keypoints = data.get("keypoints", [])
        max_id = -1
        for pt in self.keypoints:
            if "id" in pt and isinstance(pt["id"], int) and pt["id"] > max_id:
                max_id = pt["id"]
        self.next_id = max_id + 1

    def persist(self) -> None:
        self.points_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.points_path.with_suffix(".tmp")
        payload = {
            "keypoints": self.keypoints,
        }
        temp_file.write_text(json.dumps(payload, indent=2) + "\n")
        temp_file.replace(self.points_path)

    def add_keypoint(self, name: str, visibility: float, x: float, y: float, yaw: float) -> dict:
        name = name.strip().upper()
        if not (0.0 <= visibility <= 1.0):
            raise ValueError(f"Visibility must be in [0, 1], got {visibility}")

        item_id = self.next_id
        self.next_id += 1

        entry = {
            "id": item_id,
            "name": name,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "yaw_degrees": round(float(yaw), 2),
            "position": {
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "z": 0.0,
            },
            "rotation": {
                "yaw_degrees": round(float(yaw), 2),
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "visibility": round(float(visibility), 3),
        }
        self.keypoints.append(entry)
        self.persist()
        return entry

    def delete_keypoint(self, target_id: int) -> bool:
        init_len = len(self.keypoints)
        self.keypoints = [pt for pt in self.keypoints if pt.get("id") != target_id]
        if len(self.keypoints) < init_len:
            self.persist()
            return True
        return False


class TestKeypointStorage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.points_file = Path(self.temp_dir.name) / "test_points.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_auto_increment_id(self):
        collector = MockKeypointCollector(self.points_file)
        e1 = collector.add_keypoint("A", 0.5, 1.0, 2.0, 45.0)
        self.assertEqual(e1["id"], 0)
        self.assertEqual(e1["name"], "A")
        self.assertEqual(e1["visibility"], 0.5)

        e2 = collector.add_keypoint("B", 1.0, 2.0, 3.0, 90.0)
        self.assertEqual(e2["id"], 1)

    def test_multiple_entries_same_keypoint(self):
        collector = MockKeypointCollector(self.points_file)
        e1 = collector.add_keypoint("B", 1.0, 2.1, 0.45, 180.0)
        e2 = collector.add_keypoint("B", 0.3, 2.45, 0.62, 175.2)

        self.assertEqual(e1["id"], 0)
        self.assertEqual(e1["name"], "B")
        self.assertEqual(e1["visibility"], 1.0)

        self.assertEqual(e2["id"], 1)
        self.assertEqual(e2["name"], "B")
        self.assertEqual(e2["visibility"], 0.3)

        self.assertEqual(len(collector.keypoints), 2)

    def test_visibility_bounds(self):
        collector = MockKeypointCollector(self.points_file)
        with self.assertRaises(ValueError):
            collector.add_keypoint("C", -0.1, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            collector.add_keypoint("C", 1.05, 0.0, 0.0, 0.0)

    def test_persistence_reload(self):
        collector = MockKeypointCollector(self.points_file)
        collector.add_keypoint("A", 0.8, 1.25, -0.83, 89.5)
        collector.add_keypoint("B", 0.4, 3.0, 1.5, 0.0)

        # Reload from same file
        collector2 = MockKeypointCollector(self.points_file)
        self.assertEqual(len(collector2.keypoints), 2)
        self.assertEqual(collector2.next_id, 2)
        self.assertEqual(collector2.keypoints[0]["name"], "A")
        self.assertEqual(collector2.keypoints[0]["visibility"], 0.8)

        # Add third item
        e3 = collector2.add_keypoint("C", 0.95, -1.0, 2.0, 270.0)
        self.assertEqual(e3["id"], 2)

    def test_delete_keypoint(self):
        collector = MockKeypointCollector(self.points_file)
        collector.add_keypoint("A", 0.5, 1.0, 1.0, 0.0)
        collector.add_keypoint("B", 0.6, 2.0, 2.0, 0.0)
        collector.add_keypoint("C", 0.7, 3.0, 3.0, 0.0)

        self.assertTrue(collector.delete_keypoint(1))
        self.assertEqual(len(collector.keypoints), 2)
        ids = [pt["id"] for pt in collector.keypoints]
        self.assertEqual(ids, [0, 2])

        # Verify persisted file
        collector2 = MockKeypointCollector(self.points_file)
        self.assertEqual(len(collector2.keypoints), 2)


if __name__ == "__main__":
    unittest.main()

