"""Unit tests for center-pixel depth distance estimation in Stage 2."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))
ssh_dir = sys_root / "ssh"
if str(ssh_dir) not in sys.path:
    sys.path.insert(0, str(ssh_dir))

from ssh.measure import average_center_distance, robust_measurement


class TestCenterDepthEstimation(unittest.TestCase):
    def test_average_center_distance_constant_depth(self) -> None:
        depth = np.full((100, 100), 2.5, dtype=np.float32)
        bbox = [20, 20, 80, 80]
        dist = average_center_distance(depth, bbox)
        self.assertIsNotNone(dist)
        self.assertAlmostEqual(dist, 2.5, places=3)

    def test_average_center_distance_samples_center_region(self) -> None:
        # 100x100 depth map where center is 1.5m and perimeter is 5.0m
        depth = np.full((100, 100), 5.0, dtype=np.float32)
        # Center of bbox [20, 20, 80, 80] is (50, 50)
        depth[40:61, 40:61] = 1.5
        bbox = [20, 20, 80, 80]
        dist = average_center_distance(depth, bbox, center_fraction=0.3)
        self.assertIsNotNone(dist)
        # Central patch is inside [40:61, 40:61] so average should be 1.5m, not 5.0m
        self.assertAlmostEqual(dist, 1.5, places=2)

    def test_average_center_distance_with_mask(self) -> None:
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        # Background inside the center ROI has 7.0m, but mask isolates the 2.0m object
        depth[45:55, 45:55] = 7.0
        mask = np.ones((100, 100), dtype=bool)
        mask[45:55, 45:55] = False  # hole in mask at center

        bbox = [20, 20, 80, 80]
        dist = average_center_distance(depth, bbox, mask=mask, center_fraction=0.4)
        self.assertIsNotNone(dist)
        self.assertAlmostEqual(dist, 2.0, places=2)

    def test_average_center_distance_invalid_pixels(self) -> None:
        depth = np.full((100, 100), np.nan, dtype=np.float32)
        bbox = [20, 20, 80, 80]
        dist = average_center_distance(depth, bbox)
        self.assertIsNone(dist)

    def test_robust_measurement_uses_center_distance(self) -> None:
        # Create 100x100 depth map
        depth = np.full((100, 100), 4.0, dtype=np.float32)
        # Mask covers [20:80, 20:80]
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        # Object center has depth 2.5m
        depth[40:60, 40:60] = 2.5

        dist_m, w_cm, h_cm, valid_pix = robust_measurement(
            mask=mask,
            depth=depth,
            fx=500.0,
            fy=500.0,
            cx=50.0,
            cy=50.0,
            bbox=[20, 20, 80, 80],
        )
        self.assertIsNotNone(dist_m)
        # Center average should yield 2.5m
        self.assertAlmostEqual(dist_m, 2.5, places=2)
        self.assertIsNotNone(w_cm)
        self.assertIsNotNone(h_cm)


if __name__ == "__main__":
    unittest.main()

