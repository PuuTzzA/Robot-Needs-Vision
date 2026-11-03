"""Unit tests for the Audio Announcement subsystem."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys_root = Path(__file__).resolve().parent.parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from laptop.audio_announcer import AudioAnnouncer


class TestAudioAnnouncer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.announcer = AudioAnnouncer(audio_dir=self.temp_dir, enabled=False)

    def tearDown(self):
        self.announcer.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_announce_level_start_sequence(self):
        self.announcer.announce_level_start(1, wait=False)
        self.assertIn("What's up Beijing!", self.announcer.history)
        self.assertIn("Starting Level 1", self.announcer.history)

    def test_announce_going_to_point(self):
        self.announcer.announce_going_to_point("B", is_retry=False, wait=False)
        self.assertIn("Going to Point B", self.announcer.history)

        self.announcer.announce_going_to_point("0", is_retry=False, wait=False)
        self.assertIn("Going to Point 0", self.announcer.history)

    def test_announce_retrying_point(self):
        self.announcer.announce_going_to_point("C", is_retry=True, wait=False)
        self.assertIn("Retrying to go to Point C", self.announcer.history)

        self.announcer.announce_retrying_point("D", wait=False)
        self.assertIn("Retrying to go to Point D", self.announcer.history)

    def test_announce_failed_point_with_detour(self):
        self.announcer.announce_failed_point(failed_pt="B", proceeding_to="D", wait=False)
        self.assertIn("Failed to go to Point B, proceeding to Point D", self.announcer.history)

    def test_announce_failed_point_single(self):
        self.announcer.announce_failed_point(failed_pt="B", proceeding_to=None, wait=False)
        self.assertIn("Failed to go to Point B", self.announcer.history)

    def test_announce_level_finished_with_dynamic_result(self):
        dynamic_answer = "The red bottle is 15.2 centimeters tall at Point B."
        self.announcer.announce_level_finished(3, dynamic_result=dynamic_answer, wait=False)
        self.assertIn("Finished Level 3", self.announcer.history)
        self.assertIn(dynamic_answer, self.announcer.history)

    def test_case_insensitive_clip_resolution(self):
        # Create lowercase clip
        clip_file = self.temp_dir / "b.mp3"
        clip_file.write_bytes(b"dummy")

        # Query uppercase
        found = self.announcer.find_clip("B.mp3")
        self.assertIsNotNone(found)
        self.assertEqual(found.name.lower(), "b.mp3")

    def test_clip_availability_check(self):
        (self.temp_dir / "going_to_point.mp3").write_bytes(b"dummy")
        (self.temp_dir / "A.mp3").write_bytes(b"dummy")

        self.assertTrue(self.announcer.are_clips_available(["going_to_point.mp3", "A.mp3"]))
        self.assertFalse(self.announcer.are_clips_available(["going_to_point.mp3", "B.mp3"]))

    def test_taking_photo_announcements(self):
        self.announcer.announce_taking_photo_of_tasks(wait=False)
        self.assertIn("Taking photo of tasks", self.announcer.history)

        self.announcer.announce_taking_photo_of_point("B", wait=False)
        self.assertIn("Taking photo of Point B", self.announcer.history)

        self.announcer.announce_taking_photo_of_point("0", wait=False)
        self.assertIn("Taking photo of Point 0", self.announcer.history)

    def test_returning_to_start_announcement(self):
        self.announcer.announce_returning_to_start(wait=False)
        self.assertIn("Returning to start position", self.announcer.history)

    def test_alias_resolution(self):
        (self.temp_dir / "wazzup_beijing.mp3").write_bytes(b"dummy")
        (self.temp_dir / "zero.mp3").write_bytes(b"dummy")
        (self.temp_dir / "failed_to_go_to_point.mp3").write_bytes(b"dummy")

        # Querying with alternate/short alias
        self.assertIsNotNone(self.announcer.find_clip("whats_up_beijing.mp3"))
        self.assertIsNotNone(self.announcer.find_clip("0.mp3"))
        self.assertIsNotNone(self.announcer.find_clip("failed_point.mp3"))

    def test_real_audio_directory_has_all_clips(self):
        """Verify that the real Stage2/stage2_system/laptop/audio folder has all required clips."""
        real_audio_dir = sys_root / "laptop" / "audio"
        real_announcer = AudioAnnouncer(audio_dir=real_audio_dir, enabled=False)

        # Check all core action clips
        expected_actions = [
            "wazzup_beijing.mp3",
            "starting_level.mp3",
            "finished_level.mp3",
            "going_to_point.mp3",
            "retrying_to_go_to_point.mp3",
            "failed_to_go_to_point.mp3",
            "proceeding_to_point.mp3",
            "returning_to_start.mp3",
            "taking_photo_of_point.mp3",
            "taking_photo_of_tasks.mp3",
        ]
        for act in expected_actions:
            self.assertIsNotNone(
                real_announcer.find_clip(act),
                f"Missing action clip: {act} in {real_audio_dir}",
            )

        # Check numbers
        for num in ("zero", "one", "two", "three"):
            self.assertIsNotNone(
                real_announcer.find_clip(f"{num}.mp3"),
                f"Missing number clip: {num}.mp3 in {real_audio_dir}",
            )

        # Check letters A through I
        for letter in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
            self.assertIsNotNone(
                real_announcer.find_clip(f"{letter}.mp3"),
                f"Missing letter clip: {letter}.mp3 in {real_audio_dir}",
            )


if __name__ == "__main__":
    unittest.main()

