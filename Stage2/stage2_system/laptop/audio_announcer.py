"""Audio Announcement Subsystem for Stage 2 Mission Control.

Provides composable, modular audio playback using pre-recorded MP3 clips
with automatic fallback to local Windows SAPI Text-to-Speech (TTS) for
unknown strings and dynamic open-ended LLM answers.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Optional


@dataclass
class _AudioItem:
    clip_names: list[str]
    fallback_text: str
    done_event: Optional[threading.Event] = None
    force_tts_only: bool = False


class AudioAnnouncer:
    """Thread-safe, non-blocking audio announcement manager."""

    CLIP_ALIASES: dict[str, list[str]] = {
        "failed_point": ["failed_to_go_to_point", "failed_point"],
        "failed_to_go_to_point": ["failed_to_go_to_point", "failed_point"],
        "proceeding_to": ["proceeding_to_point", "proceeding_to"],
        "proceeding_to_point": ["proceeding_to_point", "proceeding_to"],
        "retrying_point": ["retrying_to_go_to_point", "retrying_point", "retrying_to_point"],
        "retrying_to_go_to_point": ["retrying_to_go_to_point", "retrying_point", "retrying_to_point"],
        "0": ["zero", "0"],
        "zero": ["zero", "0"],
        "1": ["one", "1"],
        "one": ["one", "1"],
        "2": ["two", "2"],
        "two": ["two", "2"],
        "3": ["three", "3"],
        "three": ["three", "3"],
    }

    def __init__(
        self,
        audio_dir: Optional[Path] = None,
        enabled: bool = True,
        volume: float = 1.0,
        voice_name: Optional[str] = None,
    ) -> None:
        self.audio_dir = Path(audio_dir) if audio_dir else Path(__file__).resolve().parent / "audio"
        self.enabled = bool(enabled)
        self.volume = max(0.0, min(1.0, float(volume)))
        self.voice_name = voice_name

        self._queue: queue.Queue[Optional[_AudioItem]] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_mci_alias: Optional[str] = None
        self._mci_lock = threading.Lock()

        # Recorded history for assertions in automated tests
        self.history: list[str] = []
        self._history_lock = threading.Lock()

        if self.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AudioAnnouncerWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Cleanly terminate worker and abort active playback."""
        self._stop_event.set()
        self.stop_current()
        self._queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._worker_thread = None

    def stop_current(self) -> None:
        """Immediately abort any currently playing MCI audio clip."""
        with self._mci_lock:
            if self._current_mci_alias:
                try:
                    ctypes.windll.winmm.mciSendStringW(
                        f"stop {self._current_mci_alias}", None, 0, None
                    )
                    ctypes.windll.winmm.mciSendStringW(
                        f"close {self._current_mci_alias}", None, 0, None
                    )
                except Exception:
                    pass
                self._current_mci_alias = None

    def find_clip(self, clip_name: str) -> Optional[Path]:
        """Resolve an audio filename case-insensitively within audio_dir, supporting aliases."""
        if not self.audio_dir.is_dir():
            return None

        # Direct match
        direct = self.audio_dir / clip_name
        if direct.is_file():
            return direct

        stem = Path(clip_name).stem.lower()
        candidates_stems = [stem]
        if stem in self.CLIP_ALIASES:
            candidates_stems.extend([s for s in self.CLIP_ALIASES[stem] if s != stem])

        for c_stem in candidates_stems:
            for ext in (".mp3", ".wav"):
                target_lower = f"{c_stem}{ext}".lower()
                try:
                    for candidate in self.audio_dir.iterdir():
                        if candidate.is_file() and candidate.name.lower() == target_lower:
                            return candidate
                except OSError:
                    break
        return None

    def are_clips_available(self, clip_names: list[str]) -> bool:
        """Return True only if every clip in the sequence exists on disk."""
        return all(self.find_clip(name) is not None for name in clip_names)

    def _record_history(self, text: str) -> None:
        with self._history_lock:
            self.history.append(text)

    # -------------------------------------------------------------------------
    # Public Announcement APIs
    # -------------------------------------------------------------------------

    def play_sequence(
        self,
        clip_names: list[str],
        fallback_text: str,
        wait: bool = False,
    ) -> None:
        """Enqueue a sequence of MP3 clips to play back-to-back, falling back to TTS if any clip is missing."""
        self._record_history(fallback_text)
        if not self.enabled:
            print(f"[Audio Muted] {fallback_text}")
            return

        done_event = threading.Event() if wait else None
        item = _AudioItem(
            clip_names=clip_names,
            fallback_text=fallback_text,
            done_event=done_event,
            force_tts_only=False,
        )
        self._queue.put(item)
        if done_event:
            done_event.wait()

    def speak(self, text: str, wait: bool = False) -> None:
        """Speak arbitrary text directly using on-the-fly local TTS."""
        text = text.strip()
        if not text:
            return
        self._record_history(text)
        if not self.enabled:
            print(f"[Audio Muted] {text}")
            return

        done_event = threading.Event() if wait else None
        item = _AudioItem(
            clip_names=[],
            fallback_text=text,
            done_event=done_event,
            force_tts_only=True,
        )
        self._queue.put(item)
        if done_event:
            done_event.wait()

    def _point_token(self, pt: str) -> str:
        """Return the clip filename for a keypoint identifier ('0' -> 'zero.mp3', 'A' -> 'a.mp3')."""
        p = pt.strip().lower()
        if p in ("0", "zero"):
            return "zero.mp3"
        return f"{p}.mp3"

    def _level_token(self, level: int) -> str:
        """Return the clip filename for a level number ('1' -> 'one.mp3', etc.)."""
        words = {1: "one", 2: "two", 3: "three"}
        return f"{words.get(level, str(level))}.mp3"

    def announce_level_start(self, level: int, wait: bool = True) -> None:
        """Announce 'Starting Level X'."""
        lvl_tok = self._level_token(level)
        self.play_sequence(
            clip_names=["starting_level.mp3", lvl_tok],
            fallback_text=f"Starting Level {level}",
            wait=wait,
        )

    def announce_going_to_point(self, pt: str, is_retry: bool = False, wait: bool = False) -> None:
        """Announce 'Going to Point X' or 'Retrying to go to Point X'."""
        pt_clean = pt.strip().upper()
        if is_retry:
            self.announce_retrying_point(pt_clean, wait=wait)
            return

        pt_tok = self._point_token(pt_clean)
        self.play_sequence(
            clip_names=["going_to_point.mp3", pt_tok],
            fallback_text=f"Going to Point {pt_clean}",
            wait=wait,
        )

    def announce_retrying_point(self, pt: str, wait: bool = False) -> None:
        """Announce 'Retrying to go to Point X'."""
        pt_clean = pt.strip().upper()
        pt_tok = self._point_token(pt_clean)
        self.play_sequence(
            clip_names=["retrying_to_go_to_point.mp3", pt_tok],
            fallback_text=f"Retrying to go to Point {pt_clean}",
            wait=wait,
        )

    def announce_failed_point(
        self,
        failed_pt: str,
        proceeding_to: Optional[str] = None,
        wait: bool = False,
    ) -> None:
        """Announce 'Failed to go to Point X, proceeding to Y'."""
        f_clean = failed_pt.strip().upper()
        f_tok = self._point_token(f_clean)
        if proceeding_to:
            p_clean = proceeding_to.strip().upper()
            p_tok = self._point_token(p_clean)
            self.play_sequence(
                clip_names=["failed_to_go_to_point.mp3", f_tok, "proceeding_to_point.mp3", p_tok],
                fallback_text=f"Failed to go to Point {f_clean}, proceeding to Point {p_clean}",
                wait=wait,
            )
        else:
            self.play_sequence(
                clip_names=["failed_to_go_to_point.mp3", f_tok],
                fallback_text=f"Failed to go to Point {f_clean}",
                wait=wait,
            )

    def announce_taking_photo_of_tasks(self, wait: bool = False) -> None:
        """Announce taking photo of task instructions (on laptop screen)."""
        self.play_sequence(
            clip_names=["taking_photo_of_tasks.mp3"],
            fallback_text="Taking photo of tasks",
            wait=wait,
        )

    def announce_taking_photo_of_point(self, pt: str, wait: bool = False) -> None:
        """Announce taking photo of a specific inspection point."""
        pt_clean = pt.strip().upper()
        pt_tok = self._point_token(pt_clean)
        self.play_sequence(
            clip_names=["taking_photo_of_point.mp3", pt_tok],
            fallback_text=f"Taking photo of Point {pt_clean}",
            wait=wait,
        )

    def announce_returning_to_start(self, wait: bool = False) -> None:
        """Announce returning to initial Start position."""
        self.play_sequence(
            clip_names=["returning_to_start.mp3"],
            fallback_text="Returning to start position",
            wait=wait,
        )

    def announce_level_finished(
        self,
        level: int,
        dynamic_result: Optional[str] = None,
        wait: bool = False,
    ) -> None:
        """Announce 'Finished Level X' followed immediately by dynamic LLM result string."""
        # 1. Pre-recorded MP3 composition: finished_level.mp3 + one.mp3/etc
        lvl_tok = self._level_token(level)
        self.play_sequence(
            clip_names=["finished_level.mp3", lvl_tok],
            fallback_text=f"Finished Level {level}",
            wait=True,
        )
        # 2. Dynamic TTS result readout (for open-ended multimodal LLM answer)
        if dynamic_result and dynamic_result.strip():
            self.speak(dynamic_result.strip(), wait=wait)

    # -------------------------------------------------------------------------
    # Worker Thread & Hardware Playback Engines
    # -------------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Background thread consuming audio requests."""
        # Initialize COM in this thread for SAPI Speech API
        sapi_voice = None
        has_com = False
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            has_com = True
            try:
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                if self.voice_name:
                    for v in sapi_voice.GetVoices():
                        if self.voice_name.lower() in v.GetDescription().lower():
                            sapi_voice.Voice = v
                            break
                # Set volume (0 to 100)
                sapi_voice.Volume = int(self.volume * 100)
                # Rate (-10 to 10, 0 is normal)
                sapi_voice.Rate = 0
            except Exception as e:
                print(f"[AudioAnnouncer] [WARN] SAPI Voice init failed: {e}")
        except ImportError:
            pass

        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if item is None:
                    break

                try:
                    self._process_item(item, sapi_voice)
                except Exception as err:
                    print(f"[AudioAnnouncer] [ERROR] Error playing audio: {err}")
                finally:
                    if item.done_event:
                        item.done_event.set()
                    self._queue.task_done()
        finally:
            if has_com:
                try:
                    import pythoncom

                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def _process_item(self, item: _AudioItem, sapi_voice: Optional[object]) -> None:
        if self._stop_event.is_set():
            return

        # Case 1: Force TTS or no clips provided
        if item.force_tts_only or not item.clip_names:
            self._speak_tts(item.fallback_text, sapi_voice)
            return

        # Case 2: Check if all clips exist
        resolved_paths: list[Path] = []
        all_exist = True
        for name in item.clip_names:
            p = self.find_clip(name)
            if p is not None:
                resolved_paths.append(p)
            else:
                all_exist = False
                break

        if all_exist:
            # Play each MP3 clip back-to-back using Windows MCI
            print(f"[Audio MP3] Playing clips: {[p.name for p in resolved_paths]}")
            for path in resolved_paths:
                if self._stop_event.is_set():
                    break
                self._play_mci_file_blocking(path)
        else:
            # Fallback to TTS
            missing = [name for name in item.clip_names if self.find_clip(name) is None]
            print(f"[Audio TTS] Missing {missing}; falling back to TTS: '{item.fallback_text}'")
            self._speak_tts(item.fallback_text, sapi_voice)

    def _play_mci_file_blocking(self, path: Path) -> bool:
        """Play a single audio file (.mp3 / .wav) synchronously using Windows winmm MCI."""
        if sys.platform != "win32":
            time.sleep(0.3)
            return True

        # Convert to 8.3 short path to avoid space/special character issues with MCI
        full_p = str(path.resolve())
        buf = ctypes.create_unicode_buffer(500)
        ctypes.windll.kernel32.GetShortPathNameW(full_p, buf, 500)
        short_p = buf.value or full_p

        alias = f"ann_{time.time_ns()}"
        with self._mci_lock:
            self._current_mci_alias = alias

        try:
            # Open
            cmd_open = f'open "{short_p}" alias {alias}'
            r_open = ctypes.windll.winmm.mciSendStringW(cmd_open, None, 0, None)
            if r_open != 0:
                print(f"[AudioAnnouncer] [WARN] MCI open failed ({r_open}) for {path.name}")
                return False

            # Set volume if supported by MCI
            vol_val = int(self.volume * 1000)
            ctypes.windll.winmm.mciSendStringW(
                f"setaudio {alias} volume to {vol_val}", None, 0, None
            )

            # Play blocking until completion
            cmd_play = f"play {alias} wait"
            ctypes.windll.winmm.mciSendStringW(cmd_play, None, 0, None)
            return True
        finally:
            with self._mci_lock:
                try:
                    ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, None)
                except Exception:
                    pass
                if self._current_mci_alias == alias:
                    self._current_mci_alias = None

    def _speak_tts(self, text: str, sapi_voice: Optional[object]) -> None:
        """Speak text using SAPI SpVoice synchronously."""
        if not text:
            return

        if sapi_voice is not None and sys.platform == "win32":
            try:
                # 0 = SVSFDefault (synchronous in this worker thread)
                sapi_voice.Speak(text, 0)
                return
            except Exception as e:
                print(f"[AudioAnnouncer] [WARN] SAPI Speak failed ({e}); falling back to console.")

        # Fallback simulation if SAPI is unavailable (e.g. non-Windows unit test)
        print(f"[Voice Spoken] {text}")
        time.sleep(0.2)

