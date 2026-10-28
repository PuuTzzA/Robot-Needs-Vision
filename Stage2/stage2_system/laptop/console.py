"""Thread-safe console redirector for GUI terminal output."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
import threading
import time
from typing import Optional


class ConsoleRedirector:
    """Tees stdout and stderr into a memory queue for the GUI and persists to a log file."""

    def __init__(self, log_dir: Optional[Path] = None, max_lines: int = 2000) -> None:
        self.max_lines = max_lines
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.lock = threading.Lock()
        self._listeners = []

        self.log_file = None
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_dir / "mission_control.log", "a", encoding="utf-8", buffering=1)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

    def start(self) -> None:
        sys.stdout = self  # type: ignore
        sys.stderr = self  # type: ignore

    def stop(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass

    def write(self, s: str) -> None:
        if not s:
            return

        # Write to original stdout
        self._orig_stdout.write(s)
        self._orig_stdout.flush()

        if self.log_file:
            self.log_file.write(s)

        with self.lock:
            # Split newlines into lines
            for part in s.splitlines():
                if part.strip():
                    ts = time.strftime("%H:%M:%S")
                    line_formatted = f"[{ts}] {part}"
                    self.lines.append(line_formatted)
                    for cb in self._listeners:
                        try:
                            cb(line_formatted)
                        except Exception:
                            pass

    def flush(self) -> None:
        self._orig_stdout.flush()
        if self.log_file:
            self.log_file.flush()

    def add_listener(self, cb) -> None:
        with self.lock:
            self._listeners.append(cb)

    def get_recent_lines(self, count: int = 100) -> list[str]:
        with self.lock:
            return list(self.lines)[-count:]

