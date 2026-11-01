"""Lightweight, zero-dependency .env loader for Stage 2 laptop tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

_STAGE2_SYSTEM_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT_DIR = _STAGE2_SYSTEM_DIR.parent.parent
DEFAULT_ENV_PATHS: tuple[Path, ...] = (
    _STAGE2_SYSTEM_DIR / ".env",
    _REPO_ROOT_DIR / ".env",
)

_LOADED = False


def load_env_files(paths: Iterable[Path] = DEFAULT_ENV_PATHS) -> None:
    """Populate os.environ from .env files without overriding non-empty env vars."""
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1].strip()
            if key and value and not os.environ.get(key, "").strip():
                os.environ[key] = value


def ensure_env_loaded() -> None:
    """Load default .env files at most once per Python process."""
    global _LOADED
    if not _LOADED:
        _LOADED = True
        load_env_files()

