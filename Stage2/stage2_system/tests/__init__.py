"""Tests package setup for Stage 2 System."""

from pathlib import Path
import sys

# Ensure stage2_system root is in sys.path
_sys_root = Path(__file__).resolve().parent.parent
if str(_sys_root) not in sys.path:
    sys.path.insert(0, str(_sys_root))

