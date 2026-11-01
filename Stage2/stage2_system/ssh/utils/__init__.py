"""Utility package for Stage 1 Stereo Vision."""

from .calibration_loader import StereoCalibration
from .depth_converter import (
    disparity_to_depth,
    colorize_disparity,
    colorize_depth,
    attach_depth_legend,
    calculate_compactness,
    sample_roi_depth,
)
try:
    from .frame_streamer import NetworkStereoReceiver
except ImportError:
    NetworkStereoReceiver = None

__all__ = [
    "StereoCalibration",
    "disparity_to_depth",
    "colorize_disparity",
    "colorize_depth",
    "attach_depth_legend",
    "calculate_compactness",
    "sample_roi_depth",
]
if NetworkStereoReceiver is not None:
    __all__.append("NetworkStereoReceiver")

