"""Abstract base class for all stereo depth estimators in Stage 1."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseStereoEstimator(ABC):
    """Abstract interface defining the contract for stereo depth estimation.
    
    Subclasses implement specific algorithms (traditional block matching,
    deep learning architectures like CREStereo, FoundationStereo, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name of the estimator."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Load weights, initialize inference engines, allocate GPU tensors, or configure filters."""
        pass

    @abstractmethod
    def estimate_disparity(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
    ) -> np.ndarray:
        """Compute disparity from an epipolar-rectified stereo pair.

        Args:
            left_rectified: Left camera rectified BGR image (H, W, 3).
            right_rectified: Right camera rectified BGR image (H, W, 3).

        Returns:
            disparity: 2D float32 array (H, W) where each value represents horizontal
                       disparity in pixels (d = x_left - x_right).
                       Invalid/void/occluded pixels must be np.nan or <= 0.
        """
        pass

