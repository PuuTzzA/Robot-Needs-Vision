"""Deep Learning Stereo Estimator using CREStereo ONNX with GPU acceleration."""

from __future__ import annotations
import ctypes
import glob
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import onnxruntime as ort

from .base import BaseStereoEstimator


def _setup_cuda_and_cudnn() -> None:
    """Preload cuDNN and CUDA libraries if installed via pip nvidia-* packages."""
    search_dirs = [p for p in sys.path if "site-packages" in p]
    for base in search_dirs:
        # Check for nvidia/cudnn/lib and nvidia/cuda_runtime/lib
        for pkg in ("cudnn", "cuda_runtime", "cublas"):
            lib_dir = os.path.join(base, "nvidia", pkg, "lib")
            if os.path.isdir(lib_dir):
                if lib_dir not in os.environ.get("LD_LIBRARY_PATH", ""):
                    os.environ["LD_LIBRARY_PATH"] = (
                        f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                    )
                for so in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
                    try:
                        ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass


_setup_cuda_and_cudnn()


class DeepCREStereoEstimator(BaseStereoEstimator):
    """Deep learning stereo matching using CREStereo ONNX.
    
    CREStereo uses cascaded recurrent GRU updates with adaptive correlation,
    providing 100% dense (zero-hole) depth maps even across textureless lab surfaces.
    """

    def __init__(
        self,
        model_path: str | Path = "models/crestereo_embedded.onnx",
        crop_to_4_3: bool = True,
        use_cuda: bool = True,
    ) -> None:
        self._model_path = Path(model_path)
        self._crop_to_4_3 = crop_to_4_3
        self._use_cuda = use_cuda
        self._session: ort.InferenceSession | None = None
        self._input_names: list[str] = []
        self._input_shapes: list[tuple[int, int]] = []
        self._crop_x_offset: int = 0

    @property
    def name(self) -> str:
        return "Deep CREStereo (CUDA/ONNX)"

    def initialize(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(f"CREStereo model not found at {self._model_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        providers = []
        if self._use_cuda and "CUDAExecutionProvider" in available_providers:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4 GB
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                )
            )
        providers.append("CPUExecutionProvider")

        print(f"[CREStereo] Available ONNX providers: {available_providers}")
        self._session = ort.InferenceSession(
            str(self._model_path), session_options, providers=providers
        )
        print(f"[CREStereo] Active ONNX providers: {self._session.get_providers()}")

        inputs = self._session.get_inputs()
        self._input_names = [inp.name for inp in inputs]
        self._input_shapes = []
        for inp in inputs:
            h = int(inp.shape[2])
            w = int(inp.shape[3])
            self._input_shapes.append((h, w))

    def _prepare_tensor(self, img_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
        resized = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None]  # (1, 3, H, W)
        return (tensor.astype(np.float32) / 255.0).copy()

    def _center_crop_4_3(self, img: np.ndarray) -> tuple[np.ndarray, int]:
        h, w = img.shape[:2]
        target_w = min(w, int(round(h * 4.0 / 3.0)))
        x_left = (w - target_w) // 2
        return img[:, x_left : x_left + target_w], x_left

    def estimate_disparity(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
    ) -> np.ndarray:
        if self._session is None:
            self.initialize()

        orig_h, orig_w = left_rectified.shape[:2]

        if self._crop_to_4_3:
            left_in, x_offset = self._center_crop_4_3(left_rectified)
            right_in = right_rectified[:, x_offset : x_offset + left_in.shape[1]]
            self._crop_x_offset = x_offset
        else:
            left_in, right_in = left_rectified, right_rectified
            self._crop_x_offset = 0

        crop_h, crop_w = left_in.shape[:2]

        feed = {}
        for idx, (name, (target_h, target_w)) in enumerate(
            zip(self._input_names, self._input_shapes)
        ):
            lower_name = name.lower()
            if "left" in lower_name:
                src = left_in
            elif "right" in lower_name:
                src = right_in
            else:
                src = left_in if idx % 2 == 0 else right_in
            feed[name] = self._prepare_tensor(src, target_h, target_w)

        output = self._session.run(None, feed)[0]
        disp_raw = np.squeeze(output).astype(np.float32)

        # Disparity scale correction for resizing
        model_out_h, model_out_w = disp_raw.shape
        disp_cropped = cv2.resize(
            disp_raw, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
        )
        disp_cropped *= crop_w / model_out_w

        if self._crop_to_4_3:
            # Place cropped disparity into full frame to align with original coordinates
            full_disparity = np.full((orig_h, orig_w), np.nan, dtype=np.float32)
            full_disparity[:, self._crop_x_offset : self._crop_x_offset + crop_w] = disp_cropped
            return full_disparity

        return disp_cropped

