"""HITNet Stereo Estimator using ONNX Runtime with GPU acceleration.

HITNet (Hierarchical Iterative Tile Network) is an ultra-fast stereo matching
architecture with planar surface representation and subpixel refinement.
Unlike recurrent models (CREStereo/RAFT), HITNet uses a feedforward design
that achieves high frame rates (30-60+ FPS on modern GPUs).
"""

from __future__ import annotations
import os
from pathlib import Path
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .base import BaseStereoEstimator
from .deep_crestereo import _setup_cuda_and_cudnn

_setup_cuda_and_cudnn()


class HitnetEstimator(BaseStereoEstimator):
    """Deep learning stereo matching using HITNet ONNX."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        use_cuda: bool = True,
        device_id: int = 0,
    ) -> None:
        self._specified_model_path = Path(model_path) if model_path is not None else None
        self._model_path: Path | None = None
        self._use_cuda = use_cuda
        self._device_id = device_id
        self._session: ort.InferenceSession | None = None
        self._num_inputs: int = 0
        self._input_name: str = ""
        self._input_left_name: str = ""
        self._input_right_name: str = ""
        self._output_name: str = ""
        self._model_w: int = 640
        self._model_h: int = 480
        self._nchw: bool = True
        self._active_provider: str = "Uninitialized"

    @property
    def name(self) -> str:
        if self._active_provider == "Uninitialized":
            has_cuda = self._use_cuda and ort is not None and "CUDAExecutionProvider" in ort.get_available_providers()
            prov = "CUDA" if has_cuda else "CPU"
        else:
            prov = "CUDA" if "CUDA" in self._active_provider else "CPU"
        return f"HITNet ({prov}/ONNX)"

    def _resolve_model_path(self) -> Path:
        """Find the HITNet model file from candidate locations."""
        if self._specified_model_path is not None and self._specified_model_path.exists():
            return self._specified_model_path

        candidates = [
            Path(__file__).resolve().parent.parent / "models" / "hitnet.onnx",
            Path(__file__).resolve().parent.parent / "models" / "model_float32.onnx",
            Path(__file__).resolve().parent.parent / "models" / "hitnet_middlebury_480x640.onnx",
            Path(__file__).resolve().parent.parent.parent
            / "binocular_stereo"
            / "Models"
            / "middlebury_d400"
            / "saved_model_480x640"
            / "model_float32.onnx",
            Path(__file__).resolve().parent.parent.parent
            / "binocular_stereo"
            / "Models"
            / "hitnet_middlebury_480x640.onnx",
            Path("Models/middlebury_d400/saved_model_480x640/model_float32.onnx"),
            Path("models/hitnet.onnx"),
        ]

        for p in candidates:
            if p.exists():
                return p.resolve()

        # If not found, return default target path for clear error messaging
        return candidates[0]

    def initialize(self) -> None:
        if ort is None:
            raise ImportError("onnxruntime is not installed in this environment.")

        resolved_path = self._resolve_model_path()
        if not resolved_path.exists():
            default_dir = Path(__file__).resolve().parent.parent / "models"
            raise FileNotFoundError(
                f"HITNet ONNX model not found at '{resolved_path}'.\n"
                f"Please place 'hitnet.onnx' or 'model_float32.onnx' into '{default_dir}' "
                f"or pass '--hitnet-model <path_to_onnx>'."
            )

        self._model_path = resolved_path

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 3
        session_options.intra_op_num_threads = 8
        session_options.inter_op_num_threads = 2

        available_providers = ort.get_available_providers()
        providers = []
        if self._use_cuda and "CUDAExecutionProvider" in available_providers:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": self._device_id,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": True,
                    },
                )
            )
        providers.append("CPUExecutionProvider")

        print(f"[HITNet] Loading model from: {self._model_path}")
        self._session = ort.InferenceSession(
            str(self._model_path), session_options, providers=providers
        )
        active_providers = self._session.get_providers()
        self._active_provider = active_providers[0] if active_providers else "Unknown"
        print(f"[HITNet] Active ONNX providers on GPU {self._device_id}: {active_providers}")

        inputs = self._session.get_inputs()
        self._num_inputs = len(inputs)
        self._output_name = self._session.get_outputs()[0].name
        first_shape = inputs[0].shape

        if self._num_inputs == 1:
            self._input_name = inputs[0].name
            if first_shape[1] == 6:
                self._nchw = True
                self._model_h = int(first_shape[2])
                self._model_w = int(first_shape[3])
            else:
                self._nchw = False
                self._model_h = int(first_shape[1])
                self._model_w = int(first_shape[2])
        else:
            self._input_left_name = inputs[0].name
            self._input_right_name = inputs[1].name
            if first_shape[1] == 3:
                self._nchw = True
                self._model_h = int(first_shape[2])
                self._model_w = int(first_shape[3])
            else:
                self._nchw = False
                self._model_h = int(first_shape[1])
                self._model_w = int(first_shape[2])

        print(f"[HITNet] Resolution: {self._model_w}x{self._model_h} (NCHW={self._nchw})")

        # Warm-up inference
        try:
            dummy_l = np.zeros((self._model_h, self._model_w, 3), dtype=np.uint8)
            dummy_r = np.zeros((self._model_h, self._model_w, 3), dtype=np.uint8)
            self.estimate_disparity(dummy_l, dummy_r)
            print("[HITNet] Model warm-up inference succeeded.")
        except Exception as e:
            print(f"[HITNet] Model warm-up notice: {e}")

    def estimate_disparity(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
    ) -> np.ndarray:
        if self._session is None:
            self.initialize()

        orig_h, orig_w = left_rectified.shape[:2]

        in_L = cv2.resize(left_rectified, (self._model_w, self._model_h), interpolation=cv2.INTER_LINEAR)
        in_R = cv2.resize(right_rectified, (self._model_w, self._model_h), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB float32 in range [0, 1]
        in_L = cv2.cvtColor(in_L, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        in_R = cv2.cvtColor(in_R, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self._num_inputs == 1:
            if self._nchw:
                in_L = np.transpose(in_L, (2, 0, 1))
                in_R = np.transpose(in_R, (2, 0, 1))
                combined = np.concatenate([in_L, in_R], axis=0)[np.newaxis, ...]
            else:
                combined = np.concatenate([in_L, in_R], axis=-1)[np.newaxis, ...]

            raw_disp = self._session.run([self._output_name], {self._input_name: combined})[0]
        else:
            if self._nchw:
                in_L = np.transpose(in_L, (2, 0, 1))[np.newaxis, ...]
                in_R = np.transpose(in_R, (2, 0, 1))[np.newaxis, ...]
            else:
                in_L = in_L[np.newaxis, ...]
                in_R = in_R[np.newaxis, ...]

            raw_disp = self._session.run(
                [self._output_name],
                {self._input_left_name: in_L, self._input_right_name: in_R},
            )[0]

        disp = np.squeeze(raw_disp).astype(np.float32)

        # Scale disparity if original image differs in width from model input
        if orig_w != self._model_w or orig_h != self._model_h:
            scale = float(orig_w) / float(self._model_w)
            disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) * scale

        # Invalidate non-positive disparities with NaN for downstream metric pipeline
        disp[disp <= 0] = np.nan
        return disp

