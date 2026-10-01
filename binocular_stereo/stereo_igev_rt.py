import os
import sys
import argparse
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add the IGEV-plusplus real-time core modules to the system path
# Download sceneflow.pth from: https://drive.google.com/drive/folders/1KlCs3rzqXlGrQzRBbE8yxtj0b5GaX-wI
# place it in ./Models/igev_rt/
SCRIPT_DIR = Path(__file__).resolve().parent

possible_roots = [
    SCRIPT_DIR / "Models" / "IGEV-plusplus",
    SCRIPT_DIR / "IGEV-plusplus",
    SCRIPT_DIR.parent / "Models" / "IGEV-plusplus",
    Path("Models/IGEV-plusplus").resolve(),
    Path("IGEV-plusplus").resolve(),
]

repo_root = None
for p in possible_roots:
    if (p / "core_rt" / "rt_igev_stereo.py").exists():
        repo_root = p
        break

if repo_root is None:
    # Fallback: search recursively for rt_igev_stereo.py
    found = list(SCRIPT_DIR.glob("**/rt_igev_stereo.py"))
    if found:
        repo_root = found[0].parent.parent
    else:
        print("[ERROR] Could not find 'rt_igev_stereo.py'. Ensure IGEV-plusplus is in ./Models/")
        sys.exit(1)

# Add repository root and core_rt to sys.path so internal imports resolve
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "core_rt"))

try:
    from rt_igev_stereo import IGEVStereo
    print(f"[INFO] Successfully loaded RT-IGEV from: {repo_root / 'core_rt'}")
except ImportError:
    try:
        from core_rt.rt_igev_stereo import IGEVStereo
        print(f"[INFO] Successfully loaded core_rt.rt_igev_stereo from: {repo_root}")
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("Ensure 'timm' is installed: pip install timm==0.5.4")
        sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CAM_LEFT_ID = 1
CAM_RIGHT_ID = 2

# Search possible locations for the pretrained weights
WEIGHT_CANDIDATES = [
    SCRIPT_DIR / "Models" / "igev_rt" / "sceneflow.pth",
    SCRIPT_DIR / "pretrained_models" / "igev_rt" / "sceneflow.pth",
    Path("./Models/igev_rt/sceneflow.pth").resolve(),
]

MODEL_CKPT = None
for cand in WEIGHT_CANDIDATES:
    if cand.exists():
        MODEL_CKPT = str(cand)
        break

if MODEL_CKPT is None:
    MODEL_CKPT = str(SCRIPT_DIR / "Models" / "igev_rt" / "sceneflow.pth")

CALIB_PATH = str(SCRIPT_DIR / "Calibration" / "stereo_calibration_fixed.yml")
if not os.path.exists(CALIB_PATH):
    CALIB_PATH = "stereo_calibration.yml"

# Processing resolution (lowering resolution significantly improves CPU FPS)
PROC_WIDTH = 480
PROC_HEIGHT = 320

# Number of GRU refinement iterations (2 is fastest for real-time CPU)
GRU_ITERS = 2

# ==============================================================================
# 1. RT-IGEV CPU INFERENCE WRAPPER
# ==============================================================================
class RTIgevInference:
    def __init__(self, ckpt_path, iters=2):
        self.device = torch.device("cpu")
        self.iters = iters

        # Leverage available CPU threads
        num_cores = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(num_cores)

        args = argparse.Namespace(
            restore_ckpt=ckpt_path,
            max_disp=192,
            hidden_dim=96,
            hidden_dims=[96, 96, 96],
            corr_levels=2,
            corr_radius=4,
            n_downsample=2,
            n_gru_layers=3,
            mixed_precision=False,
            precision_dtype="float32"
        )

        print(f"Instantiating RT-IGEV on CPU ({num_cores} threads)...")
        self.model = IGEVStereo(args)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")

        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Strip DataParallel 'module.' prefix if present
        state_dict = OrderedDict()
        raw_state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        for k, v in raw_state.items():
            name = k.replace("module.", "")
            state_dict[name] = v

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _pad(self, x, divis_by=32):
        """Pad image tensors to multiples of 32 (network constraint)."""
        h, w = x.shape[2], x.shape[3]
        pad_h = (divis_by - (h % divis_by)) % divis_by
        pad_w = (divis_by - (w % divis_by)) % divis_by
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)

    @torch.inference_mode()
    def compute(self, left_bgr, right_bgr):
        orig_h, orig_w = left_bgr.shape[:2]

        in_L = cv2.resize(left_bgr, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_AREA)
        in_R = cv2.resize(right_bgr, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_AREA)

        tensor_L = torch.from_numpy(cv2.cvtColor(in_L, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0)
        tensor_R = torch.from_numpy(cv2.cvtColor(in_R, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0)

        # Normalize to [-1.0, 1.0]
        tensor_L = 2.0 * (tensor_L / 255.0) - 1.0
        tensor_R = 2.0 * (tensor_R / 255.0) - 1.0

        tensor_L_pad, (h_crop, w_crop) = self._pad(tensor_L, divis_by=32)
        tensor_R_pad, _ = self._pad(tensor_R, divis_by=32)

        disp_pad = self.model(tensor_L_pad, tensor_R_pad, iters=self.iters, test_mode=True)

        if isinstance(disp_pad, (tuple, list)):
            disp_pad = disp_pad[-1]

        disp_pad = disp_pad.squeeze().cpu().numpy()
        disp = disp_pad[:h_crop, :w_crop]

        scale = float(orig_w) / float(PROC_WIDTH)
        disp_orig = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) * scale
        return disp_orig

# ==============================================================================
# 2. CAMERA INITIALIZATION (DIRECTSHOW)
# ==============================================================================
print(f"Opening cameras {CAM_LEFT_ID} and {CAM_RIGHT_ID} via DirectShow...")
camL = cv2.VideoCapture(CAM_LEFT_ID, cv2.CAP_DSHOW)
camR = cv2.VideoCapture(CAM_RIGHT_ID, cv2.CAP_DSHOW)

for cam in (camL, camR):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camL.isOpened() or not camR.isOpened():
    print("Error: Could not open one or both cameras via DirectShow.")
    sys.exit(1)

ret, test_frame = camL.read()
if not ret or test_frame is None:
    print("Error: Failed to grab initial frame.")
    sys.exit(1)

actual_h, actual_w = test_frame.shape[:2]
target_size = (actual_w, actual_h)
print(f"Camera frame size: {actual_w}x{actual_h}")

# ==============================================================================
# 3. RECTIFICATION CALIBRATION
# ==============================================================================
fs = cv2.FileStorage(CALIB_PATH, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise IOError(f"Could not open calibration file: {CALIB_PATH}")

K1 = fs.getNode("K1").mat()
D1 = fs.getNode("D1").mat()
K2 = fs.getNode("K2").mat()
D2 = fs.getNode("D2").mat()
R  = fs.getNode("R").mat()
T  = fs.getNode("T").mat()
fs.release()

calib_w, calib_h = 1920, 1080
scale_x = actual_w / float(calib_w)
scale_y = actual_h / float(calib_h)

K1_scaled = K1.copy()
K2_scaled = K2.copy()
K1_scaled[0, 0] *= scale_x
K1_scaled[1, 1] *= scale_y
K1_scaled[0, 2] *= scale_x
K1_scaled[1, 2] *= scale_y

K2_scaled[0, 0] *= scale_x
K2_scaled[1, 1] *= scale_y
K2_scaled[0, 2] *= scale_x
K2_scaled[1, 2] *= scale_y

R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
    K1_scaled, D1, K2_scaled, D2, target_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=target_size
)

mapL_x, mapL_y = cv2.initUndistortRectifyMap(K1_scaled, D1, R1, P1, target_size, cv2.CV_32FC1)
mapR_x, mapR_y = cv2.initUndistortRectifyMap(K2_scaled, D2, R2, P2, target_size, cv2.CV_32FC1)

rt_igev = RTIgevInference(MODEL_CKPT, iters=GRU_ITERS)

cv2.namedWindow("Rectified Pair", cv2.WINDOW_NORMAL)
cv2.namedWindow("RT-IGEV Disparity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rectified Pair", 640, 240)
cv2.resizeWindow("RT-IGEV Disparity", 640, 480)

print("\nControls:")
print("  [q] : Quit")
print("  [s] : Swap Left & Right camera feeds\n")

# ==============================================================================
# 4. CAPTURE & INFERENCE LOOP
# ==============================================================================
while True:
    camL.grab()
    camR.grab()
    _, frameL = camL.retrieve()
    _, frameR = camR.retrieve()

    if frameL is None or frameR is None:
        continue

    rect_L = cv2.remap(frameL, mapL_x, mapL_y, cv2.INTER_LINEAR)
    rect_R = cv2.remap(frameR, mapR_x, mapR_y, cv2.INTER_LINEAR)

    disp = rt_igev.compute(rect_L, rect_R)

    valid_mask = disp > 0
    disp_vis = np.zeros_like(disp, dtype=np.uint8)
    if np.any(valid_mask):
        disp_vis[valid_mask] = cv2.normalize(
            disp[valid_mask], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    color_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    color_disp[~valid_mask] = [0, 0, 0]

    preview_L = cv2.resize(rect_L, (320, 240))
    preview_R = cv2.resize(rect_R, (320, 240))
    rect_pair = np.hstack((preview_L, preview_R))
    for y in range(30, 240, 30):
        cv2.line(rect_pair, (0, y), (640, y), (0, 255, 0), 1)

    cv2.imshow("Rectified Pair", rect_pair)
    cv2.imshow("RT-IGEV Disparity", color_disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        camL, camR = camR, camL
        print("Swapped Left and Right cameras.")

camL.release()
camR.release()
cv2.destroyAllWindows()