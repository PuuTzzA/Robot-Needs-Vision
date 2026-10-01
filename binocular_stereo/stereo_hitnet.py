import os
import sys
import cv2
import numpy as np
import onnxruntime as ort

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CAM_LEFT_ID = 1
CAM_RIGHT_ID = 2

# Model downloaded from https://www.google.com/url?sa=E&q=https%3A%2F%2Fs3.ap-northeast-2.wasabisys.com%2Fpinto-model-zoo%2F142_HITNET%2Fresources.tar.gz
# and saved in ./Models
# Model         | Environment             | Typical Max Disparity   | Recommendation
# Middlebury    | Indoor / Desktop / Lab  | High (~400 px)          | Best for indoor webcam setups
# ETH3D         | Outdoor / Natural Light | Low-Medium (~64 px)     | Best for long-range / outdoor robotics
# FlyingThings3D| General / Synthetic     | Medium-High             | Baseline pre-trained model
MODEL_PATH = "Models/middlebury_d400/saved_model_480x640/model_float32.onnx" 

CALIB_PATH = "./Calibration/stereo_calibration_fixed.yml"
if not os.path.exists(CALIB_PATH):
    CALIB_PATH = "stereo_calibration.yml"

# ==============================================================================
# 1. VERIFY MODEL
# ==============================================================================
if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] Model file '{MODEL_PATH}' not found.")
    sys.exit(1)

# ==============================================================================
# 2. HITNET ONNX WRAPPER (CPU OPTIMIZED)
# ==============================================================================
class HitnetInference:
    def __init__(self, model_path):
        # Force CPU execution for laptops without discrete NVIDIA GPUs
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() - 1)  # Use available CPU cores
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"Loading ONNX Model: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        inputs = self.session.get_inputs()
        self.num_inputs = len(inputs)
        self.output_name = self.session.get_outputs()[0].name

        first_shape = inputs[0].shape
        print(f"Detected {self.num_inputs} model input(s) with shape: {first_shape}")

        if self.num_inputs == 1:
            # Combined tensor: [1, H, W, 6] or [1, 6, H, W]
            self.input_name = inputs[0].name
            if first_shape[1] == 6:
                self.nchw = True
                self.model_h = int(first_shape[2])
                self.model_w = int(first_shape[3])
            else:
                self.nchw = False
                self.model_h = int(first_shape[1])
                self.model_w = int(first_shape[2])
        else:
            # Dual tensor: Left [1, H, W, 3] and Right [1, H, W, 3]
            self.input_left_name = inputs[0].name
            self.input_right_name = inputs[1].name
            if first_shape[1] == 3:
                self.nchw = True
                self.model_h = int(first_shape[2])
                self.model_w = int(first_shape[3])
            else:
                self.nchw = False
                self.model_h = int(first_shape[1])
                self.model_w = int(first_shape[2])

        print(f"Model resolution: {self.model_w}x{self.model_h} (NCHW={self.nchw})")

    def compute(self, left_bgr, right_bgr):
        # Resize inputs to the model's required tensor dimensions
        in_L = cv2.resize(left_bgr, (self.model_w, self.model_h), interpolation=cv2.INTER_AREA)
        in_R = cv2.resize(right_bgr, (self.model_w, self.model_h), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB float32 in range [0, 1]
        in_L = cv2.cvtColor(in_L, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        in_R = cv2.cvtColor(in_R, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.num_inputs == 1:
            if self.nchw:
                # Transpose to [C, H, W] and concatenate along channels -> [6, H, W]
                in_L = np.transpose(in_L, (2, 0, 1))
                in_R = np.transpose(in_R, (2, 0, 1))
                combined = np.concatenate([in_L, in_R], axis=0)[np.newaxis, ...]
            else:
                # Concatenate along channels -> [1, H, W, 6]
                combined = np.concatenate([in_L, in_R], axis=-1)[np.newaxis, ...]

            raw_disp = self.session.run([self.output_name], {self.input_name: combined})[0]
        else:
            if self.nchw:
                in_L = np.transpose(in_L, (2, 0, 1))[np.newaxis, ...]
                in_R = np.transpose(in_R, (2, 0, 1))[np.newaxis, ...]
            else:
                in_L = in_L[np.newaxis, ...]
                in_R = in_R[np.newaxis, ...]

            raw_disp = self.session.run(
                [self.output_name],
                {self.input_left_name: in_L, self.input_right_name: in_R}
            )[0]

        disp = np.squeeze(raw_disp)

        # Scale disparity if original image differs in resolution
        orig_w = left_bgr.shape[1]
        orig_h = left_bgr.shape[0]
        if orig_w != self.model_w:
            scale = float(orig_w) / float(self.model_w)
            disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) * scale

        return disp

# ==============================================================================
# 3. CAMERA SETUP (DIRECTSHOW)
# ==============================================================================
print(f"Opening cameras {CAM_LEFT_ID} and {CAM_RIGHT_ID} via DirectShow...")
camL = cv2.VideoCapture(CAM_LEFT_ID, cv2.CAP_DSHOW)
camR = cv2.VideoCapture(CAM_RIGHT_ID, cv2.CAP_DSHOW)

# Request 640x480 native to match model resolution directly (saves CPU usage)
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
# 4. LOAD CALIBRATION & RECTIFICATION
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

# Scale intrinsics if calibration resolution differs from target camera resolution
calib_w = 1920  # original resolution from your calibration step
calib_h = 1080
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

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1_scaled, D1, K2_scaled, D2, target_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=target_size
)

mapL_x, mapL_y = cv2.initUndistortRectifyMap(K1_scaled, D1, R1, P1, target_size, cv2.CV_32FC1)
mapR_x, mapR_y = cv2.initUndistortRectifyMap(K2_scaled, D2, R2, P2, target_size, cv2.CV_32FC1)

hitnet = HitnetInference(MODEL_PATH)

cv2.namedWindow("Rectified Pair", cv2.WINDOW_NORMAL)
cv2.namedWindow("HITNet Disparity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rectified Pair", 640, 240)
cv2.resizeWindow("HITNet Disparity", 640, 480)

print("\nControls:")
print("  [q] : Quit")
print("  [s] : Swap Left & Right camera feeds\n")

# ==============================================================================
# 5. MAIN PROCESSING LOOP
# ==============================================================================
while True:
    camL.grab()
    camR.grab()
    _, frameL = camL.retrieve()
    _, frameR = camR.retrieve()

    if frameL is None or frameR is None:
        continue

    # Epipolar rectification
    rect_L = cv2.remap(frameL, mapL_x, mapL_y, cv2.INTER_LINEAR)
    rect_R = cv2.remap(frameR, mapR_x, mapR_y, cv2.INTER_LINEAR)

    # Compute subpixel disparity
    disp = hitnet.compute(rect_L, rect_R)

    # Colorize disparity
    valid_mask = disp > 0
    disp_vis = np.zeros_like(disp, dtype=np.uint8)
    if np.any(valid_mask):
        disp_vis[valid_mask] = cv2.normalize(
            disp[valid_mask], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    color_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    color_disp[~valid_mask] = [0, 0, 0]

    # Show horizontal epipolar alignment lines
    preview_L = cv2.resize(rect_L, (320, 240))
    preview_R = cv2.resize(rect_R, (320, 240))
    rect_pair = np.hstack((preview_L, preview_R))
    for y in range(30, 240, 30):
        cv2.line(rect_pair, (0, y), (640, y), (0, 255, 0), 1)

    cv2.imshow("Rectified Pair", rect_pair)
    cv2.imshow("HITNet Disparity", color_disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        camL, camR = camR, camL
        print("Swapped Left and Right cameras.")

camL.release()
camR.release()
cv2.destroyAllWindows()