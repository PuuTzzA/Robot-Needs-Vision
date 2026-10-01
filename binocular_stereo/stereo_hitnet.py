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

NATIVE_WIDTH = 1920
NATIVE_HEIGHT = 1080

# Specify the local model file you placed in this folder:
MODEL_PATH = "middlebury_d400/saved_model_480x640/model_float32.onnx"
# If you downloaded Middlebury 640x480 instead, change to:
# MODEL_PATH = "hitnet_middlebury_480x640.onnx"

CALIB_PATH = "./Calibration/stereo_calibration_fixed.yml"
if not os.path.exists(CALIB_PATH):
    CALIB_PATH = "stereo_calibration.yml"


# ==============================================================================
# 1. VERIFY MODEL EXISTS
# ==============================================================================
if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] Model file '{MODEL_PATH}' not found in current directory.")
    print("Please download 'hitnet_flyingthings3d_540x960.onnx' and place it here:")
    print(f"Directory: {os.path.abspath(os.getcwd())}\n")
    sys.exit(1)


# ==============================================================================
# 2. HITNET ONNX RUNTIME WRAPPER
# ==============================================================================
class HitnetInference:
    def __init__(self, model_path):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if "CUDAExecutionProvider" in ort.get_available_providers() \
            else ["CPUExecutionProvider"]

        print(f"Loading ONNX Model: {model_path}")
        print(f"Active Provider: {providers[0]}")
        self.session = ort.InferenceSession(model_path, providers=providers)

        self.input_left_name = self.session.get_inputs()[0].name
        self.input_right_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

        shape = self.session.get_inputs()[0].shape
        # Handle dynamic or static batch/channels
        if shape[1] == 3:
            self.nchw = True
            self.model_h = int(shape[2])
            self.model_w = int(shape[3])
        else:
            self.nchw = False
            self.model_h = int(shape[1])
            self.model_w = int(shape[2])

        print(f"Model resolution configured to: {self.model_w}x{self.model_h}")

    def compute(self, left_bgr, right_bgr):
        # Resize inputs to the model's required tensor dimensions
        in_L = cv2.resize(left_bgr, (self.model_w, self.model_h), interpolation=cv2.INTER_AREA)
        in_R = cv2.resize(right_bgr, (self.model_w, self.model_h), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB float32 in [0, 1]
        in_L = cv2.cvtColor(in_L, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        in_R = cv2.cvtColor(in_R, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.nchw:
            in_L = np.transpose(in_L, (2, 0, 1))[np.newaxis, ...]
            in_R = np.transpose(in_R, (2, 0, 1))[np.newaxis, ...]
        else:
            in_L = in_L[np.newaxis, ...]
            in_R = in_R[np.newaxis, ...]

        # Run inference
        raw_disp = self.session.run(
            [self.output_name],
            {self.input_left_name: in_L, self.input_right_name: in_R}
        )[0]

        disp = np.squeeze(raw_disp)

        # Scale disparity if original image differs in width from model input
        orig_w = left_bgr.shape[1]
        orig_h = left_bgr.shape[0]
        if orig_w != self.model_w:
            scale = float(orig_w) / float(self.model_w)
            disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) * scale

        return disp


# ==============================================================================
# 3. CAMERA SETUP
# ==============================================================================
print(f"Opening cameras {CAM_LEFT_ID} and {CAM_RIGHT_ID}...")
camL = cv2.VideoCapture(CAM_LEFT_ID)
camR = cv2.VideoCapture(CAM_RIGHT_ID)

if not camL.isOpened() or not camR.isOpened():
    camL = cv2.VideoCapture(CAM_LEFT_ID, cv2.CAP_DSHOW)
    camR = cv2.VideoCapture(CAM_RIGHT_ID, cv2.CAP_DSHOW)

for cam in (camL, camR):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_HEIGHT)

if not camL.isOpened() or not camR.isOpened():
    print("Error: Could not open one or both cameras.")
    sys.exit(1)

ret, test_frame = camL.read()
if not ret or test_frame is None:
    print("Error: Failed to grab initial frame.")
    sys.exit(1)

actual_h, actual_w = test_frame.shape[:2]
target_size = (actual_w, actual_h)
print(f"Camera frame size: {actual_w}x{actual_h}")


# ==============================================================================
# 4. LOAD CALIBRATION & RECTIFICATION MAPS
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

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2, target_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=target_size
)

mapL_x, mapL_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, target_size, cv2.CV_32FC1)
mapR_x, mapR_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, target_size, cv2.CV_32FC1)

hitnet = HitnetInference(MODEL_PATH)

cv2.namedWindow("Rectified Pair", cv2.WINDOW_NORMAL)
cv2.namedWindow("HITNet Disparity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rectified Pair", 960, 270)
cv2.resizeWindow("HITNet Disparity", 960, 540)

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

    # HITNet subpixel disparity computation
    disp = hitnet.compute(rect_L, rect_R)

    # Colorize valid disparity
    valid_mask = disp > 0
    disp_vis = np.zeros_like(disp, dtype=np.uint8)
    if np.any(valid_mask):
        disp_vis[valid_mask] = cv2.normalize(
            disp[valid_mask], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    color_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    color_disp[~valid_mask] = [0, 0, 0]

    # Side-by-side epipolar check lines
    preview_L = cv2.resize(rect_L, (480, 270))
    preview_R = cv2.resize(rect_R, (480, 270))
    rect_pair = np.hstack((preview_L, preview_R))
    for y in range(30, 270, 30):
        cv2.line(rect_pair, (0, y), (960, y), (0, 255, 0), 1)

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