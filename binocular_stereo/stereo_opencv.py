import os
import cv2
import numpy as np

# ==============================================================================
# CONFIGURATION & PERFORMANCE SETTINGS
# ==============================================================================
CAM_LEFT_ID = 1
CAM_RIGHT_ID = 2

# Set Downsample Factor:
# 1 = Full 1080p (Slow)
# 2 = 960x540    (Fast, Recommended for Laptops ~25-30 FPS)
# 3 = 640x360    (Ultra-fast)
DOWNSAMPLE = 2

NATIVE_WIDTH = 1920
NATIVE_HEIGHT = 1080

# ==============================================================================
# 1. SETUP CAMERAS
# ==============================================================================
print(f"Connecting to cameras {CAM_LEFT_ID} and {CAM_RIGHT_ID} via DirectShow...")
camL = cv2.VideoCapture(CAM_LEFT_ID, cv2.CAP_DSHOW)
camR = cv2.VideoCapture(CAM_RIGHT_ID, cv2.CAP_DSHOW)

for cam in (camL, camR):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_HEIGHT)

if not camL.isOpened() or not camR.isOpened():
    print("Error: Could not open one or both cameras.")
    exit(1)

ret, test_frame = camL.read()
if not ret or test_frame is None:
    print("Error: Failed to grab frame.")
    exit(1)

actual_h, actual_w = test_frame.shape[:2]
target_w = actual_w // DOWNSAMPLE
target_h = actual_h // DOWNSAMPLE
scaled_size = (target_w, target_h)
print(f"Running at: {target_w}x{target_h} (Downsample factor: {DOWNSAMPLE})")

# ==============================================================================
# 2. LOAD & SCALE CALIBRATION
# ==============================================================================
calib_path = "./Calibration/stereo_calibration_fixed.yml"
if not os.path.exists(calib_path):
    calib_path = "stereo_calibration.yml"

fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise IOError(f"Could not open {calib_path}")

K1 = fs.getNode("K1").mat()
D1 = fs.getNode("D1").mat()
K2 = fs.getNode("K2").mat()
D2 = fs.getNode("D2").mat()
R  = fs.getNode("R").mat()
T  = fs.getNode("T").mat()
fs.release()

# Scale intrinsics to match downsampled resolution
K1_scaled = K1.copy()
K2_scaled = K2.copy()
K1_scaled[0, 0] /= DOWNSAMPLE
K1_scaled[1, 1] /= DOWNSAMPLE
K1_scaled[0, 2] /= DOWNSAMPLE
K1_scaled[1, 2] /= DOWNSAMPLE

K2_scaled[0, 0] /= DOWNSAMPLE
K2_scaled[1, 1] /= DOWNSAMPLE
K2_scaled[0, 2] /= DOWNSAMPLE
K2_scaled[1, 2] /= DOWNSAMPLE

# Precompute Rectification Maps at scaled size
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1_scaled, D1, K2_scaled, D2, scaled_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=scaled_size
)

mapL_x, mapL_y = cv2.initUndistortRectifyMap(K1_scaled, D1, R1, P1, scaled_size, cv2.CV_32FC1)
mapR_x, mapR_y = cv2.initUndistortRectifyMap(K2_scaled, D2, R2, P2, scaled_size, cv2.CV_32FC1)

# ==============================================================================
# 3. INTERACTIVE CONTROLS & TRACKBARS
# ==============================================================================
cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuner", 450, 300)

def nothing(x): pass

default_num_disp = max(4, 12 // DOWNSAMPLE)
cv2.createTrackbar("Num Disp (*16)", "Tuner", default_num_disp, 16, nothing)
cv2.createTrackbar("Block Size", "Tuner", 1, 8, nothing)         # (val * 2) + 3 -> 3, 5, 7...
cv2.createTrackbar("Uniqueness", "Tuner", 10, 30, nothing)
cv2.createTrackbar("WLS Lambda (/100)", "Tuner", 80, 150, nothing)  # 80 * 100 = 8000
cv2.createTrackbar("WLS Sigma (*0.1)", "Tuner", 15, 50, nothing)    # 15 * 0.1 = 1.5

cv2.namedWindow("Rectified Pair", cv2.WINDOW_NORMAL)
cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)

# State tracker for matcher updates
prev_params = None
left_matcher = None
right_matcher = None
wls_filter = None
use_fast_mode = True

print("\nControls:")
print("  [q] : Quit")
print("  [m] : Toggle Fast 3-Way Mode vs High-Quality HH Mode")
print("  [s] : Swap Left & Right camera feeds\n")

# ==============================================================================
# 4. MAIN PROCESSING LOOP
# ==============================================================================
while True:
    # 1. Grab frames in sync
    camL.grab()
    camR.grab()
    _, frameL = camL.retrieve()
    _, frameR = camR.retrieve()

    if frameL is None or frameR is None:
        continue

    # 2. Downsample
    if DOWNSAMPLE > 1:
        frameL_s = cv2.resize(frameL, scaled_size, interpolation=cv2.INTER_AREA)
        frameR_s = cv2.resize(frameR, scaled_size, interpolation=cv2.INTER_AREA)
    else:
        frameL_s, frameR_s = frameL, frameR

    # 3. Rectify
    rect_L = cv2.remap(frameL_s, mapL_x, mapL_y, cv2.INTER_LINEAR)
    rect_R = cv2.remap(frameR_s, mapR_x, mapR_y, cv2.INTER_LINEAR)

    gray_L = cv2.cvtColor(rect_L, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(rect_R, cv2.COLOR_BGR2GRAY)

    # 4. Update Matchers & WLS Filter when trackbars move
    num_disp = max(1, cv2.getTrackbarPos("Num Disp (*16)", "Tuner")) * 16
    b_size = cv2.getTrackbarPos("Block Size", "Tuner") * 2 + 3
    unique = cv2.getTrackbarPos("Uniqueness", "Tuner")
    wls_lambda = cv2.getTrackbarPos("WLS Lambda (/100)", "Tuner") * 100
    wls_sigma = cv2.getTrackbarPos("WLS Sigma (*0.1)", "Tuner") * 0.1

    curr_params = (num_disp, b_size, unique, wls_lambda, wls_sigma, use_fast_mode)
    if curr_params != prev_params:
        mode_flag = cv2.STEREO_SGBM_MODE_SGBM_3WAY if use_fast_mode else cv2.STEREO_SGBM_MODE_HH
        
        # Left Matcher
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=b_size,
            P1=8 * 1 * (b_size ** 2),
            P2=32 * 1 * (b_size ** 2),
            disp12MaxDiff=1,
            uniquenessRatio=unique,
            speckleWindowSize=0,  # WLS handles speckle removal directly
            speckleRange=0,
            mode=mode_flag
        )
        
        # Right Matcher (derived automatically from left matcher)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        # WLS Disparity Filter
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(wls_lambda)
        wls_filter.setSigmaColor(wls_sigma)
        
        prev_params = curr_params

    # 5. Compute Left & Right Disparities, then Filter
    disp_L = left_matcher.compute(gray_L, gray_R)
    disp_R = right_matcher.compute(gray_R, gray_L)
    
    # rect_L acts as the edge guidance image to align depth edges to color boundaries
    filtered_disp = wls_filter.filter(
        disparity_map_left=disp_L,
        left_view=rect_L,
        disparity_map_right=disp_R
    )

    # 6. Normalize and Colorize for Display
    disp_float = filtered_disp.astype(np.float32) / 16.0
    valid_mask = disp_float > 0

    disp_vis = np.zeros_like(disp_float, dtype=np.uint8)
    if np.any(valid_mask):
        disp_vis[valid_mask] = cv2.normalize(
            disp_float[valid_mask], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    color_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    color_disp[~valid_mask] = [0, 0, 0]  # Mask out occlusion and invalid points

    # 7. Build Side-by-Side Rectified Pair View with Epipolar Lines
    rect_pair = np.hstack((rect_L, rect_R))
    line_step = 40
    for y in range(line_step, target_h, line_step):
        cv2.line(rect_pair, (0, y), (target_w * 2, y), (0, 255, 0), 1)

    mode_text = "FAST (3-Way)" if use_fast_mode else "HQ (HH)"
    cv2.putText(rect_pair, f"Mode: {mode_text} (Press 'm' to toggle)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Rectified Pair", rect_pair)
    cv2.imshow("Disparity", color_disp)

    # 8. Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        use_fast_mode = not use_fast_mode
        print(f"Switched mode to: {'FAST (3-Way)' if use_fast_mode else 'High Quality (HH)'}")
    elif key == ord('s'):
        camL, camR = camR, camL
        print("Swapped left and right cameras.")

camL.release()
camR.release()
cv2.destroyAllWindows()