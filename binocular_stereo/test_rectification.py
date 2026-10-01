import os
import cv2
import numpy as np

CAM_LEFT_ID = 1
CAM_RIGHT_ID = 2
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

camL = cv2.VideoCapture(CAM_LEFT_ID, cv2.CAP_DSHOW)
camR = cv2.VideoCapture(CAM_RIGHT_ID, cv2.CAP_DSHOW)

for cam in (camL, camR):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

if not camL.isOpened() or not camR.isOpened():
    print("Error: Could not open cameras.")
    exit(1)

ret, test_frame = camL.read()
actual_h, actual_w = test_frame.shape[:2]
image_size = (actual_w, actual_h)

# Load YAML
calib_path = "./Calibration/stereo_calibration_fixed.yml"
if not os.path.exists(calib_path):
    calib_path = "stereo_calibration.yml"

fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
K1_orig = fs.getNode("K1").mat()
D1_orig = fs.getNode("D1").mat()
K2_orig = fs.getNode("K2").mat()
D2_orig = fs.getNode("D2").mat()
R_orig  = fs.getNode("R").mat()
T_orig  = fs.getNode("T").mat()
fs.release()

current_mode = 1
mapL_x = mapL_y = mapR_x = mapR_y = None

def update_rectification(mode):
    global mapL_x, mapL_y, mapR_x, mapR_y, current_mode
    current_mode = mode
    
    if mode == 1:
        # Default as loaded
        k1, d1, k2, d2 = K1_orig, D1_orig, K2_orig, D2_orig
        r, t = R_orig, T_orig
        desc = "Mode 1: Default (As in YAML)"
    elif mode == 2:
        # Transpose R (Inverted rotation)
        k1, d1, k2, d2 = K1_orig, D1_orig, K2_orig, D2_orig
        r = R_orig.T
        t = -np.dot(R_orig.T, T_orig)
        desc = "Mode 2: Inverted Rotation (R^T)"
    elif mode == 3:
        # Swap Camera 1 and Camera 2
        k1, d1, k2, d2 = K2_orig, D2_orig, K1_orig, D1_orig
        r = R_orig.T
        t = -np.dot(R_orig.T, T_orig)
        desc = "Mode 3: Swapped Cameras (Cam2 <-> Cam1)"
    elif mode == 4:
        # Swap Camera 1 and Camera 2 with direct R
        k1, d1, k2, d2 = K2_orig, D2_orig, K1_orig, D1_orig
        r, t = R_orig, -T_orig
        desc = "Mode 4: Swapped Cameras + Direct R"

    print(f"\n--> Switched to: {desc}")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, image_size, r, t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=image_size
    )

    mapL_x, mapL_y = cv2.initUndistortRectifyMap(k1, d1, R1, P1, image_size, cv2.CV_32FC1)
    mapR_x, mapR_y = cv2.initUndistortRectifyMap(k2, d2, R2, P2, image_size, cv2.CV_32FC1)

update_rectification(1)

print("\n--- INSTRUCTIONS ---")
print("Press '1', '2', '3', or '4' to test alignment modes.")
print("Watch the circular ceiling light: Find the mode where it touches the EXACT SAME green line on both sides.")
print("Press 'q' to quit.\n")

cv2.namedWindow("Rectified Pair", cv2.WINDOW_NORMAL)

while True:
    camL.grab()
    camR.grab()
    _, frameL = camL.retrieve()
    _, frameR = camR.retrieve()

    if frameL is None or frameR is None:
        continue

    rectified_L = cv2.remap(frameL, mapL_x, mapL_y, interpolation=cv2.INTER_LINEAR)
    rectified_R = cv2.remap(frameR, mapR_x, mapR_y, interpolation=cv2.INTER_LINEAR)

    vis_pair = np.hstack((cv2.resize(rectified_L, (960, 540)), 
                          cv2.resize(rectified_R, (960, 540))))

    # Draw epipolar guidelines
    for y_line in range(50, 540, 40):
        cv2.line(vis_pair, (0, y_line), (1920, y_line), (0, 255, 0), 1)

    cv2.putText(vis_pair, f"Mode: {current_mode} (Press 1, 2, 3, or 4)", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Rectified Pair", vis_pair)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        update_rectification(int(chr(key)))

camL.release()
camR.release()
cv2.destroyAllWindows()