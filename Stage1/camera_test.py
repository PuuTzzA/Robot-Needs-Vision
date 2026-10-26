import cv2

# Set the indices you want to test here (e.g. [0], [1], or [0, 1])
CAMERA_INDICES = [1, 2]

# Try to open each camera
print("Starting camera test")
cameras = {}
for idx in CAMERA_INDICES:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        cameras[idx] = cap
        print(f"[OK] Camera {idx} opened successfully.")
    else:
        print(f"[FAIL] Could not open camera {idx}.")
        cap.release()

if not cameras:
    print("No cameras available. Exiting.")
    exit(1)

print("\nShowing live feed. Press 'q' on any window to quit.")

while True:
    for idx, cap in list(cameras.items()):
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imshow(f"Camera {idx}", frame)
        else:
            print(f"[WARNING] Could not read frame from camera {idx}.")

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
for cap in cameras.values():
    cap.release()
cv2.destroyAllWindows()