# Robot Needs Vision

This repository contains a binocular stereo-vision project for robot perception. It combines calibrated stereo cameras, neural depth estimation, and object segmentation to estimate the distance and size of objects in a scene.

The main application is built around a stereo camera pair and a CREStereo depth model, with YOLO-based instance segmentation used to measure visible 3D dimensions of detected objects.

## Overview

The project is designed to:

- capture stereo imagery from two USB cameras
- calibrate and rectify the camera pair
- estimate depth from disparity using a neural stereo model
- detect objects using YOLO instance segmentation
- estimate each detected object's distance and visible 3D dimensions

## Repository structure

```text
Robot-Needs-Vision/
├── README.md
├── .gitignore
├── binocular_stereo/
│   ├── .gitignore
│   ├── camera_test.py
│   ├── stereo_hitnet.py
│   ├── stereo_igev_rt.py
│   ├── stereo_opencv.py
│   ├── test_rectification.py
│   ├── calibration/
│   │   ├── StereoParams2.mat
│   │   ├── stereo_calib_cv.mat
│   │   ├── stereo_calibration_fixed.yml
│   │   └── stereo_calibration_raw.yml
│   └── finalProject/
│       └── finalProject/
│           ├── README.md
│           ├── camera_pair_check.py
│           ├── stereo_prototype.py
│           ├── stereo_webcam.py
│           ├── stereo_yolo_webcam.py
│           ├── requirements.txt
│           ├── stereo_calibration_fixed.yml
│           ├── data/
│           │   └── middlebury_cones/
│           └── models/
│               └── crestereo_embedded.onnx
```

## Main application

The most complete and documented implementation lives in:

```text
binocular_stereo/finalProject/finalProject/
```

This project targets macOS on Apple Silicon with a dual USB-camera setup. The app can:

- identify camera indices
- perform stereo calibration and rectification
- estimate depth from disparity
- run YOLO segmentation on the left image
- calculate object depth and visible dimensions (`Z`, `W`, `H`)

## Quick start

From the project directory:

```bash
cd binocular_stereo/finalProject/finalProject
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the main application:

```bash
python stereo_yolo_webcam.py \
  --left-camera 0 \
  --right-camera 1 \
  --use-coreml
```

The `--use-coreml` flag enables Apple's CoreML execution provider when available. If CoreML is unavailable, the app falls back to CPU execution.

## Camera calibration and setup

The project includes a fixed calibration file:

```text
binocular_stereo/finalProject/finalProject/stereo_calibration_fixed.yml
```

The current known test setup uses:

```text
left camera  = index 0
right camera = index 1
```

Camera indices can vary by machine and by connected devices. To check the available camera IDs:

```bash
python camera_pair_check.py --indices 0 1 2 3
```

This helps ensure the selected cameras match the calibrated left/right pair. If the camera arrangement, resolution, focus, or left/right order changes, recalibration is required.

## Offline validation

The repository also includes a sample stereo-pair for testing without live cameras:

```bash
python stereo_prototype.py \
  --model models/crestereo_embedded.onnx \
  --left data/middlebury_cones/cones/im2.png \
  --right data/middlebury_cones/cones/im6.png \
  --baseline 0.06 \
  --focal-length 700 \
  --output-dir runs/middlebury
```

This can be used to validate the stereo pipeline and model output without hardware.

## How depth is computed

The system rectifies the stereo images and feeds them into CREStereo, which predicts disparity `d`. Depth is then derived using the camera geometry:

```text
Z = fx * baseline / d
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

Where:

- `fx`, `fy` are focal lengths
- `baseline` is the stereo camera separation
- `(u, v)` is the pixel location
- `(cx, cy)` is the principal point

YOLO segmentation provides the object mask, so the depth measurement is limited to the visible area of the detected object.

## Notes and limitations

- Independent USB cameras are not hardware-synchronized.
- Moving scenes can suffer from left/right timestamp mismatch.
- Textureless, reflective, transparent, or occluded surfaces can produce poor depth estimates.
- Real-world measurement accuracy should be validated against ground-truth measurements in the intended operating range.

## Related files

- `binocular_stereo/stereo_opencv.py` — OpenCV-based stereo work
- `binocular_stereo/stereo_igev_rt.py` — another real-time stereo implementation
- `binocular_stereo/stereo_hitnet.py` — HITNet-related stereo experimentation
- `binocular_stereo/test_rectification.py` — calibration and rectification checks
- `binocular_stereo/finalProject/finalProject/README.md` — deeper project-specific documentation

## License

No explicit license is declared in the repository metadata at the time of writing.

This repository is best understood as a practical stereo-vision prototype for robot perception, with the final integrated implementation focused on real-time depth and object measurement from a calibrated dual-camera setup.
