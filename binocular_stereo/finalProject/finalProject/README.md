# Stereo YOLO Measurement

This project combines two calibrated cameras, neural stereo depth, and YOLO
instance segmentation. The live application displays the YOLO result and adds
an estimated optical-axis distance (`Z`) plus the visible 3D width (`W`) and
height (`H`) for every detected object.

The current target platform is macOS with Apple Silicon. The supplied setup
uses two USB cameras, a 59.791 mm baseline, and 1920 x 1080 capture.

## What is included

```text
camera_pair_check.py          Find and label macOS camera indices
stereo_prototype.py           Run CREStereo on a saved stereo pair
stereo_webcam.py              Stereo depth preview without YOLO
stereo_yolo_webcam.py         Main integrated application
stereo_calibration_fixed.yml  Camera intrinsics and stereo extrinsics
models/crestereo_embedded.onnx  CREStereo ONNX model
yolo26n-seg.pt                YOLO instance-segmentation weights
data/middlebury_cones/        Optional offline stereo sanity-check pair
requirements.txt              Python dependencies
```

Runtime outputs are written to `runs/`, which is created automatically and is
not part of the handoff package.

## Setup

Use Python 3.10 or newer. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS, allow the terminal or IDE that runs Python to use the camera in
**System Settings > Privacy & Security > Camera**. If permission was just
granted, restart that terminal or IDE.

## Camera order and calibration

On the current Mac, the calibrated pair is:

```text
left camera  = index 0
right camera = index 1
```

Camera indices are assigned by macOS and can change after cameras are added or
removed. Check them before running on another computer:

```bash
python camera_pair_check.py --indices 0 1 2 3
```

Open the generated images under `runs/camera_pair_check/`. The selected two
cameras must see the same scene with substantial overlap. They must remain in
the same rigid left/right mount and at the same resolution used for
calibration. Passing two unrelated or differently positioned cameras produces
plausible-looking but incorrect depth.

The supplied `stereo_calibration_fixed.yml` is the calibration file to use.
Its translation vector is in millimetres; the scripts convert the baseline to
metres for depth. Do not replace it with the raw calibration file unless it has
been repaired and validated. Any change in camera position, focus, resolution,
or left/right order requires recalibration.

## Run the main application

From `finalProject/` with the virtual environment active:

```bash
python stereo_yolo_webcam.py \
  --left-camera 0 \
  --right-camera 1 \
  --use-coreml
```

`--use-coreml` asks ONNX Runtime to use Apple's CoreML provider and falls back
to the CPU provider if CoreML cannot compile the model. To use the reference
CPU path explicitly, omit that option.

The window shows:

- the rectified left image as the main view;
- YOLO class, confidence, and instance mask;
- `Z`: median optical-axis depth over the mask;
- `W` and `H`: robust 3D extents of the visible masked points.

Press `s` to save the current annotated image, rectified pair, disparity,
depth, and a text measurement record under `runs/stereo_yolo_webcam/`. Press
`q` or `Esc` to quit.

For a stereo-only preview, run:

```bash
python stereo_webcam.py \
  --left-camera 0 \
  --right-camera 1 \
  --use-coreml
```

The CREStereo checkpoint is 320 x 240 with a 4:3 aspect ratio. By default the
scripts center-crop the rectified 1920 x 1080 frames to 4:3 before inference.
`--no-crop-to-4-3` is available for a quick visual test, but it stretches the
16:9 image and is not preferred for measurements.

## Offline checks

To test the stereo model without cameras, use the included Middlebury pair:

```bash
python stereo_prototype.py \
  --model models/crestereo_embedded.onnx \
  --left data/middlebury_cones/cones/im2.png \
  --right data/middlebury_cones/cones/im6.png \
  --baseline 0.06 \
  --focal-length 700 \
  --output-dir runs/middlebury
```

The baseline and focal length in this command are only placeholders for the
offline model test. They are not the physical parameters of the USB cameras.

## How the measurement works

The cameras are rectified first. CREStereo predicts disparity `d`, and the
calibration values convert it to metric depth:

```text
Z = fx * baseline / d
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

YOLO segmentation supplies the object mask. The program samples valid 3D
points inside that mask, removes boundary and outlier pixels, and reports the
median `Z` and robust visible `X`/`Y` extents. It measures the visible part of
an object; it cannot infer faces hidden from both cameras.

The supplied YOLO weights support their pretrained categories (COCO). Objects
outside those categories will not be detected until a custom segmentation
model is trained.

## Known limitations

- Independent USB cameras are not hardware-synchronized. Use stationary
  scenes first; moving objects can have mismatched left/right timestamps.
- The current CREStereo CPU path takes roughly 2–3 seconds per frame on the
  development Mac. CoreML may be faster, depending on the installed runtime.
- Textureless, reflective, transparent, or occluded surfaces can produce bad
  stereo depth even when the image looks reasonable.
- Distance and size accuracy must be checked against physical measurements at
  the intended working range. A neural model's smooth output is not an
  accuracy guarantee.
