# Stage 1 Stereo Vision (40% of Final Grade)
## Complete Guide: Running on SSH Server & Local Laptop

This guide provides step-by-step instructions to run the binocular perception pipeline when your **cameras are plugged into your laptop's USB ports** and the **heavy computation runs on the remote NVIDIA RTX 4090 GPU server over SSH**.

---

## Architecture Overview

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                               YOUR LAPTOP                                │
 │                                                                          │
 │  ┌─────────────────────────┐                ┌─────────────────────────┐  │
 │  │ USB Stereo Webcams      │                │ Web Browser             │  │
 │  │ (Left Cam + Right Cam)  │                │ http://localhost:8080   │  │
 │  └───────────┬─────────────┘                └────────────▲────────────┘  │
 │              │                                           │               │
 │              ▼                                           │ (Web UI &     │
 │    laptop_camera_sender.py                               │  Depth Feed)  │
 │    (Synchronized grab + JPEG)                            │               │
 └──────────────┬───────────────────────────────────────────┴───────────────┘
                │ Port 5555 (Video Stream)                  │ Port 8080
════════════════╪═══════════════════════════════════════════╪═════════════════
                ▼                                           │
 ┌──────────────────────────────────────────────────────────┴───────────────┐
 │                       REMOTE GPU SERVER (SSH)                            │
 │                                                                          │
 │   harness.py                                                             │
 │   ├── 1. Receives video stream from laptop on port 5555                  │
 │   ├── 2. Undistorts & Rectifies epipolar scanlines (YAML calibration)     │
 │   ├── 3. Computes disparity (CREStereo ONNX / Traditional StereoSGBM)    │
 │   ├── 4. Calculates absolute metric depth Z = (fx * B) / d               │
 │   └── 5. Serves interactive Web Dashboard on port 8080                   │
 └──────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. On the Remote GPU Server (Already Installed & Ready)
The environment is already set up in `/data/thomas/Robot-Needs-Vision/.venv`.
To test or verify anytime:
```bash
cd /data/thomas/Robot-Needs-Vision
.venv/bin/python Stage1/test_stage1.py
```

### 2. On Your Local Laptop
You only need standard Python and OpenCV on your laptop to grab webcam frames:
```bash
pip install opencv-python numpy
```

---

## Method 1: Running with VS Code Remote Explorer (Easiest)

If you are using **VS Code Remote SSH**:

### Step 1: Forward the Ports in VS Code
1. In VS Code, look at the bottom panel and click on the **Ports** tab (next to Terminal / Output).
2. Ensure the following two ports are forwarded:
   - **Port `8080` (Local -> Remote)**: For the interactive web dashboard. (VS Code usually auto-forwards this).
   - **Port `5555` (Reverse: Remote -> Local)**: For the camera stream.
     - In the Ports tab, click **Add Port** -> enter `5555`.
     - Right-click the forwarded port `5555` -> select **Port Forwarding Type** -> **Reverse** (forward from remote server to your local machine).

*(Alternatively, if you prefer terminal SSH, see **Method 2** below).*

### Step 2: Start the Master Pipeline on the Remote Server
In your remote terminal inside VS Code:
```bash
cd /data/thomas/Robot-Needs-Vision
.venv/bin/python Stage1/harness.py --source network --web-port 8080 --stream-port 5555
```
You will see:
```text
======================================================================
  STAGE 1 STEREO VISION HARNESS
  Source: NETWORK | Initial Estimator: CRESTEREO
======================================================================
[Calibration] Loaded successfully. Baseline: 59.79 mm | Rectified fx: 3803.9 px
[Pipeline] Initializing active estimator: Deep CREStereo (CUDA/ONNX)...
[WebUI] Interactive dashboard running at http://0.0.0.0:8080
[NetworkReceiver] Listening for laptop camera stream on 0.0.0.0:5555...
```

### Step 3: Stream Your Cameras from Your Laptop
Open a terminal on your **local laptop** (not in SSH), navigate to the project directory, and run:
```bash
# If on macOS / Linux laptop:
python3 Stage1/laptop_camera_sender.py --server 127.0.0.1 --port 5555 --left-cam 0 --right-cam 1

# If on Windows laptop:
python Stage1/laptop_camera_sender.py --server 127.0.0.1 --port 5555 --left-cam 0 --right-cam 1
```

> [!TIP]
> **Which Camera Index?**
> - On Mac / Linux: usually `--left-cam 0 --right-cam 1` (or `1` and `2` if your laptop has a built-in FaceTime/webcam).
> - On Windows: usually `--left-cam 1 --right-cam 2` (device `0` is often the built-in webcam).
> - If the left and right feeds look swapped in the browser, simply swap the indices: `--left-cam 1 --right-cam 0`.

### Step 4: Open the Dashboard in Your Browser
Open your browser on your laptop:
👉 **`http://localhost:8080`**

---

## Method 2: Running with Standard Terminal SSH (Command Line)

If you connect via terminal SSH instead of VS Code, you can forward both ports with a single SSH command:

### Step 1: Connect via SSH with Dual Port Forwarding
On your **local laptop terminal**, connect to the server with `-L 8080:localhost:8080` (for Web UI) and `-R 5555:localhost:5555` (for camera video):
```bash
ssh -L 8080:localhost:8080 -R 5555:localhost:5555 user@server_ip
```

### Step 2: Start the Pipeline on the Server
In the SSH session:
```bash
cd /data/thomas/Robot-Needs-Vision
.venv/bin/python Stage1/harness.py --source network --web-port 8080 --stream-port 5555
```

### Step 3: Run the Camera Sender on Your Laptop
In a new terminal window on your **local laptop**:
```bash
python laptop_camera_sender.py --server 127.0.0.1 --port 5555 --left-cam 0 --right-cam 1
```

### Step 4: Open Browser
Open **`http://localhost:8080`** on your laptop.

---

## Method 3: Offline / Mock Mode (Test Right Now Without Any Cameras)

You can run and test the complete pipeline, UI, depth hover, and metrics right now on the server without plugging in any cameras:

```bash
cd /data/thomas/Robot-Needs-Vision
.venv/bin/python Stage1/harness.py --source mock --web-port 8080
```
Then open `http://localhost:8080` in your browser. It continuously feeds the calibrated Middlebury stereo pair.

---

## How to Ace the Live TA Evaluation (15% Metrics)

During the evaluation, Teaching Assistants (TAs) will measure ground-truth distances using a physical tape measure. Here is how to demonstrate all 3 required metrics:

### 1. Distance Accuracy Metric (5%)
1. Aim your stereo camera pair at a lab object (cardboard box, table, chair) at a known distance (e.g. $1.50\text{ m}$ measured by tape).
2. In the web dashboard at `http://localhost:8080`, **click on the box/object face** on the live feed.
   - A green target ROI box will lock onto that position.
3. In the **TA Measuring Tape Verification Tool** at the bottom:
   - Type in the tape measurement: `1.50` metres.
   - The card instantly computes and displays:
     $$\text{Live Error: } \pm 1.8\text{ cm } (1.2\%)$$
   - A green badge confirms high precision ($< 3\text{ cm}$ error).

### 2. Compactness / Density Metric (5% - Absence of Holes)
1. Point the camera at a plain, untextured surface (e.g. white table top or smooth cardboard box).
2. Click the **`StereoSGBM (Traditional)`** button in the top bar:
   - Point out the **black void holes** where classical local matching fails on untextured surfaces.
   - Note the **Compactness score dropping to 28%–65%**.
3. Click the **`CREStereo (Deep DL)`** button in the top bar:
   - The depth map instantly becomes **100.0% solid and hole-free**.
   - Show the TAs the **100% Compactness** badge.

### 3. Speed Metric (5% - Real-time Throughput)
- The live FPS card at the top right displays real-time frame rates and latency (ms) running on the **NVIDIA RTX 4090 GPU**.

### 4. Interactive Hover Depth Inspection
- Move your cursor across the **Depth View**. A live tooltip follows your cursor displaying the exact depth:
  `Depth: 1.48 m (148.2 cm) | (x: 1042, y: 530)`
- Show the TAs the attached **Vertical Metric Depth Color Legend** displaying distances from $0.5\text{ m}$ to $5.0\text{ m}$.

---

## Troubleshooting & FAQ

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| **"Could not open camera device"** on laptop | Wrong camera index | Try `--left-cam 1 --right-cam 2` or check camera permissions in your OS settings. |
| **Left and Right views look inverted** | Camera indices swapped | Swap `--left-cam` and `--right-cam` arguments when running `laptop_camera_sender.py`. |
| **"Connection refused" on laptop** | Server harness is not running or port 5555 is blocked | Make sure `harness.py` is started on the server *before* running `laptop_camera_sender.py`, and ensure port 5555 is forwarded. |
| **Web page cannot connect to 8080** | Port 8080 not forwarded | In VS Code, check the **Ports** tab and ensure port 8080 is listed as Forwarded. In terminal SSH, use `-L 8080:localhost:8080`. |
| **Video feed is laggy** | USB bandwidth saturation | Set `--jpeg-quality 75` and `--fps-limit 25` in `laptop_camera_sender.py` to reduce bandwidth. |
