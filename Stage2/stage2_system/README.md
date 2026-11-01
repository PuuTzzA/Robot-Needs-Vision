# Stage 2: Autonomous Perception & Vision-Language Navigation

A unified, distributed system for **Stage 2 perception and multi-level autonomous tasks** on the TurtleBot 4 Lite.

---

## 1. System Architecture

The architecture coordinates three separate execution nodes communicating over low-latency HTTP and UDP:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             WINDOWS LAPTOP                                  │
│                                                                             │
│  ┌─────────────────────────┐               ┌─────────────────────────────┐  │
│  │ USB Stereo Cameras      │               │ Windows Mission Control GUI │  │
│  │ (Left + Right Webcams)  │               │ (laptop/main.py)            │  │
│  └───────────┬─────────────┘               └──────────────┬──────────────┘  │
│              │ (Local USB)                                │                 │
│              ▼                                            │                 │
│        laptop/camera.py ──────────────────────────────────┘                 │
│                                            │                                │
│                     HTTP /api/goto,        │        HTTP POST Lossless      │
│                     /api/teleop (Port 8070)│        Stereo PNG (Port 8082)  │
│                     HTTP Status (Port 8071)│                                │
└─────────────────────┬──────────────────────┼────────────────┬───────────────┘
                      │                      │                │
══════════════════════╪══════════════════════╪════════════════╪════════════════
                      ▼                      │                ▼
┌─────────────────────────────────────────┐  │  ┌───────────────────────────┐
│           UBUNTU VM (ROS 2)             │  │  │      SSH GPU SERVER       │
│                                         │  │  │                           │
│ vm/robot_controller.py                  │  │  │ ssh/size_server.py        │
│ ├── Nav2 NavigateToPose ActionClient    │  │  │ ├── Stereo Calibration    │
│ ├── TF Pose Tracker (map -> base_link)  │  │  │ ├── CREStereo ONNX        │
│ ├── /cmd_vel Publisher for Teleop       │  │  │ ├── YOLO Instance Seg     │
│ ├── In-place Keypoint & Map hot-reload  │  │  │ └── 3D Metric Bounding    │
│ └── UDP Discovery Beacon (Port 8079)    │  │  │     Boxes (cm)            │
└─────────────────────────────────────────┘  │  └───────────────────────────┘
                                             │
                                             ▼
                                ┌───────────────────────────┐
                                │      MULTIMODAL LLM       │
                                │ (Gemini / OpenAI / Claude)│
                                │ laptop/llm/llm_api.py     │
                                └───────────────────────────┘
```

### Network Ports

| Subsystem | Port | Protocol | Description |
| :--- | :--- | :--- | :--- |
| **VM Robot Controller** | `8070` | HTTP | Command API (`/api/goto`, `/api/teleop`, `/api/cancel`, `/api/keypoints`, `/api/map`) |
| **VM Discovery Beacon** | `8079` | UDP | Auto-discovery beacon responding to `TB4_ROBOT_DISCOVERY` queries |
| **Laptop Ping Receiver** | `8071` | HTTP | Status ping receiver receiving periodic 2 Hz telemetry from VM |
| **SSH Size Server** | `8082` | HTTP | Single-shot stereo height & dimension measurement API |

---

## 2. Workflows by Level

### Level 1: Directed Inspection
1. Robot moves to **Point 0** (Task Center), waits ~1.0s for camera stabilization, and captures a fresh photo of the laptop screen.
2. Multimodal LLM parses instructions to identify two target waypoint locations (e.g., `Point B` and `Point E`).
3. For each target point (ordered by closest Euclidean distance):
   - Navigates to the closest recorded keypoint entry, settles ~1.0s, and captures a fresh photo.
   - Immediately dispatches parallel navigation back to the initial RViz **Start** pose once the final photo is taken.
   - LLM classifies object category and overlays the classification result directly onto **Last Processed Frame**.
4. Displays final result in bottom banner and completes return to **Start**.

### Level 2: Category Search & Early Exit
1. Robot moves to **Point 0** (Task Center), settles ~1.0s, and captures a fresh photo of task instructions.
2. LLM extracts the two target object categories to find (e.g., `Mug` and `Book`).
3. Visits object points one by one in greedy closest-Euclidean-distance order:
   - Navigates to the closest entry, settles ~1.0s, takes a fresh photo, classifies the object, and overlays the result on **Last Processed Frame**.
   - **Early Exit**: as soon as both categories are discovered, immediately stops searching, publishes the solution, and navigates back to the initial RViz **Start** pose.

### Level 3: Async GPU Measurement & One-Shot Multimodal Reasoning
1. Robot moves to **Point 0**, settles ~1.0s, and captures `task_level3.jpg` (**no initial LLM API call**).
2. Visits **all** object locations in greedy closest-Euclidean-distance order (no visibility filtering):
   - Settles ~1.0s and captures a fresh synchronized stereo pair.
   - Submits the lossless stereo composite PNG **asynchronously** to the SSH GPU Size Server (for YOLO bounding box, class, and metric stereo height) while **immediately driving to the next point** (or driving back to **Start** after the final photo).
   - Overlays each returning GPU bounding box + class + height onto **Last Processed Frame**.
3. Executes a single end-of-run multimodal LLM query (`solve_level3`) with `task_level3.jpg`, all object photos, and all GPU estimates to transcribe the task, synthesize the inventory, and output the final answer while returning to **Start**.

---

## 3. How to Run the System

### Step 1: Start Remote GPU Size Server (SSH Server)
On the remote NVIDIA RTX 4090 GPU machine:
```bash
cd /data/thomas/Robot-Needs-Vision
./Stage2/stage2_system/ssh/run_size_server.sh --port 8082 --device cuda:0
```
*(If `ultralytics` has not been installed into `.venv` yet, run `.venv/bin/python -m pip install -r Stage2/stage2_system/requirements-ssh.txt` once first. In VS Code, ensure port **8082** is forwarded to `localhost:8082` on your Windows machine).*

---

### Step 2: Start ROS 2 & Robot Controller (Ubuntu VM)

The VM side separates the ROS 2 navigation bringup from the Python robot controller so you can quickly restart `robot_controller.py` without restarting Localization, Nav2, or RViz2:

#### Terminal 1 — Automated Navigation Bringup (`auto_bringup.sh`)
```bash
cd ~/Robot-Needs-Vision/Stage2/stage2_system/vm
chmod +x auto_bringup.sh start_robot_controller.sh
./auto_bringup.sh
```

**What this script does automatically (with live status & 300s timeout per step):**
1. **[1/5] Environment & Stale Node Cleanup**: Sources ROS 2 Humble & workspace and cleanly terminates leftover navigation nodes.
2. **[2/5] Hardware Connectivity Verification**: Checks that `/odom` and `/scan` are being actively published by the TurtleBot 4 / Create 3 base.
3. **[3/5] Localization Bringup**: Launches `map_server` and AMCL with `final_room.yaml` and `localization_params.yaml`.
4. **[4/5] Nav2 Navigation Stack**: Launches the Nav2 planners, costmaps, and behavior tree navigator with `nav2_params.yaml`.
5. **[5/5] RViz2 Visualization**: Launches RViz2 with software rendering (`LIBGL_ALWAYS_SOFTWARE=1`) so you can inspect the map and robot.

Keep Terminal 1 open (`Ctrl+C` stops Nav2 & RViz).

#### Terminal 2 — Initial Pose Confirmation & Stage 2 Python Controller (`start_robot_controller.sh`)
Open a **second terminal** in the Ubuntu VM:
```bash
cd ~/Robot-Needs-Vision/Stage2/stage2_system/vm
./start_robot_controller.sh
```
1. Verifies Nav2 `/navigate_to_pose` action server readiness.
2. Prompts you to click **2D Pose Estimate** in RViz, align the red laser dots with the map walls, press **Enter** to view the live `(x, y, yaw°)` pose, and confirm `[y/N]` (automatically clearing Nav2 costmaps).
3. Whenever you carry the robot back to the start to restart a task, either type `p` + **Enter** in Terminal 2 to re-confirm the new RViz pose in place, or press `Ctrl+C` and re-run `./start_robot_controller.sh` — without restarting Terminal 1.

---

#### Alternative: Manual / Separate ROS 2 Bringup
If you prefer running every ROS 2 launch file in its own terminal:

**Terminal 1 — Localization:**
```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot4_ws/install/setup.bash

ros2 launch turtlebot4_navigation localization.launch.py \
    map:=$HOME/Robot-Needs-Vision/Stage2/vm_map_tools/maps/final_room.yaml \
    params:=$HOME/Robot-Needs-Vision/Stage2/vm_map_tools/maps/localization_params.yaml
```

**Terminal 2 — Nav2:**
```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot4_ws/install/setup.bash

ros2 launch turtlebot4_navigation nav2.launch.py \
    params_file:=$HOME/Robot-Needs-Vision/Stage2/vm_map_tools/maps/nav2_params.yaml
```

**Terminal 3 — RViz2:**
```bash
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch turtlebot4_viz view_robot.launch.py
```
*(Use 2D Pose Estimate in RViz to set the initial pose)*

**Terminal 4 — Stage 2 Robot Controller:**
```bash
cd ~/Robot-Needs-Vision/Stage2/stage2_system/vm
./start_robot_controller.sh
```

---

### Step 3: Start Mission Control (Windows Laptop)
On your Windows laptop (in PowerShell or CMD):
```powershell
conda activate phienv
cd Stage2\stage2_system
python laptop\main.py --vm auto
```
At startup, a live preview window opens showing **all available cameras** annotated with their IDs and saved `[LEFT]` / `[RIGHT]` assignments. Press **Enter** in the terminal to confirm the saved pair, or type e.g. `left 0, right 2` to reassign and save for next time (pass `--no-cam-select` to skip the prompt).

> **Note on Auto-Discovery**: `--vm auto` automatically searches for the VM via VMware guest IP query and multi-interface UDP broadcast. You can also specify the IP manually:
> ```powershell
> python laptop\main.py --vm 192.168.31.253
> ```

---

## 4. Offline Simulation / Mock Mode

You can test and demonstrate the complete system without any physical robot or cameras:

1. **Terminal 1 (Mock Robot Controller):**
   ```powershell
   conda activate phienv
   cd Stage2\stage2_system
   python vm\robot_controller.py --sim --port 8070
   ```

2. **Terminal 2 (Mission Control with Mock Feeds):**
   ```powershell
   conda activate phienv
   cd Stage2\stage2_system
   python laptop\main.py --vm 127.0.0.1 --mock-camera --mock-llm --mock-size
   ```
   Now you can click buttons or type `Level 1`, `Level 2`, `Level 3`, `Goto B`, `W`/`A`/`S`/`D` and watch the simulated robot move on the map and complete tasks!

---

## 5. Supported GUI Commands

Enter commands into the **Input** text box in the right sidebar or focus the window for direct keyboard teleop:

| Command | Arguments | Example | Description |
| :--- | :--- | :--- | :--- |
| `Level` | `<1\|2\|3>` | `Level 1` | Starts autonomous execution of specified level |
| `Restart` | `<1\|2\|3>` | `Restart 2` | Aborts current run, clears previous results, and restarts level |
| `Stop` | *(none)* | `Stop` | Cancels active navigation and halts running mission |
| `Goto` | `<Point>` | `Goto B` | Dispatches robot to closest keypoint entry by Euclidean distance |
| `Goto` | `#<ID>` | `Goto #2` | Sends robot to specific keypoint ID |
| `Home` | *(none)* | `Home` | Navigates robot back to its latched initial RViz Start pose |
| `Ping` | `<VM\|SSH\|LLM>` | `Ping VM` | Tests round-trip latency to specified subsystem |
| `W` / `S` | *(none)* | `W` | Drives robot forward / backward (0.15 m/s burst) |
| `A` / `D` | *(none)* | `A` | Turns robot left / right (0.50 rad/s burst) |
| `Space` | *(none)* | `Space` | Immediately halts teleop motion |
| `Photo` | *(none)* | `Photo` | Captures fresh camera frame and shows it in Last Processed Frame |
| `Measure` | `[hint]` | `Measure bottle` | Sends stereo pair to SSH GPU server and overlays bounding box + height |
| `Classify` | *(none)* | `Classify` | Queries LLM to classify current camera frame and overlays result on preview |
| `CamReset` | *(none)* | `CamReset` | Force-reopens USB camera streams if a webcam glitches |
| `Points` / `List`| *(none)* | `Points` | Displays table of all loaded keypoints, coordinates, and visibilities |
| `Status` | *(none)* | `Status` | Prints robot connection, pose, Nav2 state, and loaded points |
| `Help` | *(none)* | `Help` | Prints command reference cheat-sheet |

---

## 6. Keypoints & Map Handling (In-Place Read-Only)

- **Zero Overwriting**: The system **never modifies or overwrites** `stage2_keypoints.json` or `final_room.yaml`.
- **Live Hot-Reloading**: The keypoint store checks file `mtime` on every query. If you record new keypoints using `camera_keypoints.py` on the VM, the controller detects the change immediately without requiring a restart.
- **Selection Rule**:
  - By default (`min_visibility=0.0`), selects the entry matching the target name with the smallest Euclidean distance to the robot's current $(x, y)$ position (no visibility filtering).
  - If an explicit `min_visibility > 0` threshold is supplied programmatically, filters by `visibility >= min_visibility` first and falls back to the highest available visibility if none pass.

---

## 7. Configuring the LLM API Key (DeepSeek)

Both `laptop/llm/llm_api.py` (Levels 1–3, `Classify`, `Ping LLM`) and `laptop/llm/object_analysis.py` (manual **Capture & Analyze**) authenticate with `DEEPSEEK_API_KEY`.

**Recommended (git-ignored `.env` file — set once):**
1. Copy `Stage2/stage2_system/.env.example` to `Stage2/stage2_system/.env` (if `.env` does not already exist).
2. Paste your API key after `DEEPSEEK_API_KEY=` in `Stage2/stage2_system/.env` and save. (`.env` is ignored by Git.)
3. Verify connectivity in Mission Control by running `Ping LLM`.

**Alternative (shell environment variable):**
Set `$env:DEEPSEEK_API_KEY = "..."` in PowerShell (or `export DEEPSEEK_API_KEY="..."` in Bash) before launching `laptop/main.py`. If set in the shell, the environment variable takes precedence over `.env`.

---

## 8. Running Automated Unit & Integration Tests

From the `stage2_system` folder in `phienv`:
```powershell
conda activate phienv
cd Stage2\stage2_system
python -m unittest discover -s tests -v
```

All 4 test suites verify:
- `test_protocol_roundtrip.py`: Dataclass serialization, network schemas, and sync between shared/vm/ssh copies.
- `test_keypoints.py`: Coordinate parsing, greedy nearest-neighbour ordering, visibility filtering, and mtime hot-reload without file modification.
- `test_commands.py`: Command parsing, syntax validation, and teleop dispatch.
- `test_mission_mock.py`: End-to-end integration test executing Levels 1, 2, and 3 autonomously against a simulated robot and mock perception services.

