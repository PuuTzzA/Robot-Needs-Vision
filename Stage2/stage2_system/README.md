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
1. Robot moves to **Point 0** (Task Center).
2. Captures photo of the laptop screen displaying task instructions.
3. Multimodal LLM parses instructions to identify two target waypoint locations (e.g., `Point B` and `Point E`).
4. For each target point:
   - Navigates to the closest recorded keypoint entry with visibility $> 0.10$.
   - Captures photo of the object.
   - LLM classifies object category (e.g. Cup, Banana).
5. Displays final result in bottom banner (e.g., `Point B: Cup, Point E: Banana`).

### Level 2: Category Search & Early Exit
1. Robot moves to **Point 0** (Task Center).
2. Captures photo of task instructions.
3. LLM extracts the two target object categories to find (e.g., `Mug` and `Book`).
4. Generates greedy nearest-neighbour path visiting object points one by one.
5. At each point:
   - Navigates to keypoint with visibility $> 0.10$.
   - Takes photo and classifies object.
   - If an object matches a target category, records its location.
   - **Early Exit**: as soon as both categories are discovered, immediately stops searching and displays the solution.

### Level 3: Multimodal Measurement & Reasoning
1. Robot moves to **Point 0**.
2. Captures photo of Level 3 task instructions.
3. First LLM query extracts the task text and determines whether size/height estimation is needed (`use_high_visibility`).
4. If height is needed, the robot uses keypoints with visibility $> 0.75$; otherwise, visibility $> 0.10$.
5. Visits **all** object locations in greedy order:
   - Captures visual photo.
   - If size estimation is required: captures lossless stereo pair, queries SSH GPU Size Server, and receives bounding box and metric height in cm.
6. Executes one-shot multimodal reasoning with the LLM, cross-referencing all photos and sensor height measurements against the known 9 cm pedestal reference.
7. Displays final answer and inventory.

---

## 3. How to Run the System

### Step 1: Start Remote GPU Size Server (SSH Server)
On the remote NVIDIA RTX 4090 GPU machine:
```bash
cd ~/Robot-Needs-Vision/Stage2/stage2_system/ssh
./run_size_server.sh --port 8082 --device cuda:0
```
*(In VS Code, ensure port **8082** is forwarded to `localhost:8082` on your Windows machine).*

---

### Step 2: Start ROS 2 & Robot Controller (Ubuntu VM)

The VM side includes a fully automated 1-command bringup script modeled after `auto_nav_bringup.sh`:

#### Recommended: Automated 1-Command Bringup
Open **one terminal** in the Ubuntu VM:
```bash
cd ~/Robot-Needs-Vision/Stage2/stage2_system/vm
chmod +x auto_bringup.sh
./auto_bringup.sh
```

**What this script does automatically (with live status & 300s timeout per step):**
1. **[1/7] Environment & Stale Node Cleanup**: Sources ROS 2 Humble & workspace, clears lingering FastDDS shared memory, and cleanly terminates leftover nodes.
2. **[2/7] Hardware Connectivity Verification**: Checks that `/odom` and `/scan` are being actively published by the TurtleBot 4 / Create 3 base.
3. **[3/7] Localization Bringup**: Launches `map_server` and AMCL with `final_room.yaml` and `localization_params.yaml`.
4. **[4/7] Nav2 Navigation Stack**: Launches the Nav2 planners, costmaps, and behavior tree navigator with `nav2_params.yaml`.
5. **[5/7] RViz2 Visualization**: Launches RViz2 with software rendering (`LIBGL_ALWAYS_SOFTWARE=1`) so you can inspect the map and robot.
6. **[6/7] Manual Initial Pose Confirmation**:
   - In RViz, click **2D Pose Estimate** on the top toolbar.
   - Click the robot's real position on the map and drag along its facing heading until the red laser scans align with the map walls.
   - Return to the terminal; the script reads `/amcl_pose` and prompts: `Is the robot correctly localized on the map? [y/N]`. Type `y` and press `Enter` to confirm.
7. **[7/7] Robot Controller Launch**: Launches `robot_controller.py` with unbuffered output, verifies the HTTP command server binds (port 8070), checks connection to the Nav2 `/navigate_to_pose` action server, and begins streaming live logs.

Press `Ctrl+C` at any time to cleanly stop all nodes and the controller.

---

#### Alternative: Manual / Separate Bringup
If you prefer running components in separate terminals or Nav2 is already running:

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
# Or using the bringup script in controller-only mode:
# ./auto_bringup.sh --controller-only
```

---

### Step 3: Start Mission Control (Windows Laptop)
On your Windows laptop (in PowerShell or CMD):
```powershell
conda activate phienv
cd Stage2\stage2_system
python laptop\main.py --vm auto --left-cam 0 --right-cam 2
```

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
| `Goto` | `<Point>` | `Goto B` | Dispatches robot to closest keypoint entry for point name (vis $> 0.10$) |
| `Goto` | `#<ID>` | `Goto #2` | Sends robot to specific keypoint ID |
| `Ping` | `<VM\|SSH\|LLM>` | `Ping VM` | Tests round-trip latency to specified subsystem |
| `W` / `S` | *(none)* | `W` | Drives robot forward / backward (0.15 m/s burst) |
| `A` / `D` | *(none)* | `A` | Turns robot left / right (0.50 rad/s burst) |
| `Space` | *(none)* | `Space` | Immediately halts teleop motion |
| `Photo` | *(none)* | `Photo` | Captures camera frame and shows it in Last Processed Frame |
| `Measure` | `[hint]` | `Measure bottle` | Sends stereo pair to SSH GPU server and displays bounding box + height |
| `Classify` | *(none)* | `Classify` | Queries LLM to classify object currently in camera frame |
| `Points` / `List`| *(none)* | `Points` | Displays table of all loaded keypoints, coordinates, and visibilities |
| `Status` | *(none)* | `Status` | Prints robot connection, pose, Nav2 state, and loaded points |
| `Help` | *(none)* | `Help` | Prints command reference cheat-sheet |

---

## 6. Keypoints & Map Handling (In-Place Read-Only)

- **Zero Overwriting**: The system **never modifies or overwrites** `stage2_keypoints.json` or `final_room.yaml`.
- **Live Hot-Reloading**: The keypoint store checks file `mtime` on every query. If you record new keypoints using `camera_keypoints.py` on the VM, the controller detects the change immediately without requiring a restart.
- **Selection Rule**:
  - Filters entries matching the target name by `visibility >= min_visibility`.
  - Picks the eligible entry with the smallest Euclidean distance to the robot's current $(x, y)$ position.
  - If no entry meets the threshold, logs a warning and falls back to the highest available visibility.

---

## 7. Connecting Your Chosen LLM Provider

The LLM interface in `laptop/llm/llm_api.py` is modular and provider-agnostic. To use Google Gemini:
1. Ensure the Google GenAI SDK is installed: `pip install google-genai`
2. Set your environment variable: `export GEMINI_API_KEY="your-api-key"` (or `$env:GEMINI_API_KEY="..."` in PowerShell).
3. Open `laptop/llm/llm_api.py` and implement `query_structured()` following the provided example docstring.

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

