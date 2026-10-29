# Stage 2: Camera View & Robot Keypoint Collector

A dedicated tool for Stage 2 on Ubuntu that pairs real-time camera inspection with TurtleBot 4 Nav2 navigation and TF map pose recording.

---

## What It Does
1. **Hybrid Camera Input**:
   - **Direct USB Capture**: Native V4L2 on Ubuntu (`/dev/video*`).
   - **Network Streaming from Windows Host**: Stream your laptop camera over HTTP via `laptop_camera_sender.py` with automatic UDP LAN beacon discovery (`--server auto`). Avoids VMware USB packet drops and works across NAT and Bridged networks.
2. **Real-time Map Localization**: Automatically tracks the robot's map pose $(x, y, \text{yaw})$ via ROS 2 TF (`map` $\to$ `base_link`).
3. **Structured JSON Persistence**: Records keypoints with unique `id`, `name`, `position`, `rotation`, and `visibility` (0.0 to 1.0) into `maps/stage2_keypoints.json`.
4. **Autonomous Navigation Verification**: Test saved keypoints directly on the robot using Nav2 action dispatch (`GOTO <ID>`).
5. **Snapshot Logging**: Automatically saves high-resolution camera snapshots for each recorded keypoint in `snapshots/keypoint_<id>_<name>.jpg`.

---

## Complete Workflow & Setup

### Autostart
```bash
cd ~/Robot-Needs-Vision/Stage2/vm_map_tools ./auto_nav_bringup.sh
```

### Step 0: VMware Network Configuration
To allow ROS 2 to communicate with the physical TurtleBot 4 while keeping your Windows VPN active:
- **Dual Network Adapters (Recommended)**:
  - **Adapter 1**: Set to **NAT** (keeps Windows VPN and Internet access working for the VM).
  - **Adapter 2**: In VMware (`VM -> Settings -> Add... -> Network Adapter`), add a second adapter set to **Bridged**, and in VMware's *Virtual Network Editor*, explicitly bridge it to your laptop's **physical Wi-Fi card** (not the VPN TAP adapter).
- **Single Adapter (Bridged)**:
  - Bridge the VM's network adapter directly to your physical Wi-Fi card connected to the robot's network.

---

### Step 1: Pre-Flight Cleanup (Run once in Ubuntu VM)
If you recently switched network modes or restarted ROS 2, clean stale FastDDS shared-memory locks:
```bash
~/Robot-Needs-Vision/Stage2/vm_map_tools/start_stage2_ros.sh
```

---


### Step 2: Launch ROS 2 Stack (Ubuntu VM)
Open **3 separate terminals** in Ubuntu:

**Terminal 1 — Localization (Auto-initializes `map` frame)**:
```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot4_ws/install/setup.bash

ros2 launch turtlebot4_navigation localization.launch.py \
    map:=$HOME/Robot-Needs-Vision/Stage2/vm_map_tools/maps/final_room.yaml \
    params:=$HOME/Robot-Needs-Vision/Stage2/vm_map_tools/maps/localization_params.yaml
```

**Terminal 2 — Nav2 Navigation**:
```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot4_ws/install/setup.bash

ros2 launch turtlebot4_navigation nav2.launch.py
```

**Terminal 3 — RViz Visualization**:
```bash
source /opt/ros/humble/setup.bash

ros2 launch turtlebot4_viz view_robot.launch.py
```
> **Tip**: In RViz, the Fixed Frame defaults to `map`. Click **2D Pose Estimate** on the top toolbar and drag an arrow matching the robot's physical location to align the laser scan with the walls.

---

### Step 3: Run the Camera Keypoint Tool (Ubuntu VM)
In a **4th terminal** in the VM:
```bash
source /opt/ros/humble/setup.bash
cd ~/Robot-Needs-Vision/Stage2/vm_map_tools

python3 camera_keypoints.py \
    --map maps/final_room.yaml \
    --points maps/stage2_keypoints.json
```
The console will start an HTTP receiver on port `8080` and activate a UDP discovery beacon on port `8089`.

---

### Step 4: Stream Camera Feed from Windows Host
On your Windows laptop (Command Prompt or PowerShell):
```cmd
python Stage2\laptop_camera_sender.py --single-cam 0
```
> **Auto-Discovery**: `laptop_camera_sender.py` defaults to `--server auto`. It will broadcast on the local subnet, automatically discover the VM's IP address (whether NAT or Bridged), and stream frames to port `8080`.
> 
> *If you need to specify the IP manually:*
> ```cmd
> python Stage2\laptop_camera_sender.py --server <VM_IP> --port 8080 --single-cam 0
> ```

---

## Interactive Terminal Commands (`camera_keypoints.py`)

| Command | Arguments | Example | Description |
| :--- | :--- | :--- | :--- |
| `SAVE` | `<NAME> <VISIBILITY>` | `SAVE A 0.5` | Saves current robot map pose with keypoint name and visibility score (0.0 to 1.0). Writes to JSON and takes a snapshot. |
| `SAVE` | *(none)* | `SAVE` | Manually flushes/persists all keypoints to the JSON file. |
| `GOTO` | `<ID>` | `GOTO 2` | Sends Nav2 `NavigateToPose` goal to verify the robot can navigate back to that keypoint. |
| `LIST` | *(none)* | `LIST` | Displays a table of all recorded keypoints, coordinates, yaws, visibilities, and snapshots. |
| `DELETE` | `<ID>` | `DELETE 1` | Removes the keypoint with the specified ID. |
| `STATUS` | *(none)* | `STATUS` | Shows current robot map pose, camera FPS, and Nav2 server status. |
| `CANCEL` | *(none)* | `CANCEL` | Cancels any ongoing Nav2 goal execution. |
| `HELP` | *(none)* | `HELP` | Displays command reference banner. |
| `QUIT` | *(none)* | `QUIT` | Saves all keypoints, closes camera streams, and exits cleanly. |

---

## OpenCV Window Controls
When the camera preview window is focused:
- **W / S**: Teleop drive forward / backward
- **A / D**: Teleop turn left / right
- **Space**: Stop robot motion
- **F**: Save manual snapshot (foto)
- **Q / ESC**: Exit program
