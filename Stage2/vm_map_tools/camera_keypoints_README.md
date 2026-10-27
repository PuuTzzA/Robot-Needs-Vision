# Stage 2: Camera View & Robot Keypoint Collector

A dedicated tool for Stage 2 on Ubuntu that pairs real-time camera inspection with TurtleBot 4 Nav2 navigation and TF map pose recording.

---

## What It Does
1. **Ubuntu V4L2 Camera Capture**: Captures 30 FPS video with MJPG compression from connected USB cameras (e.g. indices 0 and 2 for stereo) and displays a live OpenCV preview window with a real-time status overlay HUD.
2. **Real-time Map Localization**: Automatically retrieves the robot's current map coordinates $(x, y)$ and rotation (yaw in degrees and quaternion) via ROS 2 TF (`map` $\to$ `base_link`).
3. **Structured JSON Persistence**: Records keypoints with an auto-incrementing unique `id`, `name`, map `position` & `rotation`, and user-defined `visibility` (float between 0.0 and 1.0). Supports multiple entries per keypoint name.
4. **Autonomous Navigation Verification**: Test saved keypoints on the real robot using Nav2 action dispatch (`GOTO <ID>`).
5. **Snapshot Logging**: Automatically saves high-resolution camera snapshots for each recorded keypoint in `snapshots/keypoint_<id>_<name>.jpg`.

---

## Quick Start

### 1. Prerequisites (Launch ROS 2 Localization & Nav2)
In separate terminals on the Ubuntu robot / VM:
```bash
source /opt/ros/humble/setup.bash

# Terminal 1: Localization with map
ros2 launch turtlebot4_navigation localization.launch.py map:=$HOME/Robot-Needs-Vision/Stage2/vm_map_tools/maps/final_room.yaml

# Terminal 2: Nav2
ros2 launch turtlebot4_navigation nav2.launch.py

# Terminal 3: RViz visualization (optional)
ros2 launch turtlebot4_viz view_robot.launch.py
```

### 2. Run the Camera Keypoint Tool
In a new terminal:
```bash
source /opt/ros/humble/setup.bash
cd ~/Robot-Needs-Vision/Stage2/vm_map_tools

python3 camera_keypoints.py \
    --map maps/final_room.yaml \
    --points maps/stage2_keypoints.json
```

---

## Interactive Terminal Commands

| Command | Arguments | Example | Description |
| :--- | :--- | :--- | :--- |
| `SAVE` | `<NAME> <VISIBILITY>` | `SAVE A 0.5` | Saves current robot pose under keypoint name with visibility rating (0.0 to 1.0). Automatically assigns next unique ID and updates the JSON file. |
| `SAVE` | *(none)* | `SAVE` | Immediately writes all recorded keypoints to disk to safeguard progress. |
| `GOTO` | `<ID>` | `GOTO 2` | Navigates the robot to the specified keypoint ID using Nav2 to verify that the pose was stored correctly. |
| `LIST` | *(none)* | `LIST` | Prints a formatted ASCII table showing all saved keypoints, IDs, coordinates, yaws, and visibilities. |
| `DELETE` | `<ID>` | `DELETE 1` | Deletes the entry with the given unique ID and updates the JSON file. |
| `STATUS` | *(none)* | `STATUS` | Displays current robot map coordinates, camera FPS, and Nav2 server status. |
| `CANCEL` | *(none)* | `CANCEL` | Cancels any ongoing Nav2 navigation. |
| `HELP` | *(none)* | `HELP` | Prints the command reference and description banner. |
| `QUIT` | *(none)* | `QUIT` | Flushes data, safely shuts down cameras, and exits. |

---

## OpenCV Window Controls
When the camera window is focused:
- **W / S**: Move forward / backward
- **A / D**: Turn left / right
- **Space**: Stop movement
- **S**: Save a manual camera snapshot
- **Q / ESC**: Exit program

---

## Standalone Camera Test
To verify USB camera connectivity independently:
```bash
python3 Stage2/camera_test.py --indices 0 2 --width 1280 --height 720
```

