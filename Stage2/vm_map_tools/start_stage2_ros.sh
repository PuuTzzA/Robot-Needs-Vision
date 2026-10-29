#!/usr/bin/env bash
# Stage 2: One-Step ROS 2 Launch & Diagnostics for TurtleBot 4
# Works in both VMware Bridged mode (recommended) and clean FastDDS state.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_FILE="${SCRIPT_DIR}/maps/final_room.yaml"
LOCALIZATION_PARAMS="${SCRIPT_DIR}/maps/localization_params.yaml"
NAV2_PARAMS="${SCRIPT_DIR}/maps/nav2_params.yaml"

echo "====================================================================="
echo " TurtleBot 4 Stage 2 ROS 2 Environment & Diagnostics Bringup"
echo "====================================================================="

# 1. Clean stale FastDDS shared memory and ros2 daemon (resolves mode switch freezes)
echo "[1/4] Cleaning stale FastDDS shared memory & restarting ROS 2 daemon..."
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true
ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true

# 2. Source ROS 2 environment
source /opt/ros/humble/setup.bash
if [ -f ~/turtlebot4_ws/install/setup.bash ]; then
    source ~/turtlebot4_ws/install/setup.bash
fi

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/etc/turtlebot4/fastdds_rpi.xml
export ROS_DOMAIN_ID=0

# 3. Check Network Interface
echo "[2/4] Network interfaces and IPs detected:"
hostname -I || ip addr show ens33 | grep "inet " || true

# 4. Instructions
echo ""
echo "[3/4] Ready to launch navigation stack."
echo "Map file: ${MAP_FILE}"
echo "Localization params: ${LOCALIZATION_PARAMS}"
echo "Nav2 params: ${NAV2_PARAMS}"
echo ""
echo "To run localization, Nav2, and RViz in separate terminals (recommended):"
echo "  Terminal 1 (Localization):"
echo "    source /opt/ros/humble/setup.bash"
echo "    source ~/turtlebot4_ws/install/setup.bash"
echo "    ros2 launch turtlebot4_navigation localization.launch.py map:=${MAP_FILE} params:=${LOCALIZATION_PARAMS}"
echo ""
echo "  Terminal 2 (Nav2):"
echo "    source /opt/ros/humble/setup.bash"
echo "    source ~/turtlebot4_ws/install/setup.bash"
echo "    ros2 launch turtlebot4_navigation nav2.launch.py params_file:=${NAV2_PARAMS}"
echo ""
echo "  Terminal 3 (RViz):"
echo "    source /opt/ros/humble/setup.bash"
echo "    source ~/turtlebot4_ws/install/setup.bash"
echo "    ros2 launch turtlebot4_viz view_robot.launch.py"
echo ""
echo "  Terminal 4 (Stage 2 Keypoint Tool):"
echo "    source /opt/ros/humble/setup.bash"
echo "    cd ${SCRIPT_DIR}"
echo "    python3 camera_keypoints.py --map ${MAP_FILE} --points ${SCRIPT_DIR}/maps/stage2_keypoints.json"
echo "====================================================================="

