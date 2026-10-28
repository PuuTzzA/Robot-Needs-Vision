#!/usr/bin/env bash
# Stage 2: Robot Controller Bringup for Ubuntu VM
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_MAPS="${SCRIPT_DIR}/../vm_map_tools/maps"

echo "====================================================================="
echo " TurtleBot 4 Stage 2 Robot Controller Bringup"
echo "====================================================================="

# Clean stale FastDDS shared memory
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true

# Source ROS 2 environment
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi
if [ -f ~/turtlebot4_ws/install/setup.bash ]; then
    source ~/turtlebot4_ws/install/setup.bash
fi

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/etc/turtlebot4/fastdds_rpi.xml
export ROS_DOMAIN_ID=0

echo "Detected Network IPs:"
hostname -I || true
echo ""

exec python3 "${SCRIPT_DIR}/robot_controller.py" \
    --map "${STAGE2_MAPS}/final_room.yaml" \
    --points "${STAGE2_MAPS}/stage2_keypoints.json" \
    "$@"

