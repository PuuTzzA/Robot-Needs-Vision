#!/usr/bin/env bash
# Stage 2: Robot Controller Bringup for Ubuntu VM
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_MAPS="$(cd "${SCRIPT_DIR}/../../vm_map_tools/maps" 2>/dev/null && pwd)"

echo "====================================================================="
echo " TurtleBot 4 Stage 2 Robot Controller Bringup"
echo "====================================================================="

# NOTE: Do NOT rm /dev/shm/fastrtps_* here — Nav2/AMCL are already running and
# deleting shared memory files corrupts their active DDS transports.

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

exec python3 -u "${SCRIPT_DIR}/robot_controller.py" \
    --map "${STAGE2_MAPS}/final_room.yaml" \
    --points "${STAGE2_MAPS}/stage2_keypoints.json" \
    "$@"

