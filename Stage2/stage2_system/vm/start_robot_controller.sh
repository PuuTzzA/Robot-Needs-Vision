#!/usr/bin/env bash
# Stage 2: Robot Controller Bringup for Ubuntu VM (run in a separate terminal)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_MAPS="$(cd "${SCRIPT_DIR}/../../vm_map_tools/maps" 2>/dev/null && pwd)"

echo "====================================================================="
echo " TurtleBot 4 Stage 2 Robot Controller (Standalone Terminal)"
echo "====================================================================="

# NOTE: Do NOT rm /dev/shm/fastrtps_* here — Nav2/AMCL are already running and
# deleting shared memory files corrupts their active DDS transports.

# Clean up any leftover background robot_controller.py holding the command port
pkill -f "stage2_system/vm/robot_controller.py" 2>/dev/null || true
pkill -f "vm/robot_controller.py" 2>/dev/null || true
sleep 0.3

# Source ROS 2 environment
if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi
if [ -f "${HOME}/turtlebot4_ws/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/turtlebot4_ws/install/setup.bash"
fi

export CYCLONEDDS_URI=/etc/turtlebot4/cyclonedds_rpi.xml
if [ -f /etc/turtlebot4/fastdds_rpi.xml ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE=/etc/turtlebot4/fastdds_rpi.xml
fi
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

echo "Detected Network IPs:"
hostname -I || true
echo ""

exec python3 -u "${SCRIPT_DIR}/robot_controller.py" \
    --map "${STAGE2_MAPS}/final_room.yaml" \
    --points "${STAGE2_MAPS}/stage2_keypoints.json" \
    "$@"
