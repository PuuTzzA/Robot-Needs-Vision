#!/usr/bin/env bash
# ==============================================================================
# TurtleBot 4 Stage 2 System: Strict Step-by-Step VM Navigation Bringup
# ==============================================================================
# Adapted copy of Stage2/vm_map_tools/auto_nav_bringup.sh for the Stage 2 main
# program. Starts the ROS 2 navigation side with a single command:
#
#   1. Environment + cleanup of leftover navigation nodes
#   2. Robot connectivity check (/odom, /scan)
#   3. Localization (map_server + AMCL)
#   4. Nav2 (with vm_map_tools/maps/nav2_params.yaml)
#   5. RViz2 (required to set the initial pose)
#   6. Manual initial pose: set "2D Pose Estimate" in RViz, then confirm here
#
# NOTE: Step 7 (robot_controller.py) is intentionally NOT launched here so you
# can run and restart the Python script quickly in a separate terminal without
# restarting Localization, Nav2, or RViz2:
#     cd ~/Robot-Needs-Vision/Stage2/stage2_system/vm
#     ./start_robot_controller.sh
#
# - Every step must succeed before the next one starts.
# - Live 1-second status feedback, 300s timeout per step.
# - If any step fails or any core nav process dies, everything is cleaned up.
# - Map / params / keypoints are read in place from Stage2/vm_map_tools/maps.
# ==============================================================================

set -o pipefail

# Text styling
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
NC="\033[0m"

STEP_TIMEOUT=300  # 5 minutes (300 seconds) per step

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPS_DIR="$(cd "${SCRIPT_DIR}/../../vm_map_tools/maps" 2>/dev/null && pwd)"
MAP_FILE="${MAPS_DIR}/final_room.yaml"
PARAMS_FILE="${MAPS_DIR}/localization_params.yaml"
NAV2_PARAMS_FILE="${MAPS_DIR}/nav2_params.yaml"
POINTS_FILE="${MAPS_DIR}/stage2_keypoints.json"
CONTROLLER="${SCRIPT_DIR}/robot_controller.py"

LOG_DIR="/tmp/stage2_system_bringup_logs"
mkdir -p "${LOG_DIR}"

LOG_LOC="${LOG_DIR}/localization.log"
LOG_NAV="${LOG_DIR}/nav2.log"
LOG_RVIZ="${LOG_DIR}/rviz.log"

PID_LOC=""
PID_NAV=""
PID_RVIZ=""

CHECK_ONLY=false
CTRL_PORT=8070

kill_existing_nav_nodes() {
    local patterns=(
        "turtlebot4_navigation"
        "turtlebot4_viz"
        "nav2_map_server"
        "map_server"
        "nav2_amcl"
        "amcl"
        "controller_server"
        "smoother_server"
        "planner_server"
        "behavior_server"
        "bt_navigator"
        "waypoint_follower"
        "velocity_smoother"
        "lifecycle_manager"
        "rviz2"
    )
    for pat in "${patterns[@]}"; do
        pkill -f "${pat}" 2>/dev/null || true
    done
    sleep 1
    for pat in "${patterns[@]}"; do
        pkill -9 -f "${pat}" 2>/dev/null || true
    done
}

CLEANED_UP=false
cleanup() {
    if [ "${CLEANED_UP}" = true ]; then
        return
    fi
    CLEANED_UP=true
    echo -e "\n${YELLOW}[!] Shutting down Stage 2 navigation nodes...${NC}"

    if [ -n "${PID_RVIZ}" ] && kill -0 "${PID_RVIZ}" 2>/dev/null; then
        kill "${PID_RVIZ}" 2>/dev/null || true
    fi
    if [ -n "${PID_NAV}" ] && kill -0 "${PID_NAV}" 2>/dev/null; then
        kill "${PID_NAV}" 2>/dev/null || true
    fi
    if [ -n "${PID_LOC}" ] && kill -0 "${PID_LOC}" 2>/dev/null; then
        kill "${PID_LOC}" 2>/dev/null || true
    fi
    pkill -P $$ 2>/dev/null || true
    kill_existing_nav_nodes
    echo -e "${GREEN}[✔] Clean exit completed.${NC}"
}

fail_and_abort() {
    local step_name="$1"
    local log_file="$2"
    echo -e "\n${RED}${BOLD}=====================================================================${NC}"
    echo -e "${RED}${BOLD}[ERROR] Step Failed: ${step_name}${NC}"
    echo -e "${RED}${BOLD}Aborting bringup — will not continue to the next step.${NC}"
    if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
        echo -e "${YELLOW}--- Last 15 lines of ${log_file} ---${NC}"
        tail -n 15 "${log_file}"
        echo -e "${YELLOW}--------------------------------------------------${NC}"
    fi
    echo -e "${RED}${BOLD}=====================================================================${NC}"
    exit 1
}

# Abort if localization / Nav2 died while we were waiting for something else
check_nav_stack_alive() {
    if [ -n "${PID_LOC}" ] && ! kill -0 "${PID_LOC}" 2>/dev/null; then
        fail_and_abort "Localization process terminated unexpectedly" "${LOG_LOC}"
    fi
    if [ -n "${PID_NAV}" ] && ! kill -0 "${PID_NAV}" 2>/dev/null; then
        fail_and_abort "Nav2 process terminated unexpectedly" "${LOG_NAV}"
    fi
}

# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --check-only)
            CHECK_ONLY=true
            ;;
        --controller-only)
            echo -e "${YELLOW}[INFO] robot_controller.py is now run separately so you can restart it freely.${NC}"
            echo -e "Run in your terminal:"
            echo -e "  ${BOLD}cd ${SCRIPT_DIR} && ./start_robot_controller.sh${NC}"
            exit 0
            ;;
        --port)
            shift
            CTRL_PORT="$1"
            ;;
        --help|-h)
            echo "Usage: ./auto_bringup.sh [OPTIONS]"
            echo "Options:"
            echo "  --check-only        Run environment + /odom & /scan diagnostics only"
            echo "  --port N            Port shown in next-step instructions (default: 8070)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Note: After bringup finishes Step 6, start robot_controller.py in a"
            echo "      separate terminal with ./start_robot_controller.sh"
            echo ""
            echo "Logs: ${LOG_DIR}/"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}[WARN] Unknown option '$1'${NC}"
            ;;
    esac
    shift
done

echo -e "${BLUE}${BOLD}=====================================================================${NC}"
echo -e "${BLUE}${BOLD}   TurtleBot 4 Stage 2 System: Automated VM Navigation Bringup       ${NC}"
echo -e "${BLUE}${BOLD}   (Strict Mode: 300s timeout/step, halts on any failure)            ${NC}"
echo -e "${BLUE}${BOLD}=====================================================================${NC}"

# ------------------------------------------------------------------------------
# STEP 1: Source Environment & Clean Leftover Processes
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[1/6] Setting up ROS 2 Humble Environment & Cleaning Leftover Nodes...${NC}"

if [ "$CHECK_ONLY" = false ]; then
    echo -e "  • Stopping any leftover Nav2 / AMCL / RViz processes from previous runs..."
    kill_existing_nav_nodes
    trap cleanup EXIT
    trap 'exit 130' INT TERM
fi

if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
else
    fail_and_abort "/opt/ros/humble/setup.bash not found" ""
fi

if [ -f "${HOME}/turtlebot4_ws/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/turtlebot4_ws/install/setup.bash"
fi

export CYCLONEDDS_URI=/etc/turtlebot4/cyclonedds_rpi.xml
TUNED_FASTDDS="${SCRIPT_DIR}/fastdds_tuning.xml"
if [ -f "${TUNED_FASTDDS}" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="${TUNED_FASTDDS}"
    export FASTDDS_DEFAULT_PROFILES_FILE="${TUNED_FASTDDS}"
    export RMW_FASTRTPS_USE_QOS_FROM_XML=1
elif [ -f /etc/turtlebot4/fastdds_rpi.xml ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE=/etc/turtlebot4/fastdds_rpi.xml
fi
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

# Optional: Attempt non-blocking NTP clock resync with Create 3 base if present
if curl --connect-timeout 0.5 -s -X POST http://192.168.186.2/api/restart-ntpd >/dev/null 2>&1; then
    echo -e "  • Create 3 NTP Sync:  ${GREEN}Triggered successfully${NC}"
fi

VM_IPS=$(hostname -I 2>/dev/null || echo "127.0.0.1")
echo -e "  • ROS_DOMAIN_ID:      ${GREEN}${ROS_DOMAIN_ID}${NC}"
echo -e "  • RMW_IMPLEMENTATION: ${GREEN}${RMW_IMPLEMENTATION}${NC}"
echo -e "  • VM IP Address(es):  ${GREEN}${VM_IPS}${NC}"
echo -e "  • Map File:           ${CYAN}${MAP_FILE}${NC}"
echo -e "  • Nav2 Params File:   ${CYAN}${NAV2_PARAMS_FILE}${NC}"
echo -e "  • Keypoints File:     ${CYAN}${POINTS_FILE}${NC}"
echo -e "  • Step Timeout Limit: ${CYAN}${STEP_TIMEOUT}s (5 minutes)${NC}"

if [ -z "${MAPS_DIR}" ] || [ ! -d "${MAPS_DIR}" ]; then
    fail_and_abort "Maps directory not found (expected Stage2/vm_map_tools/maps next to stage2_system)" ""
fi
for f in "${MAP_FILE}" "${PARAMS_FILE}" "${NAV2_PARAMS_FILE}" "${POINTS_FILE}"; do
    if [ ! -f "${f}" ]; then
        fail_and_abort "Required file not found: ${f}" ""
    fi
done
for f in "${CONTROLLER}" "${SCRIPT_DIR}/keypoints.py" "${SCRIPT_DIR}/protocol.py"; do
    if [ ! -f "${f}" ]; then
        fail_and_abort "Required Stage 2 controller file not found: ${f}" ""
    fi
done
if ! grep -q '"name": *"0"' "${POINTS_FILE}" 2>/dev/null; then
    echo -e "  ${YELLOW}[WARN] No task point \"0\" found in ${POINTS_FILE}. Levels need it to read the task!${NC}"
fi

# ------------------------------------------------------------------------------
# STEP 2: Strict Pre-Flight Robot Diagnostics (/odom & /scan)
# ------------------------------------------------------------------------------
wait_for_robot_topic() {
    local topic="$1"
    local timeout_secs="$2"
    local label="$3"

    local start_time
    start_time=$(date +%s)

    while true; do
        local now
        now=$(date +%s)
        local elapsed=$((now - start_time))

        if [ "${elapsed}" -ge "${timeout_secs}" ]; then
            echo -e "\r  ✘ Waiting for ${label} (${topic})... ${RED}${BOLD}FAILED [TIMEOUT after ${timeout_secs}s]${NC}                      "
            return 1
        fi

        # 1. Check if topic is even listed in the ROS graph yet
        if ! ros2 topic list 2>/dev/null | grep -q "^${topic}$"; then
            printf "\r  ⏳ [%3ds/%ds] Waiting for ${label} (${topic}) to appear in ROS 2 topic list...          " "${elapsed}" "${timeout_secs}"
            sleep 1
            continue
        fi

        # 2. Check publisher count via ros2 daemon
        local pub_count
        pub_count=$(ros2 topic info "${topic}" 2>/dev/null | awk '/Publisher count:/ {print $3}')
        if [ -n "${pub_count}" ] && [ "${pub_count}" -ge 1 ] 2>/dev/null; then
            echo -e "\r  ✔ ${label} (${topic}): ${GREEN}${BOLD}ONLINE (${pub_count} active publisher(s)) [OK]${NC}                         "
            return 0
        fi

        # 3. Fallback: try receiving 1 best_effort message directly
        printf "\r  ⏳ [%3ds/%ds] ${label} (${topic}) in graph, waiting for active publisher / message... " "${elapsed}" "${timeout_secs}"
        if timeout 2 ros2 topic echo --once --qos-reliability best_effort --qos-durability volatile "${topic}" >/dev/null 2>&1; then
            echo -e "\r  ✔ ${label} (${topic}): ${GREEN}${BOLD}ONLINE (received live message) [OK]${NC}                                    "
            return 0
        fi
    done
}

echo -e "\n${BOLD}[2/6] Verifying Robot Hardware Connectivity (Timeout: ${STEP_TIMEOUT}s)...${NC}"

if ! wait_for_robot_topic "/odom" "${STEP_TIMEOUT}" "Create 3 Odometry"; then
    echo -e "${YELLOW}Hint: Ensure the Create 3 base is awake, VMware is Bridged to Wi-Fi (192.168.31.x), and ROS_DOMAIN_ID=0.${NC}"
    fail_and_abort "Robot Odometry (/odom) not available within ${STEP_TIMEOUT}s" ""
fi

if ! wait_for_robot_topic "/scan" "${STEP_TIMEOUT}" "RPLidar Laser Scan"; then
    echo -e "${YELLOW}Hint: Ensure the Raspberry Pi RPLidar node is running on the TurtleBot 4.${NC}"
    fail_and_abort "Robot Laser Scan (/scan) not available within ${STEP_TIMEOUT}s" ""
fi

echo -e "  ${GREEN}[✔] Step 2 Complete: Both /odom and /scan are verified online.${NC}"

if [ "$CHECK_ONLY" = true ]; then
    echo -e "\n${GREEN}[✔] Diagnostic check completed successfully.${NC}"
    exit 0
fi

# ------------------------------------------------------------------------------
# STEP 3: Launch Localization & Wait for Map + AMCL TF Lock
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[3/6] Starting Localization (map_server & amcl) (Timeout: ${STEP_TIMEOUT}s)...${NC}"
echo -e "  • Log file: ${CYAN}${LOG_LOC}${NC}"
: > "${LOG_LOC}"

ros2 launch turtlebot4_navigation localization.launch.py \
    map:="${MAP_FILE}" \
    params:="${PARAMS_FILE}" > "${LOG_LOC}" 2>&1 &
PID_LOC=$!

# Sub-step 3a: Wait for lifecycle_manager_localization to activate map_server and amcl
START_LOC=$(date +%s)
while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_LOC))

    if ! kill -0 "${PID_LOC}" 2>/dev/null; then
        fail_and_abort "localization.launch.py process exited unexpectedly" "${LOG_LOC}"
    fi

    if grep -q "Managed nodes are active" "${LOG_LOC}" 2>/dev/null; then
        echo -e "\r  ✔ [3a] Localization lifecycle: ${GREEN}${BOLD}map_server & amcl are ACTIVE (${ELAPSED}s) [OK]${NC}                            "
        break
    fi

    if [ "${ELAPSED}" -ge "${STEP_TIMEOUT}" ]; then
        echo ""
        fail_and_abort "Localization lifecycle manager did not activate within ${STEP_TIMEOUT}s" "${LOG_LOC}"
    fi

    LOC_STATUS="Starting localization nodes..."
    if grep -q "Activating amcl" "${LOG_LOC}" 2>/dev/null; then
        LOC_STATUS="Activating amcl & setting initial pose..."
    elif grep -q "Activating map_server" "${LOG_LOC}" 2>/dev/null; then
        LOC_STATUS="Activating map_server & publishing /map..."
    elif grep -q "Configuring amcl" "${LOG_LOC}" 2>/dev/null; then
        LOC_STATUS="Configuring amcl parameters & transforms..."
    elif grep -q "Configuring map_server" "${LOG_LOC}" 2>/dev/null; then
        LOC_STATUS="Configuring map_server & loading final_room.pgm..."
    fi

    printf "\r  ⏳ [3a] [%3ds/%ds] %s                                      " "${ELAPSED}" "${STEP_TIMEOUT}" "${LOC_STATUS}"
    sleep 1
done

# Sub-step 3b: Wait for AMCL to discover Create 3 TF (odom -> base_link) and lock onto laser scan
START_TF=$(date +%s)
while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TF))

    if ! kill -0 "${PID_LOC}" 2>/dev/null; then
        fail_and_abort "localization.launch.py crashed while waiting for TF" "${LOG_LOC}"
    fi

    if grep -q "createLaserObject" "${LOG_LOC}" 2>/dev/null; then
        echo -e "\r  ✔ [3b] AMCL TF & Laser Lock: ${GREEN}${BOLD}Connected to odom -> base_link -> rplidar_link (${ELAPSED}s) [OK]${NC}            "
        break
    fi

    if [ "${ELAPSED}" -ge "${STEP_TIMEOUT}" ]; then
        echo ""
        fail_and_abort "AMCL failed to lock onto odom -> base_link -> rplidar_link TF within ${STEP_TIMEOUT}s" "${LOG_LOC}"
    fi

    TF_STATUS="Waiting for first laser scan and TF frames from robot..."
    if grep -q "earlier than all the data in the transform cache" "${LOG_LOC}" 2>/dev/null; then
        TF_STATUS="Received /tf_static; syncing Create 3 odom -> base_link transform cache..."
    elif grep -q "discarding message because the queue is full" "${LOG_LOC}" 2>/dev/null; then
        TF_STATUS="Receiving /scan; waiting to discover Create 3 /tf (odom -> base_link)..."
    fi

    printf "\r  ⏳ [3b] [%3ds/%ds] %s                         " "${ELAPSED}" "${STEP_TIMEOUT}" "${TF_STATUS}"
    sleep 1
done

echo -e "  ${GREEN}[✔] Step 3 Complete: Map server and AMCL localization are fully operational.${NC}"
sleep 2

# ------------------------------------------------------------------------------
# STEP 4: Launch Nav2 Stack & Wait for All Costmaps and Servers to Activate
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[4/6] Starting Nav2 Navigation Stack (Timeout: ${STEP_TIMEOUT}s)...${NC}"
echo -e "  • Log file: ${CYAN}${LOG_NAV}${NC}"
: > "${LOG_NAV}"

ros2 launch turtlebot4_navigation nav2.launch.py \
    params_file:="${NAV2_PARAMS_FILE}" > "${LOG_NAV}" 2>&1 &
PID_NAV=$!

START_NAV=$(date +%s)
while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_NAV))

    if ! kill -0 "${PID_NAV}" 2>/dev/null; then
        fail_and_abort "nav2.launch.py process exited unexpectedly" "${LOG_NAV}"
    fi
    if ! kill -0 "${PID_LOC}" 2>/dev/null; then
        fail_and_abort "Localization process terminated while Nav2 was starting" "${LOG_LOC}"
    fi

    if grep -q "Managed nodes are active" "${LOG_NAV}" 2>/dev/null; then
        echo -e "\r  ✔ [4a] Nav2 Bringup: ${GREEN}${BOLD}All Nav2 servers, local_costmap & global_costmap ACTIVE (${ELAPSED}s) [OK]${NC}             "
        break
    fi

    if [ "${ELAPSED}" -ge "${STEP_TIMEOUT}" ]; then
        echo ""
        fail_and_abort "Nav2 stack did not finish activating within ${STEP_TIMEOUT}s" "${LOG_NAV}"
    fi

    NAV_STATUS="Spawning 8 Nav2 lifecycle nodes..."
    if grep -q "Activating velocity_smoother" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Activating velocity_smoother (final Nav2 server)..."
    elif grep -q "Activating waypoint_follower" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Activating waypoint_follower..."
    elif grep -q "Activating bt_navigator" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Activating bt_navigator (Behavior Tree Navigator)..."
    elif grep -q "Activating behavior_server" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Activating behavior_server (spin, backup, wait)..."
    elif grep -q "Activating planner_server" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="local_costmap ready! Activating planner_server & global_costmap..."
    elif grep -q "\[local_costmap.local_costmap\]: start" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Nav2 local_costmap started! Connecting controller_server bond..."
    elif grep -q "not part of the same tree" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Waiting for Nav2 local_costmap: syncing odom -> base_link TF with Create 3..."
    elif grep -q "Invalid frame ID \"odom\"" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Waiting for Nav2 local_costmap: discovering odom frame via DDS..."
    elif grep -q "Activating controller_server" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Activating controller_server & checking local_costmap TF..."
    elif grep -q "Configuring velocity_smoother" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Configuring velocity_smoother..."
    elif grep -q "Configuring waypoint_follower" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Configuring waypoint_follower..."
    elif grep -q "Configuring bt_navigator" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Configuring bt_navigator..."
    elif grep -q "Configuring behavior_server" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Configuring behavior_server..."
    elif grep -q "Configuring planner_server" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Configuring planner_server & global_costmap (static + obstacle + inflation)..."
    elif grep -q "Configuring smoother_server" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Configuring smoother_server..."
    elif grep -q "Configuring controller_server" "${LOG_NAV}" 2>/dev/null; then
        NAV_STATUS="Configuring controller_server & local_costmap plugins..."
    fi

    printf "\r  ⏳ [4a] [%3ds/%ds] %-78s" "${ELAPSED}" "${STEP_TIMEOUT}" "${NAV_STATUS}"
    sleep 1
done

echo -e "  ${GREEN}[✔] Step 4 Complete: Nav2 navigation stack is 100% active.${NC}"

# ------------------------------------------------------------------------------
# STEP 5: Launch RViz2 (always — required to set the initial pose)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[5/6] Launching RViz2 (Software-Rendering mode) (Timeout: ${STEP_TIMEOUT}s)...${NC}"
echo -e "  • Log file: ${CYAN}${LOG_RVIZ}${NC}"
: > "${LOG_RVIZ}"

export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch turtlebot4_viz view_robot.launch.py > "${LOG_RVIZ}" 2>&1 &
PID_RVIZ=$!

START_RVIZ=$(date +%s)
while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_RVIZ))

    if ! kill -0 "${PID_RVIZ}" 2>/dev/null; then
        fail_and_abort "RViz2 process exited unexpectedly" "${LOG_RVIZ}"
    fi
    check_nav_stack_alive

    if grep -q "OpenGl version" "${LOG_RVIZ}" 2>/dev/null; then
        echo -e "\r  ✔ [5a] RViz2 Window Initialized (PID ${PID_RVIZ}, ${ELAPSED}s) ${GREEN}${BOLD}[OK]${NC}                                    "
        break
    fi

    if [ "${ELAPSED}" -ge "${STEP_TIMEOUT}" ]; then
        echo ""
        fail_and_abort "RViz2 did not initialize within ${STEP_TIMEOUT}s" "${LOG_RVIZ}"
    fi

    printf "\r  ⏳ [5a] [%3ds/%ds] Waiting for RViz2 OpenGL window to initialize...                " "${ELAPSED}" "${STEP_TIMEOUT}"
    sleep 1
done
echo -e "  ${GREEN}[✔] Step 5 Complete: RViz2 is running.${NC}"

# ------------------------------------------------------------------------------
# STEP 6: Manual Initial Pose (2D Pose Estimate in RViz) + Confirmation
# ------------------------------------------------------------------------------
read_amcl_pose() {
    # Prints "x y yaw_deg" of the latest /amcl_pose (transient_local), or nothing on timeout.
    timeout 4 ros2 topic echo --once \
        --qos-durability transient_local --qos-reliability reliable \
        /amcl_pose 2>/dev/null | python3 -c '
import math, sys
try:
    import yaml
    d = yaml.safe_load(sys.stdin.read())
    p = d["pose"]["pose"]["position"]; q = d["pose"]["pose"]["orientation"]
    yaw = math.degrees(math.atan2(2*(q["w"]*q["z"] + q["x"]*q["y"]), 1 - 2*(q["y"]**2 + q["z"]**2)))
    print(f"{p[\"x\"]:.2f} {p[\"y\"]:.2f} {yaw:.1f}")
except Exception:
    pass
'
}

echo -e "\n${BOLD}[6/6] Set the Robot's Initial Pose in RViz${NC}"
echo -e "  1. In RViz, click ${BOLD}2D Pose Estimate${NC} on the top toolbar."
echo -e "  2. Click the robot's real position on the map and drag in the direction it faces."
echo -e "  3. Check that the laser scan (red dots) lines up with the map walls."
echo -e "  4. Come back to this terminal and press ${BOLD}Enter${NC} to confirm."

while true; do
    echo ""
    read -r -p "  ➜ Press Enter once the initial pose is set in RViz (or type 'q' to abort): " ANSWER < /dev/tty
    if [ "${ANSWER}" = "q" ] || [ "${ANSWER}" = "Q" ]; then
        fail_and_abort "Bringup aborted by user at initial pose confirmation" ""
    fi

    check_nav_stack_alive

    POSE=$(read_amcl_pose)
    if [ -n "${POSE}" ]; then
        read -r PX PY PYAW <<< "${POSE}"
        echo -e "  • AMCL pose: ${CYAN}x=${PX} m, y=${PY} m, yaw=${PYAW}°${NC}"
    else
        echo -e "  ${YELLOW}• Could not read /amcl_pose (AMCL publishes it after a pose estimate or movement).${NC}"
    fi

    read -r -p "  ➜ Is the robot correctly localized on the map? [y/N]: " CONFIRM < /dev/tty
    case "${CONFIRM}" in
        y|Y|yes|YES|Yes)
            echo -e "  ${GREEN}[✔] Step 6 Complete: Initial pose confirmed by user.${NC}"
            break
            ;;
        *)
            echo -e "  ${YELLOW}Okay — set the pose again with 2D Pose Estimate in RViz, then press Enter.${NC}"
            ;;
    esac
done

FIRST_IP=$(echo "${VM_IPS}" | awk '{print $1}')
echo -e "\n${GREEN}${BOLD}=====================================================================${NC}"
echo -e "${GREEN}${BOLD}  ALL 6 STEPS SUCCEEDED — ROS 2 Navigation Stack is READY!           ${NC}"
echo -e "${GREEN}${BOLD}=====================================================================${NC}"
echo -e "  • VM IP Address(es): ${CYAN}${VM_IPS}${NC}"
echo -e "  • Logs:              ${CYAN}${LOG_DIR}/${NC}"
echo -e ""
echo -e "${YELLOW}${BOLD}Next step 1 — Open a SEPARATE VM terminal and run the Python controller:${NC}"
echo -e "  ${BOLD}cd ${SCRIPT_DIR}${NC}"
echo -e "  ${BOLD}./start_robot_controller.sh --port ${CTRL_PORT}${NC}"
echo -e "  ${CYAN}(You can press Ctrl+C and re-run ./start_robot_controller.sh anytime${NC}"
echo -e "  ${CYAN} without restarting Localization, Nav2, or RViz!)${NC}"
echo -e ""
echo -e "${YELLOW}${BOLD}Next step 2 — On the Windows PC:${NC}"
echo -e "  ${BOLD}conda activate phienv${NC}"
echo -e "  ${BOLD}cd Stage2\\stage2_system${NC}"
echo -e "  ${BOLD}python laptop\\main.py --vm auto${NC}      (or: --vm ${FIRST_IP} --vm-port ${CTRL_PORT})"
echo -e ""
echo -e "Keep this terminal open. Press ${BOLD}Ctrl+C${NC} here only when you want to stop Nav2 & RViz."
echo -e "=====================================================================\n"

# Monitor background navigation processes continuously
while true; do
    if [ -n "${PID_LOC}" ] && ! kill -0 "${PID_LOC}" 2>/dev/null; then
        fail_and_abort "Localization process terminated unexpectedly during runtime" "${LOG_LOC}"
    fi
    if [ -n "${PID_NAV}" ] && ! kill -0 "${PID_NAV}" 2>/dev/null; then
        fail_and_abort "Nav2 process terminated unexpectedly during runtime" "${LOG_NAV}"
    fi
    if [ -n "${PID_RVIZ}" ] && ! kill -0 "${PID_RVIZ}" 2>/dev/null; then
        echo -e "\n${YELLOW}[WARN] RViz2 was closed. Navigation keeps running; restart RViz manually if you need to re-set the pose:${NC}"
        echo -e "${YELLOW}       LIBGL_ALWAYS_SOFTWARE=1 ros2 launch turtlebot4_viz view_robot.launch.py${NC}"
        PID_RVIZ=""
    fi
    sleep 2
done
