#!/usr/bin/env bash
# ==============================================================================
# TurtleBot 4 Stage 2: Hardware & ROS 2 Health Diagnostic Tool
# ==============================================================================
# Quickly diagnoses the Raspberry Pi, RPLiDAR, OAK-D Camera, Create 3 Base,
# FastDDS shared memory, and /odom transform status.
# ==============================================================================

set -o pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
NC="\033[0m"

echo -e "${CYAN}${BOLD}=====================================================================${NC}"
echo -e "${CYAN}${BOLD}       TurtleBot 4 Stage 2: Comprehensive Hardware Diagnostics        ${NC}"
echo -e "${CYAN}${BOLD}=====================================================================${NC}\n"

# 1. ROS 2 Environment Setup
if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
else
    echo -e "${RED}[FAIL] ROS 2 Humble not found in /opt/ros/humble${NC}"
    exit 1
fi

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNED_FASTDDS="${SCRIPT_DIR}/fastdds_tuning.xml"
if [ -f "${TUNED_FASTDDS}" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="${TUNED_FASTDDS}"
    export FASTDDS_DEFAULT_PROFILES_FILE="${TUNED_FASTDDS}"
    export RMW_FASTRTPS_USE_QOS_FROM_XML=1
fi

# Clean stale FastDDS cache if requested or needed
if [ "$1" = "--clean-cache" ] || [ "$1" = "-c" ]; then
    echo -e "${YELLOW}[!] Flushing FastDDS shared memory and restarting ROS 2 daemon...${NC}"
    rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true
    ros2 daemon stop >/dev/null 2>&1 || true
    ros2 daemon start >/dev/null 2>&1 || true
    echo -e "${GREEN}[✔] Cache flushed.${NC}\n"
fi

# 2. Network Check
echo -e "${BOLD}[1/5] Network & Companion Computer Connectivity:${NC}"
RPI_IP="192.168.31.242"
VM_IPS=$(hostname -I 2>/dev/null || echo "127.0.0.1")
echo -e "  • VM IP Address:      ${CYAN}${VM_IPS}${NC}"
echo -e "  • ROS_DOMAIN_ID:      ${CYAN}${ROS_DOMAIN_ID}${NC}"
echo -e "  • RMW_IMPLEMENTATION: ${CYAN}${RMW_IMPLEMENTATION}${NC}"

if ping -c 1 -W 1 "${RPI_IP}" >/dev/null 2>&1; then
    echo -e "  • Raspberry Pi (${RPI_IP}): ${GREEN}${BOLD}REACHABLE [OK]${NC}"
else
    echo -e "  • Raspberry Pi (${RPI_IP}): ${YELLOW}NO PING RESPONSE (may be normal if ICMP blocked)${NC}"
fi

# 3. Raspberry Pi ROS 2 Nodes
echo -e "\n${BOLD}[2/5] Active ROS 2 Nodes in Graph:${NC}"
NODES=$(ros2 node list 2>/dev/null || true)
if [ -z "${NODES}" ]; then
    echo -e "  ${RED}[FAIL] No ROS 2 nodes detected! Check Wi-Fi and ROS_DOMAIN_ID.${NC}"
else
    NODE_COUNT=$(echo "${NODES}" | wc -l)
    echo -e "  • Discovered ${GREEN}${NODE_COUNT} active ROS node(s)${NC}:"
    while IFS= read -r nd; do
        [ -n "$nd" ] && echo -e "    - ${CYAN}${nd}${NC}"
    done <<< "${NODES}"
fi

# 4. Sensor Feeds (LiDAR & Camera)
echo -e "\n${BOLD}[3/5] Perception Sensor Feeds (Raspberry Pi):${NC}"
LIDAR_PUBS=$(ros2 topic info /scan 2>/dev/null | awk '/Publisher count:/ {print $3}')
if [ -n "${LIDAR_PUBS}" ] && [ "${LIDAR_PUBS}" -ge 1 ] 2>/dev/null; then
    echo -e "  • RPLiDAR Laser Scan (/scan):        ${GREEN}${BOLD}ONLINE (${LIDAR_PUBS} publisher) [OK]${NC}"
else
    echo -e "  • RPLiDAR Laser Scan (/scan):        ${RED}${BOLD}OFFLINE (0 publishers) [FAIL]${NC}"
fi

CAM_PUBS=$(ros2 topic info /oakd/rgb/preview/image_raw 2>/dev/null | awk '/Publisher count:/ {print $3}')
if [ -n "${CAM_PUBS}" ] && [ "${CAM_PUBS}" -ge 1 ] 2>/dev/null; then
    echo -e "  • OAK-D RGB Camera (/oakd/...):       ${GREEN}${BOLD}ONLINE (${CAM_PUBS} publisher) [OK]${NC}"
else
    echo -e "  • OAK-D RGB Camera (/oakd/...):       ${YELLOW}OFFLINE / Idle${NC}"
fi

# 5. Create 3 Mobile Base Health & Odometry
echo -e "\n${BOLD}[4/5] iRobot Create 3 Mobile Base & Odometry Health:${NC}"
WHEEL_PUBS=$(ros2 topic info /wheel_status 2>/dev/null | awk '/Publisher count:/ {print $3}')
ODOM_PUBS=$(ros2 topic info /odom 2>/dev/null | awk '/Publisher count:/ {print $3}')

if [ -n "${ODOM_PUBS}" ] && [ "${ODOM_PUBS}" -ge 1 ] 2>/dev/null; then
    echo -e "  • Mobile Base Odometry (/odom):      ${GREEN}${BOLD}ONLINE (${ODOM_PUBS} publisher) [OK]${NC}"
else
    echo -e "  • Mobile Base Odometry (/odom):      ${RED}${BOLD}OFFLINE / MISSING (0 publishers) [FAIL]${NC}"
fi

if [ -n "${WHEEL_PUBS}" ] && [ "${WHEEL_PUBS}" -ge 1 ] 2>/dev/null; then
    echo -e "  • Base Motor Telemetry (/wheel_status): ${GREEN}${BOLD}ONLINE (${WHEEL_PUBS} publisher) [OK]${NC}"
    BASE_ONLINE=true
else
    echo -e "  • Base Motor Telemetry (/wheel_status): ${RED}${BOLD}OFFLINE (0 publishers) [FAIL]${NC}"
    BASE_ONLINE=false
fi

# 6. Diagnostics Aggregator Inspection
echo -e "\n${BOLD}[5/5] Subsystem Diagnostics & Battery Status:${NC}"
DIAG_SAMPLE=$(timeout 3 ros2 topic echo --once /diagnostics_agg 2>/dev/null || true)

if [ -n "${DIAG_SAMPLE}" ]; then
    BATT_PCT=$(echo "${DIAG_SAMPLE}" | awk '/key: Battery Percentage/ {getline; print $2}' | tr -d "'" | head -n 1)
    BATT_VOLT=$(echo "${DIAG_SAMPLE}" | awk '/key: Battery Voltage/ {getline; print $2}' | tr -d "'" | head -n 1)
    WHEEL_STAT=$(echo "${DIAG_SAMPLE}" | awk '/key: Wheel Status/ {getline; print $2}' | tr -d "'" | head -n 1)
    CREATE_STALE=$(echo "${DIAG_SAMPLE}" | grep -A 2 "/Turtlebot4/Create3" | grep "message: Stale" || true)

    if [ -n "${BATT_PCT}" ]; then
        PCT_NUM=$(awk -v p="${BATT_PCT}" 'BEGIN {printf "%.0f", p*100}')
        if [ "${PCT_NUM}" -lt 20 ]; then
            echo -e "  • Battery Percentage: ${RED}${BOLD}${PCT_NUM}% (CRITICAL: Low Battery! Recharge now!)${NC}"
        elif [ "${PCT_NUM}" -lt 30 ]; then
            echo -e "  • Battery Percentage: ${YELLOW}${BOLD}${PCT_NUM}% (Low Battery Warning)${NC}"
        else
            echo -e "  • Battery Percentage: ${GREEN}${BOLD}${PCT_NUM}% [OK]${NC}"
        fi
    fi

    if [ -n "${BATT_VOLT}" ]; then
        echo -e "  • Battery Voltage:    ${CYAN}${BATT_VOLT} V${NC}"
    fi

    if [ -n "${WHEEL_STAT}" ]; then
        if [ "${WHEEL_STAT}" = "False" ]; then
            echo -e "  • Wheel Driver State: ${RED}${BOLD}Disabled (Emergency stop or low-voltage safety cutoff)${NC}"
        else
            echo -e "  • Wheel Driver State: ${GREEN}Enabled [OK]${NC}"
        fi
    fi

    if [ -n "${CREATE_STALE}" ]; then
        echo -e "  • Base Link Status:   ${RED}${BOLD}STALE (Raspberry Pi lost communication with Create 3 base)${NC}"
    fi
else
    echo -e "  ${YELLOW}No /diagnostics_agg messages received within 3s.${NC}"
fi

echo -e "\n${BOLD}=====================================================================${NC}"
if [ "${BASE_ONLINE}" = true ] && [ -n "${ODOM_PUBS}" ] && [ "${ODOM_PUBS}" -ge 1 ] 2>/dev/null; then
    echo -e "${GREEN}${BOLD}  RESULT: ALL SYSTEMS OPERATIONAL! Ready for ./auto_bringup.sh       ${NC}"
else
    echo -e "${RED}${BOLD}  RESULT: CREATE 3 BASE IS NOT RESPONDING TO ROS 2                   ${NC}"
    echo -e "${YELLOW}  Required Recovery Actions:${NC}"
    echo -e "    1. Lift the robot off the charging dock onto the floor."
    echo -e "    2. Press & hold the ${BOLD}center Power button${NC} on the Create 3 base for ${BOLD}10-15s${NC}"
    echo -e "       until all lights go completely dark (shutdown chime plays)."
    echo -e "    3. Push the robot ${BOLD}back onto the charging dock contacts${NC}."
    echo -e "    4. Wait for the 4-note ascending boot chime."
    echo -e "    5. Re-run: ${CYAN}./check_robot_health.sh${NC}"
fi
echo -e "${BOLD}=====================================================================\n"${NC}

