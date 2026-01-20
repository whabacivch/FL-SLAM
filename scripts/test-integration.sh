#!/usr/bin/env bash
# FL-SLAM Integration Test Suite
# Full end-to-end test with rosbag replay, Foxglove visualization, and SLAM validation.
# Longer execution (~90 seconds) for comprehensive system validation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
BAG_PATH="${BAG_PATH:-${PROJECT_DIR}/rosbags/tb3_slam3d_small_ros2}"
TIMEOUT_SEC="${TIMEOUT_SEC:-90}"
STARTUP_SEC="${STARTUP_SEC:-20}"
REQUIRE_LOOP="${REQUIRE_LOOP:-1}"
REQUIRE_SLAM_ACTIVE="${REQUIRE_SLAM_ACTIVE:-1}"
ENABLE_FOXGLOVE="${ENABLE_FOXGLOVE:-1}"

echo "=========================================="
echo "FL-SLAM Integration Test Suite"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Bag:          ${BAG_PATH}"
echo "  Timeout:      ${TIMEOUT_SEC}s"
echo "  Startup wait: ${STARTUP_SEC}s"
echo "  Require loop: ${REQUIRE_LOOP}"
echo "  SLAM active:  ${REQUIRE_SLAM_ACTIVE}"
if [[ "${ENABLE_FOXGLOVE}" -eq 1 ]]; then
    echo "  Foxglove:     ws://localhost:8765"
fi
echo ""

# Detect environment
if [ -f /.dockerenv ]; then
    echo "Environment: Docker container"
    ROS_SETUP="/opt/ros/jazzy/setup.bash"
    WS_ROOT="/ros2_ws"
    IN_DOCKER=1
else
    echo "Environment: Native"
    ROS_SETUP="/opt/ros/jazzy/setup.bash"
    WS_ROOT="${PROJECT_DIR}/fl_ws"
    IN_DOCKER=0
fi

# Check if bag exists
if [ ! -d "${BAG_PATH}" ]; then
    echo "ERROR: Rosbag not found at ${BAG_PATH}" >&2
    echo "" >&2
    echo "Download the test bag with:" >&2
    echo "  ./scripts/download_tb3_rosbag.sh" >&2
    exit 1
fi

# Source ROS environment
if [ ! -f "${ROS_SETUP}" ]; then
    echo "ERROR: ROS 2 Jazzy not found at ${ROS_SETUP}" >&2
    exit 1
fi

set +u
source "${ROS_SETUP}"
set -u

# Build workspace if needed
INSTALL_SETUP="${WS_ROOT}/install/setup.bash"
if [ ! -f "${INSTALL_SETUP}" ]; then
    echo "Building fl_slam_poc package..."
    (cd "${WS_ROOT}" && colcon build --symlink-install --packages-select fl_slam_poc)
fi

set +u
source "${INSTALL_SETUP}"
set -u

# Configure ROS environment
export ROS_HOME="${ROS_HOME:-${PROJECT_DIR}/.ros}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${PROJECT_DIR}/.ros/log}"
export RMW_FASTRTPS_USE_SHM="${RMW_FASTRTPS_USE_SHM:-0}"
mkdir -p "${ROS_LOG_DIR}"

# Create log directory
LOG_DIR="${PROJECT_DIR}/diagnostic_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/integration_test_${TS}.log"

echo "=========================================="
echo "Starting Integration Test"
echo "=========================================="
echo "Log: ${RUN_LOG}"
echo ""

# Launch the system with rosbag
(
    timeout "${TIMEOUT_SEC}" ros2 launch fl_slam_poc poc_tb3_rosbag.launch.py \
        play_bag:=true \
        bag:="${BAG_PATH}" \
        use_sim_time:=true \
        bag_start_delay_sec:=3.0 \
        enable_foxglove:="${ENABLE_FOXGLOVE}" \
        enable_decompress:=true \
        enable_image:=true \
        enable_depth:=true \
        enable_camera_info:=true \
        publish_rgbd_evidence:=true \
        rgbd_publish_every_n_scans:=5 \
        rgbd_max_points_per_msg:=500 \
        sensor_qos_reliability:=both
) 2>&1 | tee "${RUN_LOG}" &

LAUNCH_PID=$!

echo "Waiting ${STARTUP_SEC}s for system startup..."
sleep "${STARTUP_SEC}"

echo ""
echo "=========================================="
echo "System Validation Checks"
echo "=========================================="
echo ""

# Initialize check results
anchor_ok=0
loop_ok=0
backend_ok=0
backend_mode=""

# Check 1: Anchor creation
echo "Check 1: Anchor creation"
if timeout 5s ros2 topic echo -n 1 /sim/anchor_create >/dev/null 2>&1; then
    anchor_ok=1
    echo "  ✓ Detected /sim/anchor_create topic"
else
    # Fallback: check log for anchor creation messages
    if grep -qE "Created anchor|Backend received anchor" "${RUN_LOG}"; then
        anchor_ok=1
        echo "  ✓ Detected anchor creation in logs"
    else
        echo "  ✗ No anchor creation detected"
    fi
fi

# Check 2: Loop factor detection
echo "Check 2: Loop closure detection"
if timeout 5s ros2 topic echo -n 1 /sim/loop_factor >/dev/null 2>&1; then
    loop_ok=1
    echo "  ✓ Detected /sim/loop_factor topic"
else
    # Fallback: check log for loop factors
    if grep -qE "Published loop factor|Backend received loop factor" "${RUN_LOG}"; then
        loop_ok=1
        echo "  ✓ Detected loop factor in logs"
    else
        echo "  ✗ No loop factor detected"
    fi
fi

# Check 3: Backend status
echo "Check 3: Backend status"
backend_json="$(timeout 5s ros2 topic echo -n 1 /cdwm/backend_status 2>/dev/null || true)"
if [[ -n "${backend_json}" ]]; then
    backend_mode="$(
        python3 - <<'PY' "${backend_json}" 2>/dev/null || true
import json, sys
raw = sys.argv[1].strip() if len(sys.argv) > 1 else ""
# ros2 topic echo prints "data: '<json>'" sometimes
if raw.startswith("data:"):
    raw = raw.split("data:", 1)[1].strip()
if raw.startswith("'") and raw.endswith("'"):
    raw = raw[1:-1]
try:
    obj = json.loads(raw)
    print(obj.get("mode", ""))
except:
    print("")
PY
    )"
    if [[ -n "${backend_mode}" ]]; then
        backend_ok=1
        echo "  ✓ Backend status: ${backend_mode}"
    fi
fi

# Fallback: check log for backend status
if [[ "${backend_ok}" -ne 1 ]]; then
    if grep -q "Backend status: mode=" "${RUN_LOG}"; then
        backend_ok=1
        backend_mode="$(grep "Backend status: mode=" "${RUN_LOG}" | tail -n 1 | sed -E 's/.*mode=([^, ]+).*/\1/')"
        echo "  ✓ Backend status from log: ${backend_mode}"
    else
        echo "  ✗ No backend status detected"
    fi
fi

# Stop the launch process
kill "${LAUNCH_PID}" 2>/dev/null || true
wait "${LAUNCH_PID}" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Test Results"
echo "=========================================="
echo ""

# Evaluate results
fail=0

if [[ "${anchor_ok}" -ne 1 ]]; then
    echo "✗ FAIL: No anchor creation observed"
    fail=1
else
    echo "✓ PASS: Anchor creation detected"
fi

if [[ "${REQUIRE_LOOP}" -eq 1 && "${loop_ok}" -ne 1 ]]; then
    echo "✗ FAIL: No loop closure observed (required)"
    fail=1
elif [[ "${loop_ok}" -eq 1 ]]; then
    echo "✓ PASS: Loop closure detected"
else
    echo "⊘ SKIP: Loop closure not required"
fi

if [[ "${backend_ok}" -ne 1 ]]; then
    echo "✗ FAIL: No backend status observed"
    fail=1
else
    echo "✓ PASS: Backend running (mode: ${backend_mode})"
fi

if [[ "${REQUIRE_SLAM_ACTIVE}" -eq 1 && "${backend_mode}" != "SLAM_ACTIVE" ]]; then
    echo "✗ FAIL: Backend mode is not SLAM_ACTIVE (got '${backend_mode}')"
    fail=1
elif [[ "${backend_mode}" == "SLAM_ACTIVE" ]]; then
    echo "✓ PASS: Backend in SLAM_ACTIVE mode"
fi

echo ""
if [[ "${fail}" -eq 0 ]]; then
    echo "=========================================="
    echo "✓ ALL INTEGRATION TESTS PASSED"
    echo "=========================================="
    echo ""
    echo "Log saved to: ${RUN_LOG}"
    echo ""
else
    echo "=========================================="
    echo "✗ INTEGRATION TESTS FAILED"
    echo "=========================================="
    echo ""
    echo "Review the log for details: ${RUN_LOG}"
    echo ""
    exit 1
fi
