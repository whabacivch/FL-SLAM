#!/usr/bin/env bash
# 3D Point Cloud Integration Test for FL-SLAM
#
# Tests FL-SLAM in 3D point cloud mode with GPU acceleration.
# Uses r2b_storage dataset or any PointCloud2 rosbag.
#
# Usage:
#   ./scripts/test-3d-integration.sh
#   BAG_PATH=/path/to/bag ./scripts/test-3d-integration.sh
#   USE_GPU=false ./scripts/test-3d-integration.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
BAG_PATH="${BAG_PATH:-${PROJECT_DIR}/rosbags/r2b_storage}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
STARTUP_SEC="${STARTUP_SEC:-30}"
USE_GPU="${USE_GPU:-true}"
VOXEL_SIZE="${VOXEL_SIZE:-0.05}"
ENABLE_FOXGLOVE="${ENABLE_FOXGLOVE:-1}"

# Topic configuration (override for different datasets)
POINTCLOUD_TOPIC="${POINTCLOUD_TOPIC:-/lidar/points}"
ODOM_TOPIC="${ODOM_TOPIC:-/odom}"
CAMERA_TOPIC="${CAMERA_TOPIC:-/camera/color/image_raw}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/depth/image_raw}"

# Test requirements
REQUIRE_POINTCLOUD="${REQUIRE_POINTCLOUD:-1}"
REQUIRE_ANCHOR="${REQUIRE_ANCHOR:-1}"
REQUIRE_LOOP="${REQUIRE_LOOP:-0}"  # r2b may not have loop closures
REQUIRE_SLAM_ACTIVE="${REQUIRE_SLAM_ACTIVE:-0}"

echo "============================================"
echo "FL-SLAM 3D Point Cloud Integration Test"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Bag:            $BAG_PATH"
echo "  Timeout:        ${TIMEOUT_SEC}s"
echo "  Startup delay:  ${STARTUP_SEC}s"
echo "  GPU mode:       $USE_GPU"
echo "  Voxel size:     ${VOXEL_SIZE}m"
echo "  Foxglove:       ${ENABLE_FOXGLOVE}"
echo ""
echo "Topics:"
echo "  PointCloud:     ${POINTCLOUD_TOPIC}"
echo "  Odometry:       ${ODOM_TOPIC}"
echo "  Camera:         ${CAMERA_TOPIC}"
echo "  Depth:          ${DEPTH_TOPIC}"
echo ""
echo "Requirements:"
echo "  Point cloud:    $([ "$REQUIRE_POINTCLOUD" = "1" ] && echo "REQUIRED" || echo "optional")"
echo "  Anchors:        $([ "$REQUIRE_ANCHOR" = "1" ] && echo "REQUIRED" || echo "optional")"
echo "  Loop closures:  $([ "$REQUIRE_LOOP" = "1" ] && echo "REQUIRED" || echo "optional")"
echo "  SLAM_ACTIVE:    $([ "$REQUIRE_SLAM_ACTIVE" = "1" ] && echo "REQUIRED" || echo "optional")"
echo ""

# Check bag exists
if [[ ! -d "$BAG_PATH" && ! -f "$BAG_PATH" ]]; then
  echo "ERROR: Rosbag not found at $BAG_PATH"
  echo ""
  echo "To download the r2b dataset:"
  echo "  ./scripts/download_r2b_dataset.sh"
  echo ""
  echo "Or specify a different bag:"
  echo "  BAG_PATH=/path/to/your/bag ./scripts/test-3d-integration.sh"
  exit 1
fi

# Source ROS
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
fi

# Source workspace
INSTALL_SETUP="${PROJECT_DIR}/fl_ws/install/setup.bash"
if [[ -f "$INSTALL_SETUP" ]]; then
  set +u
  source "$INSTALL_SETUP"
  set -u
else
  echo "ERROR: Workspace not built. Run:"
  echo "  cd fl_ws && colcon build --symlink-install"
  exit 1
fi

# Cleanup function
cleanup() {
  echo ""
  echo "Cleaning up..."
  
  # Kill launch process
  if [[ -n "${LAUNCH_PID:-}" ]]; then
    kill -INT "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
  
  # Kill any remaining nodes
  pkill -f "fl_frontend" 2>/dev/null || true
  pkill -f "fl_backend" 2>/dev/null || true
  pkill -f "foxglove_bridge" 2>/dev/null || true
  
  sleep 2
  
  echo "Cleanup complete."
}
trap cleanup EXIT

echo "============================================"
echo "Launching FL-SLAM (3D Point Cloud Mode)"
echo "============================================"
echo ""

# Build launch command
LAUNCH_ARGS=(
  "bag:=${BAG_PATH}"
  "play_bag:=true"
  "use_3d_pointcloud:=true"
  "use_gpu:=${USE_GPU}"
  "voxel_size:=${VOXEL_SIZE}"
  "enable_foxglove:=$([ "$ENABLE_FOXGLOVE" = "1" ] && echo "true" || echo "false")"
  "pointcloud_topic:=${POINTCLOUD_TOPIC}"
  "odom_topic:=${ODOM_TOPIC}"
  "camera_topic:=${CAMERA_TOPIC}"
  "depth_topic:=${DEPTH_TOPIC}"
)

ros2 launch fl_slam_poc poc_3d_rosbag.launch.py "${LAUNCH_ARGS[@]}" &
LAUNCH_PID=$!

echo "Launch PID: $LAUNCH_PID"
echo "Waiting ${STARTUP_SEC}s for system startup..."
sleep "$STARTUP_SEC"

# Check if launch is still running
if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
  echo "ERROR: Launch process died during startup"
  exit 1
fi

echo ""
echo "============================================"
echo "Monitoring Test Progress"
echo "============================================"
echo ""

# Initialize counters
ANCHOR_COUNT=0
LOOP_COUNT=0
POINTCLOUD_RECEIVED=0
ODOM_RECEIVED=0
SLAM_ACTIVE=0

echo "Checking topic availability..."
echo ""

# List available topics for debugging
echo "Available topics:"
ros2 topic list 2>/dev/null | head -20 || echo "  (none yet)"
echo ""

# Monitor for test duration
START_TIME=$(date +%s)
while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))
  
  if [[ $ELAPSED -ge $TIMEOUT_SEC ]]; then
    echo "Timeout reached (${TIMEOUT_SEC}s)"
    break
  fi
  
  # Check launch process
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "Launch process ended"
    break
  fi
  
  # Check if point cloud topic has data (only check once)
  if [[ "$POINTCLOUD_RECEIVED" -eq 0 ]]; then
    PC_HZ=$(ros2 topic hz "${POINTCLOUD_TOPIC}" --window 3 2>/dev/null | grep "average rate" | head -1 || echo "")
    if [[ -n "$PC_HZ" ]]; then
      POINTCLOUD_RECEIVED=1
      echo "[${ELAPSED}s] PointCloud2 data flowing on ${POINTCLOUD_TOPIC}: $PC_HZ"
    fi
  fi
  
  # Check if odom topic has data (only check once)
  if [[ "$ODOM_RECEIVED" -eq 0 ]]; then
    ODOM_HZ=$(ros2 topic hz "${ODOM_TOPIC}" --window 3 2>/dev/null | grep "average rate" | head -1 || echo "")
    if [[ -n "$ODOM_HZ" ]]; then
      ODOM_RECEIVED=1
      echo "[${ELAPSED}s] Odometry data flowing on ${ODOM_TOPIC}: $ODOM_HZ"
    fi
  fi
  
  # Check frontend status for point cloud processing
  FRONTEND_STATUS=$(ros2 topic echo /cdwm/frontend_status --once --no-arr 2>/dev/null || echo "")
  if [[ -n "$FRONTEND_STATUS" ]]; then
    echo "[${ELAPSED}s] Frontend status: active"
  fi
  
  # Count anchor messages
  NEW_ANCHORS=$(ros2 topic echo /sim/anchor_create --once --no-arr 2>/dev/null | grep -c "anchor_id" || echo "0")
  if [[ "$NEW_ANCHORS" -gt 0 ]]; then
    ANCHOR_COUNT=$((ANCHOR_COUNT + NEW_ANCHORS))
    echo "[${ELAPSED}s] Anchor detected (total: $ANCHOR_COUNT)"
  fi
  
  # Count loop messages
  NEW_LOOPS=$(ros2 topic echo /sim/loop_factor --once --no-arr 2>/dev/null | grep -c "from_anchor_id" || echo "0")
  if [[ "$NEW_LOOPS" -gt 0 ]]; then
    LOOP_COUNT=$((LOOP_COUNT + NEW_LOOPS))
    echo "[${ELAPSED}s] Loop factor detected (total: $LOOP_COUNT)"
  fi
  
  # Check backend status for SLAM_ACTIVE
  STATUS=$(ros2 topic echo /cdwm/backend_status --once --no-arr 2>/dev/null || echo "")
  if echo "$STATUS" | grep -q "SLAM_ACTIVE"; then
    SLAM_ACTIVE=1
    echo "[${ELAPSED}s] Backend in SLAM_ACTIVE mode"
  fi
  
  sleep 5
done

echo ""
echo "============================================"
echo "Test Results"
echo "============================================"
echo ""
echo "Data Flow:"
echo "  PointCloud received: $([ "$POINTCLOUD_RECEIVED" = "1" ] && echo "YES" || echo "NO")"
echo "  Odometry received:   $([ "$ODOM_RECEIVED" = "1" ] && echo "YES" || echo "NO")"
echo ""
echo "SLAM Status:"
echo "  Anchors created:     $ANCHOR_COUNT"
echo "  Loop factors:        $LOOP_COUNT"
echo "  SLAM_ACTIVE mode:    $([ "$SLAM_ACTIVE" = "1" ] && echo "YES" || echo "NO")"
echo ""

# Evaluate results
PASS=1

# Check data flow first
if [[ "$REQUIRE_POINTCLOUD" = "1" && "$POINTCLOUD_RECEIVED" -eq 0 ]]; then
  echo "FAIL: No point cloud data received"
  echo "  Check: Is the rosbag playing? Are topic names correct?"
  echo "  Expected topic: ${POINTCLOUD_TOPIC}"
  echo ""
  echo "  To see available topics in the bag:"
  echo "    ros2 bag info $BAG_PATH"
  echo ""
  echo "  To override topic name:"
  echo "    POINTCLOUD_TOPIC=/your/topic ./scripts/test-3d-integration.sh"
  PASS=0
fi

if [[ "$ODOM_RECEIVED" -eq 0 ]]; then
  echo "WARNING: No odometry data received on ${ODOM_TOPIC}"
  echo "  The rosbag may use a different odometry topic."
  echo ""
  echo "  To see available topics: ros2 bag info $BAG_PATH"
  echo "  To override: ODOM_TOPIC=/your/odom/topic ./scripts/test-3d-integration.sh"
fi

if [[ "$REQUIRE_ANCHOR" = "1" && "$ANCHOR_COUNT" -lt 1 ]]; then
  echo "FAIL: No anchors created"
  if [[ "$POINTCLOUD_RECEIVED" -eq 0 ]]; then
    echo "  Likely cause: No point cloud data"
  elif [[ "$ODOM_RECEIVED" -eq 0 ]]; then
    echo "  Likely cause: No odometry data"
  fi
  PASS=0
fi

if [[ "$REQUIRE_LOOP" = "1" && "$LOOP_COUNT" -lt 1 ]]; then
  echo "FAIL: No loop factors detected"
  PASS=0
fi

if [[ "$REQUIRE_SLAM_ACTIVE" = "1" && "$SLAM_ACTIVE" -ne 1 ]]; then
  echo "FAIL: Backend did not reach SLAM_ACTIVE mode"
  PASS=0
fi

echo ""
if [[ "$PASS" = "1" ]]; then
  echo "============================================"
  echo "TEST PASSED"
  echo "============================================"
  exit 0
else
  echo "============================================"
  echo "TEST FAILED"
  echo "============================================"
  exit 1
fi
