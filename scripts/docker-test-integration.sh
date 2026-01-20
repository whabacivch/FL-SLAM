#!/usr/bin/env bash
# Run integration tests inside the FL-SLAM Docker container
# Full end-to-end test with rosbag replay and system validation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

BAG_PATH="${BAG_PATH:-rosbags/tb3_slam3d_small_ros2}"
ENABLE_FOXGLOVE="${ENABLE_FOXGLOVE:-1}"

echo "=========================================="
echo "FL-SLAM Integration Test (Docker)"
echo "=========================================="
echo ""
echo "Bag: $BAG_PATH"
if [[ "${ENABLE_FOXGLOVE}" -eq 1 ]]; then
  echo "Foxglove: ws://localhost:8765"
fi
echo ""

# Check if bag exists
if [ ! -d "$BAG_PATH" ]; then
    echo "ERROR: Bag not found at $BAG_PATH"
    echo ""
    echo "Download the test bag first:"
    echo "  ./scripts/download_tb3_rosbag.sh"
    exit 1
fi

# Create temporary docker-compose override to mount the bag directory
TMP_OVERRIDE="$(mktemp -t fl-slam-integration-override.XXXXXX.yml)"
cleanup() {
  rm -f "$TMP_OVERRIDE"
}
trap cleanup EXIT

cat > "$TMP_OVERRIDE" <<EOF
services:
  fl-slam:
    volumes:
      - ./rosbags:/ros2_ws/rosbags:ro
EOF

echo "Building/starting Docker containers..."
docker compose -f docker/docker-compose.yml -f "$TMP_OVERRIDE" up -d --build

echo ""
echo "Waiting for containers to be ready (5s)..."
sleep 5

# Check if container is running
if ! docker ps | grep -q fl-slam-poc; then
    echo "ERROR: Container fl-slam-poc not running!"
    docker compose -f docker/docker-compose.yml -f "$TMP_OVERRIDE" logs fl-slam
    exit 1
fi

echo ""
echo "=========================================="
echo "Running Integration Test Inside Container"
echo "=========================================="
echo ""

# Rebuild to ensure latest code
docker exec fl-slam-poc bash -c "
    source /opt/ros/jazzy/setup.bash
    cd /ros2_ws
    colcon build --symlink-install --packages-select fl_slam_poc >/tmp/colcon_build.log 2>&1
" || {
    echo "ERROR: Build failed. Fetching build log..."
    docker exec fl-slam-poc cat /tmp/colcon_build.log
    exit 1
}

# Run the integration test script inside container
docker exec fl-slam-poc bash -c "
    export BAG_PATH=/ros2_ws/$BAG_PATH
    export ENABLE_FOXGLOVE=${ENABLE_FOXGLOVE}
    export TIMEOUT_SEC=90
    export STARTUP_SEC=20
    export REQUIRE_LOOP=1
    export REQUIRE_SLAM_ACTIVE=1
    /ros2_ws/scripts/test-integration.sh
"

rc=$?

echo ""
echo "=========================================="
if [[ $rc -eq 0 ]]; then
    echo "✓ Integration test passed"
else
    echo "✗ Integration test failed"
fi
echo "=========================================="
echo ""
echo "To stop containers: ./scripts/docker-stop.sh"
echo "To copy logs from container:"
echo "  docker cp fl-slam-poc:/ros2_ws/diagnostic_logs ./diagnostic_logs/"
echo ""

exit $rc
