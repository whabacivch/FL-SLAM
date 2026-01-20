#!/bin/bash
# Run minimal tests inside the FL-SLAM Docker container
# Quick validation of core functionality and invariants
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^fl-slam-poc$"; then
    echo "Container not running. Starting..."
    docker compose -f docker/docker-compose.yml up -d fl-slam
    sleep 2
fi

echo "=== Running FL-SLAM Minimal Tests (Docker) ==="
echo ""

# Rebuild inside container (picks up source changes via volume mount)
echo "Rebuilding package..."
docker exec fl-slam-poc bash -c '
    source /opt/ros/jazzy/setup.bash
    cd /ros2_ws
    colcon build --symlink-install --packages-select fl_slam_poc
' >/dev/null 2>&1

# Run minimal test script inside container
docker exec fl-slam-poc bash -c '
    /ros2_ws/scripts/test-minimal.sh
'

echo ""
echo "=== Minimal tests complete ==="
echo ""
echo "To run full integration tests:"
echo "  ./scripts/docker-test-integration.sh"

