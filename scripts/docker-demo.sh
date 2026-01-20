#!/bin/bash
# Run FL-SLAM POC demo with Foxglove visualization
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^fl-slam-poc$"; then
    echo "Container not running. Starting..."
    docker compose -f docker/docker-compose.yml up -d
    sleep 3
fi

echo "=== FL-SLAM POC Demo ==="
echo ""
echo "Foxglove Studio: Connect to ws://localhost:8765"
echo ""
echo "Available launch files:"
echo "  poc_a     - Basic simulation with frontend + backend"
echo "  poc_b     - Alternative configuration"  
echo "  poc_all   - Full system demo"
echo ""

LAUNCH_FILE="${1:-poc_a}"

echo "Starting: $LAUNCH_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Note: `docker/docker-compose.yml` already starts a dedicated `foxglove` service
# that runs `foxglove_bridge` on port 8765. We intentionally DO NOT start another
# bridge here to avoid duplicate publishers / confusing connection behavior.

# Run the POC launch file
docker exec -it fl-slam-poc bash -c "
    source /opt/ros/jazzy/setup.bash
    source /ros2_ws/install/setup.bash
    ros2 launch fl_slam_poc ${LAUNCH_FILE}.launch.py
"

