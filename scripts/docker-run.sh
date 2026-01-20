#!/bin/bash
# Run FL-SLAM in Docker with Foxglove visualization
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Starting FL-SLAM Docker environment ==="
echo ""
echo "Services:"
echo "  - fl-slam: Main development container"
echo "  - foxglove: Foxglove bridge on port 8765"
echo ""
echo "Connect Foxglove Studio to: ws://localhost:8765"
echo ""

# Start services
docker compose -f docker/docker-compose.yml up -d

echo ""
echo "=== Container started ==="
echo ""
echo "To enter the container:"
echo "  docker exec -it fl-slam-poc bash"
echo ""
echo "To run tests inside container:"
echo "  docker exec -it fl-slam-poc bash -c 'source /ros2_ws/install/setup.bash && cd /ros2_ws && colcon test'"
echo ""
echo "To stop:"
echo "  ./scripts/docker-stop.sh"

