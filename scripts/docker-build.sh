#!/bin/bash
# Build the FL-SLAM Docker image
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building FL-SLAM Docker image ==="
echo "Project directory: $PROJECT_DIR"

cd "$PROJECT_DIR"
docker compose -f docker/docker-compose.yml build fl-slam

echo ""
echo "=== Build complete ==="
echo "Run './scripts/docker-run.sh' to start the container"

