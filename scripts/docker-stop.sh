#!/bin/bash
# Stop FL-SLAM Docker containers
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Stopping FL-SLAM Docker containers ==="
docker compose -f docker/docker-compose.yml down

echo "Done."

