#!/bin/bash
# Start Rerun viewer in isolated environment with numpy 2.x
# The SLAM system will connect to this viewer remotely

RERUN_VENV="/tmp/rerun_env"

# Check if venv exists
if [ ! -d "$RERUN_VENV" ]; then
    echo "Creating Rerun virtual environment..."
    python3 -m venv "$RERUN_VENV" --clear
    "$RERUN_VENV/bin/pip" install --upgrade pip
    "$RERUN_VENV/bin/pip" install rerun-sdk
fi

echo "Starting Rerun viewer..."
echo "Connect from SLAM system using: rr.connect('127.0.0.1:9876')"
echo ""

# Start Rerun viewer listening for connections
"$RERUN_VENV/bin/rerun" --port 9876 --memory-limit 75%
