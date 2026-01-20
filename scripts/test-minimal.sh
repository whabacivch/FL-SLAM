#!/usr/bin/env bash
# Minimal FL-SLAM Test Suite
# Tests core functionality, wiring, and mathematical invariants without requiring full system launch.
# Fast execution (~30 seconds) for quick validation during development.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "FL-SLAM Minimal Test Suite"
echo "=========================================="
echo ""
echo "Testing:"
echo "  - Core module imports"
echo "  - Mathematical invariants (SE3, information geometry)"
echo "  - Geometric operators (ICP, Frobenius correction)"
echo "  - Models (NIG, adaptive parameters)"
echo ""

# Detect if running inside Docker or native
if [ -f /.dockerenv ]; then
    echo "Environment: Docker container"
    ROS_SETUP="/opt/ros/jazzy/setup.bash"
    WS_ROOT="/ros2_ws"
else
    echo "Environment: Native (requires ROS 2 Jazzy)"
    ROS_SETUP="/opt/ros/jazzy/setup.bash"
    WS_ROOT="${PROJECT_DIR}/fl_ws"
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

# Configure Python path
export PYTHONPATH="${WS_ROOT}/install/lib/python3.12/site-packages:${WS_ROOT}/install/fl_slam_poc/lib/python3.12/site-packages:${PYTHONPATH:-}"

echo ""
echo "=========================================="
echo "1. Module Import Validation"
echo "=========================================="

python3 - <<'PY'
import sys

# Test core imports
try:
    import fl_slam_poc
    from fl_slam_poc import msg
    from fl_slam_poc.geometry import se3
    from fl_slam_poc.operators import gaussian_info, information_distances
    from fl_slam_poc.models import nig, adaptive
    print("✓ Core modules imported successfully")
except Exception as exc:
    print(f"✗ Import failed: {exc}")
    sys.exit(1)

# Test node imports
try:
    from fl_slam_poc.nodes import frontend_node, fl_backend_node
    print("✓ Node modules imported successfully")
except Exception as exc:
    print(f"✗ Node import failed: {exc}")
    sys.exit(1)

# Test message types
try:
    from fl_slam_poc.msg import AnchorCreate, LoopFactor
    print("✓ Message types imported successfully")
except Exception as exc:
    print(f"✗ Message type import failed: {exc}")
    sys.exit(1)

print("\nAll imports validated ✓")
PY

echo ""
echo "=========================================="
echo "2. Unit Tests - Invariants & Operators"
echo "=========================================="
echo ""

cd "${WS_ROOT}/src/fl_slam_poc"

# Run audit invariants (core mathematical properties)
echo "Running: test_audit_invariants.py"
python3 -m pytest test/test_audit_invariants.py -v --tb=short

# Run RGB-D multimodal tests
echo ""
echo "Running: test_rgbd_multimodal.py"
python3 -m pytest test/test_rgbd_multimodal.py -v --tb=short

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "✓ All minimal tests passed"
echo ""
echo "Next steps:"
echo "  - Run full integration test: ./scripts/test-integration.sh"
echo "  - Start development environment: ./scripts/docker-run.sh"
echo ""
