# FL-SLAM Testing Guide

This document describes the consolidated testing framework for the FL-SLAM project.

## Testing Philosophy

The FL-SLAM testing framework is organized into two tiers:

1. **Minimal Tests** - Fast validation of core functionality (~30 seconds)
2. **Integration Tests** - Full end-to-end system validation with rosbag data (~90 seconds)

## Test Scripts Overview

### Minimal Testing

**Purpose:** Validate core functionality, mathematical invariants, and module wiring without launching the full SLAM system.

**Native execution:**
```bash
./scripts/test-minimal.sh
```

**Docker execution:**
```bash
./scripts/docker-test.sh
```

**What it tests:**
- ✓ Core module imports (geometry, operators, models, nodes)
- ✓ Message type definitions (AnchorCreate, LoopFactor)
- ✓ SE(3) operations and covariance transport
- ✓ Information geometry operators (Gaussian, Dirichlet, vMF)
- ✓ ICP solver properties and bounds
- ✓ Mathematical invariants (associativity, symmetry, triangle inequality)
- ✓ Frobenius corrections and adaptive models
- ✓ RGB-D processing and multimodal fusion

**When to use:**
- Before committing code changes
- During development iterations
- As a pre-push hook
- In CI/CD pipelines (fast feedback)

### Integration Testing

**Purpose:** Validate the complete SLAM pipeline with real sensor data, including loop closure detection and backend optimization.

**Native execution:**
```bash
./scripts/test-integration.sh
```

**Docker execution:**
```bash
./scripts/docker-test-integration.sh
```

**What it tests:**
- ✓ Full ROS 2 node launch and communication
- ✓ Rosbag replay with TurtleBot3 data
- ✓ Anchor creation by frontend
- ✓ Loop closure detection
- ✓ Backend state estimation (SLAM_ACTIVE mode)
- ✓ Foxglove visualization bridge (optional)
- ✓ End-to-end SLAM pipeline

**When to use:**
- Before releasing new versions
- After major architectural changes
- Weekly regression testing
- When debugging integration issues

## Configuration

Both test scripts support environment variables for customization:

### Minimal Test Configuration
```bash
# No configuration needed - uses sensible defaults
```

### Integration Test Configuration

```bash
# Rosbag path (default: rosbags/tb3_slam3d_small_ros2)
export BAG_PATH=/path/to/custom/bag

# Timeout for full test run (default: 90 seconds)
export TIMEOUT_SEC=120

# Startup wait before validation (default: 20 seconds)
export STARTUP_SEC=30

# Require loop closure detection (default: 1)
export REQUIRE_LOOP=0  # Set to 0 to allow anchor-only tests

# Require SLAM_ACTIVE backend mode (default: 1)
export REQUIRE_SLAM_ACTIVE=0  # Set to 0 to allow other modes

# Enable Foxglove visualization (default: 1)
export ENABLE_FOXGLOVE=0  # Set to 0 to disable

./scripts/test-integration.sh
```

## Test Data

The integration tests require the TurtleBot3 SLAM rosbag dataset.

**Download test data:**
```bash
./scripts/download_tb3_rosbag.sh
```

This script:
1. Downloads the ROS1 bag from ROBOTIS Japan GitHub
2. Converts it to ROS2 format using `rosbag2_bag_v2`
3. Saves to `rosbags/tb3_slam3d_small_ros2/`

**Inspect bag contents:**
```bash
./scripts/inspect_bag_direct.py [path/to/bag]
```

## Docker Testing Workflow

### Quick Development Loop

```bash
# 1. Start development environment
./scripts/docker-run.sh

# 2. Make code changes in your editor

# 3. Run minimal tests (fast)
./scripts/docker-test.sh

# 4. If minimal tests pass, run integration (slower)
./scripts/docker-test-integration.sh

# 5. Stop containers when done
./scripts/docker-stop.sh
```

### One-Shot Testing

```bash
# Build and test in one command
./scripts/docker-build.sh && \
./scripts/docker-test.sh && \
./scripts/docker-test-integration.sh
```

## Native Testing Workflow

Requirements:
- ROS 2 Jazzy
- Built workspace (`colcon build`)

```bash
# Quick validation
./scripts/test-minimal.sh

# Full validation
./scripts/test-integration.sh
```

## Understanding Test Results

### Minimal Test Output

```
✓ Core modules imported successfully
✓ Node modules imported successfully
✓ Message types imported successfully

Running: test_audit_invariants.py
test_hellinger_identical_distributions ... PASSED
test_fr_identical ... PASSED
...
✓ All minimal tests passed
```

### Integration Test Output

```
Check 1: Anchor creation
  ✓ Detected /sim/anchor_create topic

Check 2: Loop closure detection
  ✓ Detected loop factor in logs

Check 3: Backend status
  ✓ Backend status: SLAM_ACTIVE

✓ PASS: Anchor creation detected
✓ PASS: Loop closure detected
✓ PASS: Backend running (mode: SLAM_ACTIVE)
✓ PASS: Backend in SLAM_ACTIVE mode

✓ ALL INTEGRATION TESTS PASSED
```

## Debugging Failed Tests

### Minimal Tests Fail

1. Check import errors - ensure workspace is built:
   ```bash
   cd fl_ws
   colcon build --symlink-install --packages-select fl_slam_poc
   source install/setup.bash
   ```

2. Run individual test files:
   ```bash
   cd fl_ws/src/fl_slam_poc
   python3 -m pytest test/test_audit_invariants.py -v
   python3 -m pytest test/test_rgbd_multimodal.py -v
   ```

### Integration Tests Fail

1. Check the detailed log:
   ```bash
   cat diagnostic_logs/integration_test_*.log
   ```

2. Review specific failures:
   - **No anchor creation**: Frontend not processing odometry correctly
   - **No loop closure**: Loop detection threshold too strict or insufficient motion
   - **Backend not SLAM_ACTIVE**: Check for errors in backend node startup

3. Run with extended timeout:
   ```bash
   TIMEOUT_SEC=180 STARTUP_SEC=30 ./scripts/test-integration.sh
   ```

4. Disable Foxglove if it's causing issues:
   ```bash
   ENABLE_FOXGLOVE=0 ./scripts/test-integration.sh
   ```

## CI/CD Integration

For automated testing pipelines:

```yaml
# Example GitHub Actions workflow
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: ./scripts/docker-build.sh
      
      - name: Run minimal tests
        run: ./scripts/docker-test.sh
      
      - name: Download test data
        run: ./scripts/download_tb3_rosbag.sh
      
      - name: Run integration tests
        run: ./scripts/docker-test-integration.sh
```

## Test Coverage

### Currently Tested

- ✓ Information geometry operators
- ✓ SE(3) operations
- ✓ ICP solver
- ✓ Adaptive models (NIG, process noise)
- ✓ Frontend anchor creation
- ✓ Loop closure detection
- ✓ Backend optimization
- ✓ RGB-D processing
- ✓ Multimodal fusion

### Not Yet Tested

- ⊘ Gazebo simulation integration
- ⊘ Live sensor streams (non-rosbag)
- ⊘ Performance benchmarks
- ⊘ Memory leak detection
- ⊘ Multi-robot scenarios

## Contributing

When adding new features:

1. Add unit tests to `test/test_audit_invariants.py` or `test/test_rgbd_multimodal.py`
2. Ensure minimal tests pass
3. Verify integration tests still pass
4. Update this document if adding new test scripts

## Summary

| Script | Duration | Use Case | Requires Bag |
|--------|----------|----------|--------------|
| `test-minimal.sh` | ~30s | Quick validation | No |
| `test-integration.sh` | ~90s | Full system test | Yes |
| `docker-test.sh` | ~30s | Quick Docker validation | No |
| `docker-test-integration.sh` | ~90s | Full Docker system test | Yes |

For most development work, run `docker-test.sh` frequently and `docker-test-integration.sh` before commits.
