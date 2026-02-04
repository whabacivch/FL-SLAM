#!/bin/bash
# Geometric Compositional SLAM v2: Run + Evaluate
# Tests the branch-free implementation against ground truth
#
# Status bar shows: [STAGE] elapsed time | sensor counts | health
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# ============================================================================
# CONFIGURATION (single IMU gravity scale for evidence + preintegration)
IMU_GRAVITY_SCALE="${IMU_GRAVITY_SCALE:-1.0}"
DESKEW_ROTATION_ONLY="${DESKEW_ROTATION_ONLY:-false}"
# Rosbag playback: 1/4 speed gives 4x wall-clock time per scan.
# Playback duration: only first N seconds of bag (for testing; set BAG_DURATION to play more).
BAG_PLAY_RATE="${BAG_PLAY_RATE:-0.5}"
BAG_DURATION="${BAG_DURATION:-60}"
FULL_REPORT="${FULL_REPORT:-false}"
RERUN_REPLAY="${RERUN_REPLAY:-true}"

# ============================================================================
# Canonical bag: single Kimera bag used for all testing. See docs/KIMERA_DATASET_AND_PIPELINE.md.
# ============================================================================
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-$PROJECT_ROOT/fl_ws/src/fl_slam_poc/config/gc_unified.yaml}"
BAG_PATH="${BAG_PATH:-}"
GT_FILE="${GT_FILE:-}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/fl_ws/src/fl_slam_poc/config/gc_unified.yaml}"
ODOM_FRAME="${ODOM_FRAME:-acl_jackal2/odom}"
BASE_FRAME="${BASE_FRAME:-acl_jackal2/base}"
POINTCLOUD_LAYOUT="${POINTCLOUD_LAYOUT:-vlp16}"
LIDAR_SIGMA_MEAS="${LIDAR_SIGMA_MEAS:-0.001}"
BAG_DURATION="${BAG_DURATION:-60}"
# Extrinsics now loaded from config yaml (single source of truth) - no inline overrides
IMU_ACCEL_SCALE="${IMU_ACCEL_SCALE:-1.0}"
[ -n "$CONFIG_PATH" ] && CONFIG_ARG="config_path:=$CONFIG_PATH" || CONFIG_ARG=""

# Source common venv so PYTHON is set before reading eval config
source "$(dirname "$0")/common_venv.sh"

# Resolve BAG_PATH and GT_FILE from eval config if not explicitly set
if [ -z "$BAG_PATH" ] || [ -z "$GT_FILE" ]; then
  if [ -f "$EVAL_CONFIG_PATH" ]; then
    IFS=$'\t' read -r BAG_PATH_FROM_CFG GT_FILE_FROM_CFG < <(env -u PYTHONPATH "$PYTHON" - "$EVAL_CONFIG_PATH" <<'PY'
import sys, yaml
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
eval_cfg = cfg.get("eval", {}) if isinstance(cfg, dict) else {}
bag = str(eval_cfg.get("bag_path", "") or "")
gt = str(eval_cfg.get("gt_file", "") or "")
print(f"{bag}\t{gt}")
PY
    )
    if [ -z "$BAG_PATH" ] && [ -n "$BAG_PATH_FROM_CFG" ]; then
      BAG_PATH="$BAG_PATH_FROM_CFG"
    fi
    if [ -z "$GT_FILE" ] && [ -n "$GT_FILE_FROM_CFG" ]; then
      GT_FILE="$GT_FILE_FROM_CFG"
    fi
  fi
fi

# Fallback defaults if still unset
BAG_PATH="${BAG_PATH:-$PROJECT_ROOT/rosbags/Kimera_Data/ros2/10_14_acl_jackal-005}"
GT_FILE="${GT_FILE:-$PROJECT_ROOT/rosbags/Kimera_Data/ground_truth/1014/acl_jackal_gt.tum}"

EST_FILE="/tmp/gc_slam_trajectory.tum"
EST_BODY="/tmp/gc_slam_trajectory_body.tum"
GT_ALIGNED="/tmp/gt_ground_truth_aligned.tum"
WIRING_SUMMARY="/tmp/gc_wiring_summary.json"
DIAGNOSTICS_FILE="$PROJECT_ROOT/results/gc_slam_diagnostics.npz"
RESULTS_DIR="$PROJECT_ROOT/results/gc_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/slam_run.log"
# $PYTHON and $VENV_PATH already set by common_venv.sh (sourced above)

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ============================================================================
# STATUS BAR FUNCTIONS
# ============================================================================
status_line() {
    # Single-line status update (overwrites previous line)
    printf "\r%-80s" "$1"
}

status_bar() {
    local stage="$1"
    local elapsed="$2"
    local detail="$3"
    local color="${4:-$CYAN}"
    printf "\r${BOLD}[${color}%-12s${NC}${BOLD}]${NC} %3ds | %s" "$stage" "$elapsed" "$detail"
}

clear_status() {
    printf "\r%-80s\r" ""
}

# ASCII-only progress/stage rendering for terminals without UTF-8 (no unicode box/block chars)
BAR_FILL="${BAR_FILL:-#}"
BAR_EMPTY="${BAR_EMPTY:--}"
HEALTH_CHAR="${HEALTH_CHAR:-*}"
STAGE_LINE="${STAGE_LINE:-========================================}"

print_stage() {
    local num="$1"
    local total="$2"
    local name="$3"
    echo ""
    echo -e "${BOLD}${STAGE_LINE}${NC}"
    echo -e "${BOLD}[${num}/${total}] ${name}${NC}"
    echo -e "${BOLD}${STAGE_LINE}${NC}"
}

print_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# ============================================================================
# ERROR HANDLING
# ============================================================================
cleanup() {
    local exit_code=$?
    if [ -n "${PERF_PID:-}" ] && kill -0 "$PERF_PID" 2>/dev/null; then
        kill "$PERF_PID" 2>/dev/null || true
    fi
    if [ $exit_code -ne 0 ]; then
        clear_status
        echo ""
        echo -e "${RED}${STAGE_LINE:-========================================}${NC}"
        echo -e "${RED}${BOLD}CRASHED!${NC} Exit code: $exit_code"
        echo -e "${RED}${STAGE_LINE:-========================================}${NC}"
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo -e "${YELLOW}Last 30 lines of log:${NC}"
            tail -30 "$LOG_FILE" 2>/dev/null || true
        fi
        echo ""
        echo -e "Results dir: ${CYAN}$RESULTS_DIR${NC}"
    fi
}
trap cleanup EXIT

# ============================================================================
# HEADER
# ============================================================================
echo ""
echo -e "${BOLD}+==============================================================+${NC}"
echo -e "${BOLD}|       ${CYAN}GEOMETRIC COMPOSITIONAL SLAM v2${NC}${BOLD} - Evaluation Pipeline            |${NC}"
echo -e "${BOLD}+==============================================================+${NC}"
echo ""
echo -e "Bag:     ${CYAN}$(basename "$BAG_PATH")${NC} (play rate ${BAG_PLAY_RATE}, first ${BAG_DURATION}s)"
echo -e "Results: ${CYAN}$RESULTS_DIR${NC}"

# Clean previous
rm -f "$EST_FILE" "$GT_ALIGNED" "$WIRING_SUMMARY" "$DIAGNOSTICS_FILE"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# STAGE 0: PREFLIGHT
# ============================================================================
print_stage 0 6 "Preflight Checks"

# Venv is already set up by common_venv.sh
print_ok "Python venv selected: $VENV_PATH"
print_ok "Using python: $PYTHON"

# Check JAX + Geometric Compositional imports
# Run with a clean PYTHONPATH so system Python packages can't shadow venv wheels
# (common when ROS setup scripts have been sourced in the parent shell).
env -u PYTHONPATH "$PYTHON" - <<'PY'
import os, sys
project_root = os.environ.get("PROJECT_ROOT", ".")
pkg_root = os.path.join(project_root, "fl_ws", "src", "fl_slam_poc")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    from fl_slam_poc.common.jax_init import jax
    devices = jax.devices()
    gpu_ok = any(d.platform == "gpu" for d in devices)
    if not gpu_ok:
        print(f"ERROR: No GPU. Devices: {devices}")
        sys.exit(1)
    print(f"  JAX {jax.__version__} with GPU: OK")
except Exception as e:
    print(f"ERROR: JAX init failed: {e}")
    sys.exit(1)

try:
    from fl_slam_poc.backend.pipeline import RuntimeManifest
    from fl_slam_poc.common.belief import BeliefGaussianInfo
    print("  Geometric Compositional imports: OK")
except Exception as e:
    print(f"ERROR: Import failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

try:
    import evo, matplotlib
    print("  Evaluation tools: OK")
except Exception as e:
    print(f"ERROR: Missing eval dependency: {e}")
    sys.exit(1)
PY
print_ok "All preflight checks passed"

# ============================================================================
# STAGE 1: BUILD
# ============================================================================
print_stage 1 6 "Build Package (Fresh)"

# Drop stale paths (e.g. removed livox_ros_driver2) so colcon does not warn
filter_existing_paths() {
  local var_name="$1"
  local IFS=':' val= result=
  for val in ${!var_name}; do
    [ -d "$val" ] && result="${result:+$result:}$val"
  done
  export "$var_name=$result"
}
filter_existing_paths AMENT_PREFIX_PATH
filter_existing_paths CMAKE_PREFIX_PATH

source /opt/ros/jazzy/setup.bash
cd "$PROJECT_ROOT/fl_ws"

# Ensure a fresh build for fl_slam_poc only (avoid stale installs)
rm -rf "$PROJECT_ROOT/fl_ws/build/fl_slam_poc" "$PROJECT_ROOT/fl_ws/install/fl_slam_poc"

BUILD_START=$(date +%s)
colcon build --packages-select fl_slam_poc --cmake-clean-cache 2>&1 | while read line; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - BUILD_START))
    status_bar "BUILDING" "$ELAPSED" "$line"
done
source install/setup.bash
cd "$PROJECT_ROOT"

clear_status
print_ok "Package built successfully"

# ============================================================================
# STAGE 2: RUN SLAM
# ============================================================================
print_stage 2 6 "Run Geometric Compositional SLAM"

# ROS environment (use domain 1 to avoid CycloneDDS "free participant index" exhaustion on domain 0)
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_HOME="${ROS_HOME:-/tmp/ros_home}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_log}"
export RMW_FASTRTPS_USE_SHM="${RMW_FASTRTPS_USE_SHM:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://${PROJECT_ROOT}/config/cyclonedds.xml}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
mkdir -p "$ROS_HOME" "$ROS_LOG_DIR"

# Playback: only first BAG_DURATION seconds (e.g. 60 for quick Kimera testing).
# Timeout = playback duration + buffer for JIT and processing (do not use full bag length).
PLAYBACK_DURATION="$BAG_DURATION"
TIMEOUT_SEC=$((PLAYBACK_DURATION + 120))

echo -e "  Playback: ${CYAN}first ${PLAYBACK_DURATION}s${NC} of bag (timeout: ${TIMEOUT_SEC}s)"
echo -e "  Log: ${CYAN}$LOG_FILE${NC}"
echo ""

# Fail clearly if Kimera profile but bag or GT missing
if [ "$PROFILE" = "kimera" ]; then
  if [ ! -d "$BAG_PATH" ] && [ ! -f "$BAG_PATH" ]; then
    echo "ERROR: PROFILE=kimera but BAG_PATH missing: $BAG_PATH"
    exit 1
  fi
  if [ ! -f "$GT_FILE" ]; then
    echo "ERROR: PROFILE=kimera but GT_FILE missing: $GT_FILE"
    exit 1
  fi
  GT_LINES=$(grep -v '^#' "$GT_FILE" | wc -l)
  if [ "$GT_LINES" -lt 10 ]; then
    echo "ERROR: GT_FILE has too few poses ($GT_LINES): $GT_FILE"
    exit 1
  fi
fi

# Launch (pass config_path and frame params when set)
LAUNCH_OPTS=(
  bag:="$BAG_PATH"
  trajectory_export_path:="$EST_FILE"
  wiring_summary_path:="$WIRING_SUMMARY"
  diagnostics_export_path:="$DIAGNOSTICS_FILE"
  imu_gravity_scale:="$IMU_GRAVITY_SCALE"
  deskew_rotation_only:="$DESKEW_ROTATION_ONLY"
  bag_play_rate:="$BAG_PLAY_RATE"
  bag_duration:="$PLAYBACK_DURATION"
  odom_frame:="$ODOM_FRAME"
  base_frame:="$BASE_FRAME"
  pointcloud_layout:="$POINTCLOUD_LAYOUT"
  lidar_sigma_meas:="$LIDAR_SIGMA_MEAS"
  odom_belief_diagnostic_file:="$RESULTS_DIR/odom_belief_diagnostic.csv"
  odom_belief_diagnostic_max_scans:="200"
  use_rerun:=false
  splat_export_path:="$RESULTS_DIR/splat_export.npz"
)
[ -n "$CONFIG_ARG" ] && LAUNCH_OPTS+=( "$CONFIG_ARG" )
# Extrinsics from config yaml only (no inline overrides)
[ -n "$IMU_ACCEL_SCALE" ] && LAUNCH_OPTS+=( "imu_accel_scale:=$IMU_ACCEL_SCALE" )

# Ensure ROS launch uses the same venv Python (required for jax)
if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

ros2 launch fl_slam_poc gc_rosbag.launch.py "${LAUNCH_OPTS[@]}" \
  > "$LOG_FILE" 2>&1 &
LAUNCH_PID=$!

# Live perf monitor (CPU + GPU) for quick visibility during run.
start_perf_monitor() {
    local gpu_ok=false
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_ok=true
    fi
    (
        local prev_idle prev_total
        read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
        prev_idle=$((idle + iowait))
        prev_total=$((user + nice + system + idle + iowait + irq + softirq + steal))
        while kill -0 "$LAUNCH_PID" 2>/dev/null; do
            sleep 2
            read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
            local idle_now=$((idle + iowait))
            local total_now=$((user + nice + system + idle + iowait + irq + softirq + steal))
            local diff_idle=$((idle_now - prev_idle))
            local diff_total=$((total_now - prev_total))
            local cpu_pct=0
            if [ "$diff_total" -gt 0 ]; then
                cpu_pct=$(( (100 * (diff_total - diff_idle)) / diff_total ))
            fi
            prev_idle=$idle_now
            prev_total=$total_now

            local gpu_line=""
            if [ "$gpu_ok" = true ]; then
                gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | awk -F',' '{printf(" GPU:%s%% MEM:%s%% VRAM:%s/%sMiB",$1,$2,$3,$4)}')
            fi
            local proc_cpu=""
            if [ -n "$LAUNCH_PID" ] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
                proc_cpu=$(ps -p "$LAUNCH_PID" -o %cpu= 2>/dev/null | awk '{printf(" ROS%%:%s",$1)}')
            fi
            echo "[perf] CPU:${cpu_pct}%${proc_cpu}${gpu_line}"
        done
    ) &
    PERF_PID=$!
}
start_perf_monitor

# Monitor with status bar
SLAM_START=$(date +%s)
LAST_ODOM=0
LAST_SCAN=0
LAST_IMU=0
ALIVE=true
BACKEND_DIED=false

while [ $ALIVE = true ]; do
    sleep 2
    NOW=$(date +%s)
    ELAPSED=$((NOW - SLAM_START))
    
    # Check if process still running
    if ! kill -0 $LAUNCH_PID 2>/dev/null; then
        ALIVE=false
        break
    fi
    
    # Parse status from log
    if [ -f "$LOG_FILE" ]; then
        # Fail fast if the backend node died; otherwise we can "complete" with only a few poses.
        if grep -q "process has died.*gc_backend_node" "$LOG_FILE" 2>/dev/null; then
            BACKEND_DIED=true
            ALIVE=false
            break
        fi
        if grep -q "Pipeline error on scan" "$LOG_FILE" 2>/dev/null; then
            BACKEND_DIED=true
            ALIVE=false
            break
        fi

        STATUS_LINE=$(grep -o 'GC Status: odom=[0-9]*, scans=[0-9]*, imu=[0-9]*' "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
        if [ -n "$STATUS_LINE" ]; then
            LAST_ODOM=$(echo "$STATUS_LINE" | grep -o 'odom=[0-9]*' | cut -d= -f2)
            LAST_SCAN=$(echo "$STATUS_LINE" | grep -o 'scans=[0-9]*' | cut -d= -f2)
            LAST_IMU=$(echo "$STATUS_LINE" | grep -o 'imu=[0-9]*' | cut -d= -f2)
        fi
    fi
    
    # Health indicator (ASCII so it renders in any terminal)
    if [ $LAST_IMU -gt 0 ] && [ $LAST_SCAN -gt 0 ] && [ $LAST_ODOM -gt 0 ]; then
        HEALTH="${GREEN}${HEALTH_CHAR}${NC}"
    elif [ $LAST_ODOM -gt 0 ]; then
        HEALTH="${YELLOW}${HEALTH_CHAR}${NC}"
    else
        HEALTH="${RED}${HEALTH_CHAR}${NC}"
    fi
    
    # Progress bar (ASCII fill/empty so it renders in any terminal)
    PCT=$((ELAPSED * 100 / TIMEOUT_SEC))
    BAR_LEN=20
    FILLED=$((PCT * BAR_LEN / 100))
    EMPTY=$((BAR_LEN - FILLED))
    BAR=$(printf "%${FILLED}s" | tr ' ' "$BAR_FILL")$(printf "%${EMPTY}s" | tr ' ' "$BAR_EMPTY")
    
    printf "\r  ${BOLD}[${BAR}]${NC} %3d%% | %3ds/${TIMEOUT_SEC}s | odom:${CYAN}%d${NC} scan:${CYAN}%d${NC} imu:${CYAN}%d${NC} %b  " \
        "$PCT" "$ELAPSED" "$LAST_ODOM" "$LAST_SCAN" "$LAST_IMU" "$HEALTH"
    
    # Timeout
    if [ $ELAPSED -ge $TIMEOUT_SEC ]; then
        break
    fi
done

# Cleanup
clear_status
echo ""
echo "  Stopping SLAM..."
pkill -P $LAUNCH_PID 2>/dev/null || true
kill $LAUNCH_PID 2>/dev/null || true
sleep 2

# Check output
if [ ! -f "$EST_FILE" ]; then
    print_fail "No trajectory output!"
    echo ""
    echo -e "${YELLOW}Log tail:${NC}"
    tail -30 "$LOG_FILE"
    exit 1
fi

POSE_COUNT=$(grep -v '^#' "$EST_FILE" | wc -l)
if [ "$BACKEND_DIED" = true ]; then
    print_fail "SLAM backend crashed (trajectory has ${CYAN}$POSE_COUNT${NC} poses)"
    echo ""
    echo -e "${YELLOW}Log tail:${NC}"
    tail -50 "$LOG_FILE"
    exit 1
fi

if [ "$POSE_COUNT" -lt 10 ]; then
    print_fail "Too few poses (${CYAN}$POSE_COUNT${NC}) — likely an early backend failure"
    echo ""
    echo -e "${YELLOW}Log tail:${NC}"
    tail -50 "$LOG_FILE"
    exit 1
fi

print_ok "SLAM complete: ${CYAN}$POSE_COUNT${NC} poses"
echo "    odom=$LAST_ODOM  scan=$LAST_SCAN  imu=$LAST_IMU"

# ============================================================================
# STAGE 3: AUDIT INVARIANTS CHECK
# ============================================================================
print_stage 3 6 "Audit Invariants Check"

echo "  Running audit invariant tests..."
AUDIT_LOG="$RESULTS_DIR/audit_invariants.log"

# Run pytest on audit invariants test file
cd "$PROJECT_ROOT/fl_ws/src/fl_slam_poc"
env -u PYTHONPATH PYTHONPATH="$PROJECT_ROOT/fl_ws/src/fl_slam_poc:$PYTHONPATH" \
    "$PYTHON" -m pytest test/test_audit_invariants.py -v --tb=short 2>&1 | tee "$AUDIT_LOG"
AUDIT_EXIT_CODE=${PIPESTATUS[0]}

cd "$PROJECT_ROOT"

echo ""
if [ $AUDIT_EXIT_CODE -eq 0 ]; then
    print_ok "All audit invariants PASSED"
    PASSED=$(grep -c "PASSED" "$AUDIT_LOG" 2>/dev/null || echo "0")
    echo -e "    Tests passed: ${GREEN}$PASSED${NC}"
else
    print_warn "Some audit invariants FAILED (exit code: $AUDIT_EXIT_CODE)"
    FAILED=$(grep -c "FAILED" "$AUDIT_LOG" 2>/dev/null || echo "0")
    PASSED=$(grep -c "PASSED" "$AUDIT_LOG" 2>/dev/null || echo "0")
    echo -e "    Tests passed: ${GREEN}$PASSED${NC}"
    echo -e "    Tests failed: ${RED}$FAILED${NC}"
    echo ""
    echo -e "    ${YELLOW}See $AUDIT_LOG for details${NC}"
    exit 1
fi

# ============================================================================
# STAGE 4: EVALUATE
# ============================================================================
print_stage 4 6 "Evaluate Trajectory"

# Kimera: estimate and GT are in same body/base convention; no body-frame transform.
EST_FOR_EVAL="$EST_FILE"

# Align ground truth to estimate
echo "  Aligning ground truth..."
env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/align_ground_truth.py" \
  "$GT_FILE" \
  "$EST_FOR_EVAL" \
  "$GT_ALIGNED" 2>&1 | sed 's/^/    /'

# Verify GT/EST overlap after alignment
GT_CHECK_JSON="$RESULTS_DIR/gt_check.json"
env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/gt_checks.py" \
  --gt "$GT_ALIGNED" \
  --est "$EST_FOR_EVAL" \
  --out "$GT_CHECK_JSON" 2>&1 | sed 's/^/    /'
GT_CHECK_EXIT=${PIPESTATUS[0]}
if [ "$GT_CHECK_EXIT" -ne 0 ]; then
  echo "ERROR: Ground-truth/estimate overlap check failed."
  exit 1
fi

# Copy diagnostics if available and build cert summary
CERT_SUMMARY="$RESULTS_DIR/cert_summary.json"
if [ -f "$DIAGNOSTICS_FILE" ]; then
    cp "$DIAGNOSTICS_FILE" "$RESULTS_DIR/diagnostics.npz"
    DIAG_SCANS=$(env -u PYTHONPATH "$PYTHON" -c "import numpy as np; d=np.load('$DIAGNOSTICS_FILE'); print(int(d['n_scans']))" 2>/dev/null || echo "0")
    print_ok "Diagnostics collected: ${CYAN}$DIAG_SCANS${NC} scans"
    env -u PYTHONPATH "$PYTHON" - "$RESULTS_DIR/diagnostics.npz" "$CERT_SUMMARY" <<'PYSCRIPT'
import json
import sys
import numpy as np

diag_path = sys.argv[1]
out_path = sys.argv[2]
d = np.load(diag_path, allow_pickle=True)
n = int(d.get("n_scans", 0))
summary = {"n_scans": n}
if n > 0:
    def _get(key, default=0.0):
        return np.asarray(d.get(key, np.full((n,), default)))
    cert_exact = _get("cert_exact", 1.0).astype(np.float64)
    frob = _get("cert_frobenius_applied", 0.0).astype(np.float64)
    n_trig = _get("cert_n_triggers", 0.0).astype(np.float64)
    summary.update({
        "cert_exact_frac": float(np.mean(cert_exact)),
        "frobenius_frac": float(np.mean(frob)),
        "mean_triggers": float(np.mean(n_trig)),
        "max_triggers": float(np.max(n_trig)),
        "support_ess_total_mean": float(np.mean(_get("support_ess_total", 0.0))),
        "support_frac_mean": float(np.mean(_get("support_frac", 0.0))),
        "mismatch_nll_per_ess_mean": float(np.mean(_get("mismatch_nll_per_ess", 0.0))),
        "overconfidence_dt_asymmetry_mean": float(np.mean(_get("overconfidence_dt_asymmetry", 0.0))),
        "overconfidence_z_to_xy_ratio_mean": float(np.mean(_get("overconfidence_z_to_xy_ratio", 0.0))),
    })
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(json.dumps(summary, indent=2, sort_keys=True))
PYSCRIPT
fi

# Run evaluation (lean by default)
echo ""
echo "  Computing metrics..."
EVAL_ARGS=(
  "$GT_ALIGNED"
  "$EST_FOR_EVAL"
  "$RESULTS_DIR"
)
if [ -f "$CERT_SUMMARY" ]; then
  EVAL_ARGS+=( --cert-summary "$CERT_SUMMARY" )
fi
if [ -f "$AUDIT_LOG" ]; then
  EVAL_ARGS+=( --audit-log "$AUDIT_LOG" )
fi
if [ "$FULL_REPORT" = "true" ]; then
  EVAL_ARGS+=( --full-report )
fi
env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/evaluate_slam.py" \
  "${EVAL_ARGS[@]}" 2>&1 | sed 's/^/    /'

# Copy files
cp "$EST_FOR_EVAL" "$RESULTS_DIR/estimated_trajectory.tum"
cp "$EST_FILE" "$RESULTS_DIR/estimated_trajectory_wheel.tum"
cp "$GT_ALIGNED" "$RESULTS_DIR/ground_truth_aligned.tum"

# Copy wiring summary if available
if [ -f "$WIRING_SUMMARY" ]; then
    cp "$WIRING_SUMMARY" "$RESULTS_DIR/wiring_summary.json"
fi

print_ok "Evaluation complete"

# ============================================================================
# STAGE 4: RESULTS SUMMARY
# ============================================================================
print_stage 5 6 "Results Summary"

echo ""
echo -e "  ${BOLD}Rosbag:${NC} $(basename "$BAG_PATH") (first ${BAG_DURATION}s) | ${BOLD}GT:${NC} $(basename "$GT_FILE")"
if [ -f "$RESULTS_DIR/metrics.json" ]; then
    env -u PYTHONPATH "$PYTHON" - "$RESULTS_DIR/metrics.json" <<'PYSCRIPT'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

ate_t = data.get("ate", {}).get("translation", {}).get("full", {})
ate_r = data.get("ate", {}).get("rotation", {}).get("full", {})
rpe_1m = data.get("rpe", {}).get("1m", {}).get("translation", {})

def _fmt(v):
    try:
        return f"{float(v):.6f}"
    except Exception:
        return "N/A"

print(f"  ATE translation RMSE (m):   {_fmt(ate_t.get('rmse'))}")
print(f"  ATE rotation RMSE (deg):    {_fmt(ate_r.get('rmse'))}")
print(f"  RPE translation @ 1m (m/m): {_fmt(rpe_1m.get('rmse'))}")
PYSCRIPT
fi

# Display wiring summary if available
if [ -f "$RESULTS_DIR/wiring_summary.json" ]; then
    echo ""
    echo -e "${BOLD}${STAGE_LINE}${NC}"
    echo -e "${BOLD}  WIRING SUMMARY${NC}"
    echo -e "${BOLD}${STAGE_LINE}${NC}"
    
    # Parse JSON with Python
    env -u PYTHONPATH "$PYTHON" - "$RESULTS_DIR/wiring_summary.json" <<'PYSCRIPT'
import sys
import json

with open(sys.argv[1]) as f:
    data = json.load(f)

proc = data.get("processed", {})
dead = data.get("dead_ended", {})

print(f"  PROCESSED:")
print(f"    LiDAR scans:  {proc.get('lidar_scans', 0):>6}  → pipeline: {proc.get('pipeline_runs', 0):>6}")
print(f"    Odom msgs:    {proc.get('odom_msgs', 0):>6}  [{'FUSED' if proc.get('odom_fused') else 'NOT FUSED'}]")
print(f"    IMU msgs:     {proc.get('imu_msgs', 0):>6}  [{'FUSED' if proc.get('imu_fused') else 'NOT FUSED'}]")

if dead:
    print(f"  DEAD-ENDED:")
    for topic, count in sorted(dead.items()):
        topic_short = topic if len(topic) <= 40 else "..." + topic[-37:]
        print(f"    {topic_short:<40} {count:>6} msgs")

# Warnings
if not proc.get('odom_fused') and proc.get('odom_msgs', 0) > 0:
    print(f"  [!] Odom subscribed but NOT FUSED")
if not proc.get('imu_fused') and proc.get('imu_msgs', 0) > 0:
    print(f"  [!] IMU subscribed but NOT FUSED")
PYSCRIPT
    echo -e "${BOLD}${STAGE_LINE}${NC}"
fi

echo ""
echo -e "  ${BOLD}Outputs:${NC}"
for f in metrics.json wiring_summary.json audit_invariants.log cert_summary.json gt_check.json diagnostics.npz; do
    if [ -f "$RESULTS_DIR/$f" ]; then
        echo -e "    ${GREEN}[OK]${NC} $f"
    fi
done
if [ "$FULL_REPORT" = "true" ]; then
    ls "$RESULTS_DIR"/*.png 2>/dev/null | while IFS= read -r f; do
        echo -e "    ${GREEN}[OK]${NC} $(basename "$f")"
    done
    ls "$RESULTS_DIR"/*.txt "$RESULTS_DIR"/*.csv 2>/dev/null | while IFS= read -r f; do
        echo -e "    ${GREEN}[OK]${NC} $(basename "$f")"
    done
fi

echo ""
echo -e "${BOLD}+==============================================================+${NC}"
echo -e "${BOLD}|  ${GREEN}EVALUATION COMPLETE${NC}${BOLD}                                        |${NC}"
echo -e "${BOLD}|  Results: ${CYAN}$RESULTS_DIR${NC}"
echo -e "${BOLD}+==============================================================+${NC}"
echo ""

# ============================================================================
# STAGE 6: Launch Dashboard / Rerun Replay (optional)
# ============================================================================
if [ "$FULL_REPORT" = "true" ] && [ -f "$RESULTS_DIR/diagnostics.npz" ]; then
    echo ""
    echo -e "${BOLD}${STAGE_LINE}${NC}"
    echo -e "${BOLD}Launching Diagnostics Dashboard...${NC}"
    echo -e "${BOLD}${STAGE_LINE}${NC}"
    DASHBOARD_OUT="$RESULTS_DIR/dashboard.html"
    env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/slam_dashboard.py" \
        "$RESULTS_DIR/diagnostics.npz" \
        --output "$DASHBOARD_OUT"
    print_ok "Dashboard written: ${CYAN}$DASHBOARD_OUT${NC}"
    
    # Open dashboard in browser
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$DASHBOARD_OUT" 2>/dev/null &
        print_ok "Dashboard opened in browser"
    elif [ -n "$BROWSER" ]; then
        "$BROWSER" "$DASHBOARD_OUT" 2>/dev/null &
        print_ok "Dashboard opened in browser"
    else
        print_warn "Could not auto-open browser. Open manually: ${CYAN}$DASHBOARD_OUT${NC}"
    fi

    # Post-hoc Rerun: build .rrd from splat_export + trajectory, then open
    RERUN_RRD="$RESULTS_DIR/gc_slam.rrd"
    SPLAT_NPZ="$RESULTS_DIR/splat_export.npz"
    BEV15_RRD="$RESULTS_DIR/gc_bev15.rrd"
    if [ -f "$SPLAT_NPZ" ]; then
        if env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/build_rerun_from_splat.py" \
            "$RESULTS_DIR" --output "$RERUN_RRD" --bev15-output "$BEV15_RRD"; then
            print_ok "Rerun recording built: ${CYAN}$RERUN_RRD${NC}"
            print_ok "BEV15 Rerun recording built: ${CYAN}$BEV15_RRD${NC}"
        else
            print_warn "Rerun build failed (see above). Splat: ${CYAN}$SPLAT_NPZ${NC}"
        fi
    else
        print_warn "No splat_export.npz (backend may have exited before shutdown). Rerun not built."
    fi
    if [ -f "$RERUN_RRD" ]; then
        RERUN_CMD=""
        if [ -n "$VENV_PATH" ] && [ -x "$VENV_PATH/bin/rerun" ]; then
            RERUN_CMD="$VENV_PATH/bin/rerun"
        elif command -v rerun >/dev/null 2>&1; then
            RERUN_CMD="rerun"
        fi
        if [ -n "$RERUN_CMD" ]; then
            RERUN_WEB_PORT="${RERUN_WEB_PORT:-9090}"
            "$RERUN_CMD" --serve-web --web-viewer --web-viewer-port "$RERUN_WEB_PORT" "$RERUN_RRD" 2>/dev/null &
            RERUN_PID=$!
            sleep 2
            RERUN_URL="http://127.0.0.1:$RERUN_WEB_PORT"
            if command -v xdg-open >/dev/null 2>&1; then
                xdg-open "$RERUN_URL" 2>/dev/null &
            elif [ -n "$BROWSER" ]; then
                "$BROWSER" "$RERUN_URL" 2>/dev/null &
            fi
            if kill -0 "$RERUN_PID" 2>/dev/null; then
                print_ok "Rerun recording opened in web viewer: ${CYAN}$RERUN_RRD${NC} → $RERUN_URL"
            else
                print_warn "Rerun viewer may have exited. Open manually: ${CYAN}$RERUN_URL${NC}"
            fi
        else
            print_warn "Rerun recording saved: ${CYAN}$RERUN_RRD${NC} (install rerun-sdk and run: rerun --serve-web --web-viewer \"$RERUN_RRD\")"
        fi
    fi

    # BEV15 post-run view-layer (separate Rerun viewer)
    if [ -f "$BEV15_RRD" ]; then
        RERUN_CMD=""
        if [ -n "$VENV_PATH" ] && [ -x "$VENV_PATH/bin/rerun" ]; then
            RERUN_CMD="$VENV_PATH/bin/rerun"
        elif command -v rerun >/dev/null 2>&1; then
            RERUN_CMD="rerun"
        fi
        if [ -n "$RERUN_CMD" ]; then
            BEV15_WEB_PORT="${BEV15_WEB_PORT:-9091}"
            "$RERUN_CMD" --serve-web --web-viewer --web-viewer-port "$BEV15_WEB_PORT" "$BEV15_RRD" 2>/dev/null &
            BEV15_PID=$!
            sleep 2
            BEV15_URL="http://127.0.0.1:$BEV15_WEB_PORT"
            if command -v xdg-open >/dev/null 2>&1; then
                xdg-open "$BEV15_URL" 2>/dev/null &
            elif [ -n "$BROWSER" ]; then
                "$BROWSER" "$BEV15_URL" 2>/dev/null &
            fi
            if kill -0 "$BEV15_PID" 2>/dev/null; then
                print_ok "BEV15 Rerun viewer opened: ${CYAN}$BEV15_RRD${NC} → $BEV15_URL"
            else
                print_warn "BEV15 Rerun viewer may have exited. Open manually: ${CYAN}$BEV15_URL${NC}"
            fi
        else
            print_warn "BEV15 Rerun recording saved: ${CYAN}$BEV15_RRD${NC} (install rerun-sdk and run: rerun --serve-web --web-viewer \"$BEV15_RRD\")"
        fi
    fi

    # JAXsplat visualization: render splat export and open image
    SPLAT_NPZ="$RESULTS_DIR/splat_export.npz"
    if [ -f "$SPLAT_NPZ" ]; then
        SPLAT_OUT="$RESULTS_DIR/splat_render.png"
        # Run in a subshell so bash's own SIGSEGV diagnostic doesn't leak to stderr.
        if ( env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/view_splat_jaxsplat.py" \
            "$SPLAT_NPZ" --output "$SPLAT_OUT" --open-image ) 2>/dev/null; then
            print_ok "JAXsplat render saved: ${CYAN}$SPLAT_OUT${NC}"
        else
            print_warn "JAXsplat render skipped (install jaxsplat or check logs). Splat export: ${CYAN}$SPLAT_NPZ${NC}"
        fi
    fi
fi

# Rerun replay without dashboard (lean default)
if [ "$FULL_REPORT" != "true" ] && [ "$RERUN_REPLAY" = "true" ]; then
    RERUN_RRD="$RESULTS_DIR/gc_slam.rrd"
    SPLAT_NPZ="$RESULTS_DIR/splat_export.npz"
    BEV15_RRD="$RESULTS_DIR/gc_bev15.rrd"
    if [ -f "$SPLAT_NPZ" ]; then
        if env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/build_rerun_from_splat.py" \
            "$RESULTS_DIR" --output "$RERUN_RRD" --bev15-output "$BEV15_RRD"; then
            print_ok "Rerun recording built: ${CYAN}$RERUN_RRD${NC}"
            print_ok "BEV15 Rerun recording built: ${CYAN}$BEV15_RRD${NC}"
        else
            print_warn "Rerun build failed (see above). Splat: ${CYAN}$SPLAT_NPZ${NC}"
        fi
    else
        print_warn "No splat_export.npz (backend may have exited before shutdown). Rerun not built."
    fi
    if [ -f "$RERUN_RRD" ]; then
        RERUN_CMD=""
        if [ -n "$VENV_PATH" ] && [ -x "$VENV_PATH/bin/rerun" ]; then
            RERUN_CMD="$VENV_PATH/bin/rerun"
        elif command -v rerun >/dev/null 2>&1; then
            RERUN_CMD="rerun"
        fi
        if [ -n "$RERUN_CMD" ]; then
            RERUN_WEB_PORT="${RERUN_WEB_PORT:-9090}"
            "$RERUN_CMD" --serve-web --web-viewer --web-viewer-port "$RERUN_WEB_PORT" "$RERUN_RRD" 2>/dev/null &
            RERUN_PID=$!
            sleep 2
            RERUN_URL="http://127.0.0.1:$RERUN_WEB_PORT"
            if command -v xdg-open >/dev/null 2>&1; then
                xdg-open "$RERUN_URL" 2>/dev/null &
            elif [ -n "$BROWSER" ]; then
                "$BROWSER" "$RERUN_URL" 2>/dev/null &
            fi
            if kill -0 "$RERUN_PID" 2>/dev/null; then
                print_ok "Rerun recording opened in web viewer: ${CYAN}$RERUN_RRD${NC} → $RERUN_URL"
            else
                print_warn "Rerun viewer may have exited. Open manually: ${CYAN}$RERUN_URL${NC}"
            fi
        else
            print_warn "Rerun recording saved: ${CYAN}$RERUN_RRD${NC} (install rerun-sdk and run: rerun --serve-web --web-viewer \"$RERUN_RRD\")"
        fi
    fi

    # BEV15 post-run view-layer (separate Rerun viewer)
    if [ -f "$BEV15_RRD" ]; then
        RERUN_CMD=""
        if [ -n "$VENV_PATH" ] && [ -x "$VENV_PATH/bin/rerun" ]; then
            RERUN_CMD="$VENV_PATH/bin/rerun"
        elif command -v rerun >/dev/null 2>&1; then
            RERUN_CMD="rerun"
        fi
        if [ -n "$RERUN_CMD" ]; then
            BEV15_WEB_PORT="${BEV15_WEB_PORT:-9091}"
            "$RERUN_CMD" --serve-web --web-viewer --web-viewer-port "$BEV15_WEB_PORT" "$BEV15_RRD" 2>/dev/null &
            BEV15_PID=$!
            sleep 2
            BEV15_URL="http://127.0.0.1:$BEV15_WEB_PORT"
            if command -v xdg-open >/dev/null 2>&1; then
                xdg-open "$BEV15_URL" 2>/dev/null &
            elif [ -n "$BROWSER" ]; then
                "$BROWSER" "$BEV15_URL" 2>/dev/null &
            fi
            if kill -0 "$BEV15_PID" 2>/dev/null; then
                print_ok "BEV15 Rerun viewer opened: ${CYAN}$BEV15_RRD${NC} → $BEV15_URL"
            else
                print_warn "BEV15 Rerun viewer may have exited. Open manually: ${CYAN}$BEV15_URL${NC}"
            fi
        else
            print_warn "BEV15 Rerun recording saved: ${CYAN}$BEV15_RRD${NC} (install rerun-sdk and run: rerun --serve-web --web-viewer \"$BEV15_RRD\")"
        fi
    fi
fi
