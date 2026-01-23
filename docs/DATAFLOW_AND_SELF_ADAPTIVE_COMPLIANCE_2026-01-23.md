# Current Dataflow + Self-Adaptive Systems Compliance Review — 2026-01-23

This document maps the **current end-to-end runtime dataflow** (topics → nodes → methods/operators) and compares it against the normative constraints in `docs/Self-Adaptive Systems Guide.md`.

Scope: **audit + explanation** only (no fixes). References point at the current working tree.

## 0) Executive Summary

**What works (high level)**
- The system is now **observable** (frontend wiring banner, backend status, OpReports).
- Core fusion is largely implemented as **order-robust, information-form updates** with explicit approximation auditing (OpReport + Frobenius where required).
- Several “hidden wiring” problems have been removed by parameterizing `/sim/*` topics and fixing RGB-D publish reachability.

**Where we deviate from the Self-Adaptive Systems guide**
- **No Hard Gates** is **partially violated**: several codepaths drop evidence via `return`/`continue` when a factor cannot be formed (ICP unavailable, IMU segment invalid/insufficient, unsupported encodings). These often emit OpReports, but the evidence influence does not enter inference as a continuously downweighted term.
- **Certified Approximate Operator** is **not implemented as a contract**: we log approximations via `OpReport`, but we do not have a uniform `(result, certificate, expected_effect)` return signature used end-to-end.
- **Expected vs Realized Benefit** is **not consistently logged** for adaptation/budgeting decisions (e.g., dense module culling, routing updates, budget truncation): we log triggers and some metrics, but not predicted vs realized improvement in a declared internal objective.
- **Constants as Priors/Budgets** is **partially met**: many parameters are surfaced as priors/budgets, but there are still operational “magic behaviors” that are not expressed as priors/certificates (e.g., some decoding/availability fallbacks).

## 1) Current MVP Runtime: Nodes + Topics

Primary launch entry:
- `fl_ws/src/fl_slam_poc/launch/poc_m3dgr_rosbag.launch.py`

Canonical topic truth and message semantics:
- `docs/BAG_TOPICS_AND_USAGE.md`

### 1.1 Node graph (M3DGR pipeline)

Bag inputs → utility nodes → frontend → backend → outputs:

1) LiDAR
- Bag: `/livox/mid360/lidar` (`livox_ros_driver2/msg/CustomMsg`)
- Node: `livox_converter` (`fl_slam_poc.frontend.sensors.livox_converter`)
- Output: `/lidar/points` (`sensor_msgs/PointCloud2`)

2) Odometry
- Bag: `/odom` (`nav_msgs/Odometry`, absolute)
- Node: `odom_bridge` (`fl_slam_poc.frontend.sensors.odom_bridge`)
- Output: `/sim/odom` (`nav_msgs/Odometry`, delta)

3) RGB-D (optional)
- Bag: `/camera/*/compressed` (`sensor_msgs/CompressedImage`)
- Node: `image_decompress_cpp` (C++; installed by CMake)
- Output: `/camera/image_raw` (`sensor_msgs/Image`), `/camera/depth/image_raw` (`sensor_msgs/Image`)

4) Frontend orchestration
- Node: `frontend_node` (`fl_slam_poc.frontend.frontend_node`)
- Subscribes (via SensorIO):
  - `/sim/odom`
  - `/lidar/points` (3D pointcloud mode) or `/scan` (2D mode)
  - `/livox/mid360/imu` (optional)
  - `/camera/image_raw`, `/camera/depth/image_raw`, `/camera/*/camera_info` (optional)
- Publishes:
  - `/sim/anchor_create` (`fl_slam_poc/msg/AnchorCreate`)
  - `/sim/loop_factor` (`fl_slam_poc/msg/LoopFactor`)
  - `/sim/imu_segment` (`fl_slam_poc/msg/IMUSegment`, optional)
  - `/sim/rgbd_evidence` (`std_msgs/String` JSON, optional)
  - `/cdwm/op_report` and `/cdwm/frontend_status`

5) Backend inference
- Node: `backend_node` (`fl_slam_poc.backend.backend_node`)
- Subscribes:
  - `/sim/odom`, `/sim/anchor_create`, `/sim/loop_factor`
  - `/sim/imu_segment` (if IMU fusion enabled)
  - `/sim/rgbd_evidence`
- Publishes:
  - `/cdwm/state`, `/cdwm/trajectory`, `/cdwm/map`
  - `/cdwm/op_report`, `/cdwm/backend_status`
  - `/cdwm/markers`, `/cdwm/loop_markers`, `/cdwm/debug`

### 1.2 Topic parametrization status

The following previously hardcoded `/sim/*` topics are now parameterized:
- Frontend output topics: `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:117`
- Backend input topics: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:84`

This materially improves auditability because launch/YAML now actually controls wiring.

## 2) Current Computation/Dataflow by Subsystem

### 2.1 Sensor I/O (frontend.sensors.SensorIO)

**Role:** subscription, buffering, TF/extrinsic handling, timestamp queries for alignment.

Key behaviors:
- Duplicate suppression for multi-QoS subscriptions: `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/dedup.py:1`
- QoS selection: `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/qos_utils.py:1`
- RGB and depth decoding: now pure-NumPy for common encodings; unsupported encodings are logged and dropped:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/sensor_io.py:516`
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/sensor_io.py:593`

**Dataflow outputs**
- `get_nearest_pose()` for odom alignment
- `get_synchronized_rgbd()` for dense evidence extraction
- `scan_to_points()` / pointcloud buffering for ICP source points

### 2.2 Frontend inference “methods” (frontend_node orchestration)

#### A) Alignment and continuous weights
- Uses `TimeAlignmentModel` to convert timestamp offsets into continuous weights:
  - Pose weight, image weight, depth weight
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:571`

**Self-adaptive alignment principle:** this is a *continuous scaling*, not a branch (good).

#### B) Descriptor construction + soft association
- Build a composed descriptor (scan + image feature descriptor + depth descriptor).
- Compute responsibilities over anchors + new component:
  - `LoopProcessor.compute_responsibilities()` uses Fisher-Rao distance (NIG predictive) → likelihood kernel → normalized responsibilities.
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/loops/loop_processor.py:44`

#### C) Birth model (anchor creation)
- Stochastic birth via `StochasticBirthModel` (Poisson sampling) in anchor manager:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/keyframes/anchor_manager.py:78`
- Birth probability is derived from effective new-component mass (`r_new_eff`), not a hard threshold.

**Self-adaptive alignment principle:** probabilistic creation avoids hard “if r_new > τ” gating (good).

#### D) Loop closure evidence extraction (ICP)
- For each anchor, run ICP between current points and anchor points, compute a loop factor with continuous weight scaling.
- If ICP cannot run (insufficient points / solver failure), the loop factor is not published and an OpReport is emitted:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:804` and nearby

**Deviation from “No Hard Gates”:**
- evidence is **dropped** when ICP evidence is unavailable (even if logged).

#### E) IMU segmentation (Contract B)
- Frontend buffers IMU and publishes raw segments between keyframes:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:947`
- Skips segment if insufficient samples; emits OpReport.

**Deviation from “No Hard Gates”:**
- segment is **dropped** when insufficient samples (even if logged).

#### F) Dense RGB-D evidence publishing
- When enabled and synchronized RGB-D is available, frontend publishes a JSON evidence payload to `/sim/rgbd_evidence`:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:604`
- Evidence extraction + transform lives in `frontend/sensors/rgbd_processor.py`.

### 2.3 Backend inference “methods” (backend_node orchestration)

#### A) State representation
- Pose-only (6D) or full 15D (pose + velocity + biases) depending on `enable_imu_fusion`:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:214`

#### B) Odom prediction + trust-scaled fusion
- Odom factor processed in `backend/factors/odom.py`.
- Uses information-form fusion, with a divergence-based trust scalar (α-divergence / power posterior style) to avoid catastrophic jumps:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/factors/odom.py:1`
  - (trust diagnostics are included in OpReports; see loop factor code for explicit trust metrics)

#### C) Anchor creation
- AnchorCreate produces a new anchor module/state entry:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/state/anchor_manager.py:1`

#### D) Loop factor recomposition (one-shot)
- Loop closure is handled as late evidence; backend recomposes anchor and current beliefs via Gaussian information fusion with trust scaling:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/factors/loop.py:1`

#### E) IMU factor (Contract B)
- Backend validates IMUSegment payload, buffers if anchor not yet present, then performs a two-state update + e-projection + Schur marginalization:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/factors/imu.py:1`
- Contract violations are logged and the segment is skipped:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/factors/imu.py:1`

**Deviation from “No Hard Gates”:**
- invalid segment → **return** (drops evidence influence).

#### F) Dense RGB-D layer
- Backend parses RGBD JSON payload and performs soft association to anchors + new component; culls dense modules via budgeted recomposition (with Frobenius correction):
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/state/rgbd.py:1`

## 3) What “Self-Adaptation” Exists Today (Mechanisms)

Current self-adaptive mechanisms are “local” (no global coordinator):

1) Adaptive scalar parameters via Bayesian regularization:
- `AdaptiveParameter` posterior mean with `prior_strength`:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/models/adaptive.py:39`

2) Timestamp alignment modeled as an adaptive sigma:
- `TimeAlignmentModel` uses `AdaptiveParameter` on |dt|:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/models/timestamp.py:1`

3) Adaptive process noise from residuals:
- `AdaptiveProcessNoise` updates from residual vectors:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/models/process_noise.py:1`

4) Trust-scaled fusion for loop closures:
- Backend loop factor processing emits trust diagnostics (β, divergence):
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/factors/loop.py:1`

5) Compute budgets expressed as parameters:
- Dense module max + keep fraction + pending buffers, etc:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:84`

## 4) Compliance vs `Self-Adaptive Systems Guide.md`

### 4.1 No Hard Gates (Required)

**Guide requirement:** evidence may not be discarded by control flow; only continuously downweighted.

**Status: PARTIAL**
- Good: responsibilities are continuous; trust scaling is continuous; birth is probabilistic (no hard threshold).
- Violations: evidence is dropped when “cannot form factor”, e.g.:
  - ICP unavailable → skip factor publish (`frontend_node`)
  - IMU segment invalid/insufficient → return (`frontend_node` and `backend/factors/imu.py`)
  - Unsupported sensor encodings → drop buffer entry (`SensorIO`)

### 4.2 Certified Approximate Operator (Required Contract)

**Guide requirement:** approximate operators return (result, certificate, expected_effect); downstream scales but does not branch.

**Status: NOT MET (in contract form)**
- We do have `OpReport` (audit logging) and Frobenius enforcement:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/common/op_report.py:1`
- But operators do not uniformly return an explicit “certificate” object that downstream consumes for scaling, nor do they return “expected effect” in a declared internal objective.

### 4.3 Coordinator Constraint (No optimization creep)

**Status: GENERALLY OK**
- No global iterative re-optimization coordinator is present; adaptation is local and myopic.

### 4.4 Startup Is Not a Mode

**Status: OK**
- There is initialization logic (first message), but no time-based special mode branching like `if t < N_startup`.

### 4.5 Expected vs Realized Benefit (Internal objective only)

**Status: PARTIAL**
- We log internal diagnostics for some operations (trust metrics, dropped mass, etc).
- We do not consistently log “expected vs realized” benefit for:
  - budgeted recomposition decisions
  - routing updates
  - any compute budgeting beyond “dropped/kept”

### 4.6 Constants Must Be Surfaced as Priors/Budgets

**Status: PARTIAL**
- Many are surfaced as priors/budgets via parameters and `constants.py`.
- Still missing a universal convention for:
  - certificate risk δ
  - frame budget fractions
  - explicit hazard priors for regime change detection

## 5) Deviations to Address (Prioritized)

1) **Hard-gate factor drops** (highest priority to align with SAS)
   - Replace “skip factor” with “publish factor + certificate + downweight”, where possible.
   - Candidates: ICP unavailable, IMU insufficient samples, IMU contract violations.

2) **Certified Approximate Operator contract**
   - Introduce a common return type for approximate operators: `(result, certificate, expected_effect)` and ensure consumers never branch on certificate (only scale).

3) **Expected vs realized benefit**
   - For any budgeting/truncation/routing: log predicted objective improvement vs realized after application (e.g., divergence reduction).

4) **Spec anchor missing**
   - `docs/Project_Implimentation_Guide.sty` is currently missing from the repo (only present under `archive/legacy_docs/`), which weakens enforceability of “design invariants”.

