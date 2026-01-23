# Duplication Audit (Functions/Helpers) — 2026-01-23

Scope: identify duplicated or near-duplicated functions/helpers across the repo to improve engineering discipline and reduce “mystery wiring” from copy/paste drift. This is an **audit only** (no fixes applied); each “Candidate Fix” is for explicit approval.

## Snapshot

- Python files scanned (package + tools): ~71
- Function/method definitions scanned: ~442
- Exact (AST-identical) duplicate groups (non-trivial): 4
- Notable same-name near-duplicates (high similarity): QoS resolver helper

## Status Check (Current Working Tree)

Re-scan results after your recent edits:
- **DUP-001:** still present (unchanged).
- **DUP-002:** still present (unchanged).
- **DUP-003:** still present (unchanged; line numbers shifted).
- **DUP-004:** still present (near-duplicate; unchanged).
- **DUP-005:** still present (intentional dual backend; parity discipline still needed).
- **DUP-006:** still present (signatures differ; still a duplication risk).

## Exact Duplicates (AST-identical)

### DUP-001 — `resolve_db3_path` / `_resolve_db3_path` repeated across tools

**Where**
- `tools/estimate_lidar_base_extrinsic.py:50` (`_resolve_db3_path`)
- `tools/inspect_camera_frames.py:26` (`_resolve_db3_path`)
- `tools/inspect_odom_source.py:16` (`_resolve_db3_path`)
- `tools/inspect_rosbag_deep.py:39` (`resolve_db3_path`)
- `tools/validate_livox_converter.py:31` (`resolve_db3_path`)

**Risk**
- When a bugfix or behavior change is needed (e.g., rosbag2 layout edge cases), it will likely be applied to only one copy.

**Candidate Fix (needs approval)**
- Create a shared helper module (e.g., `tools/rosbag_sqlite_utils.py`) and import it from all scripts; keep CLI behavior identical.

---

### DUP-002 — Tiny rosbag topic helpers duplicated in tools

**Where**
- `tools/estimate_lidar_base_extrinsic.py:65` (`_topic_id`)
- `tools/validate_livox_converter.py:42` (`topic_id`)
- `tools/estimate_lidar_base_extrinsic.py:71` (`_topic_type`)
- `tools/validate_livox_converter.py:48` (`topic_type`)

**Candidate Fix (needs approval)**
- Fold into the same shared helper module as DUP-001.

---

### DUP-003 — “publish OpReport as JSON” helper duplicated

**Where**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:1382` (`Frontend._publish_report`)
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/tb3_odom_bridge.py:337` (`Tb3OdomBridge.publish_report`)

**Risk**
- Small divergence (topic, validation behavior, throttling, exception handling) can make op-reporting inconsistent across nodes.

**Candidate Fix (needs approval)**
- Use one canonical publisher helper (likely `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/diagnostics/publish.py:122`), or move a minimal `publish_op_report(node, pub, report)` into `fl_slam_poc/common/`.

## Near Duplicates / Same-Name Collisions Worth Reviewing

These are not identical, but represent likely copy/paste patterns or API parity requirements.

### DUP-004 — QoS “reliability string → QoSProfile list” duplicated

**Where**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/sensor_io.py:217` (`SensorIO._resolve_qos_profiles`)
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/tb3_odom_bridge.py:117` (`Tb3OdomBridge._resolve_qos_profiles`)

**Risk**
- If “both/system_default/reliable/best_effort” behavior changes (or QoS depth defaults change), we get mismatch between sensor subscriptions and bridge subscriptions.

**Candidate Fix (needs approval)**
- Centralize in a shared module (likely `fl_slam_poc/common/qos.py`) and have both call the same function.

---

### DUP-005 — SE(3) API duplicated across NumPy and JAX backends (intentional, but needs parity discipline)

**Where (examples)**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/geometry/se3_numpy.py` vs `fl_ws/src/fl_slam_poc/fl_slam_poc/common/geometry/se3_jax.py`
- Same-name functions include: `se3_compose`, `se3_inverse`, `se3_exp`, `se3_relative`, `se3_adjoint`, `se3_cov_compose`

**Risk**
- API drift: different semantics/edge-case handling between NumPy vs JAX implementations can silently change behavior depending on which backend is used.

**Candidate Fix (needs approval)**
- Add a small “parity test” suite that asserts both implementations agree on randomly sampled inputs (within tolerances), and document any intentional differences.

---

### DUP-006 — `_is_duplicate` helper duplicated

**Where**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/sensor_io.py:290` (`SensorIO._is_duplicate`)
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/tb3_odom_bridge.py:153` (`Tb3OdomBridge._is_duplicate`)

**Note**
- These are not AST-identical (different signatures / semantics), but they are doing the same “duplicate stamp/message suppression” job and are at risk of drifting.

**Candidate Fix (needs approval)**
- Centralize into a shared “stamp key” helper (or unify the logic and naming).

## IDE/Open-Tab Note

Your open tab path `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/visual_feature_extractor.py` does not exist in this repo; the file is `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/scan/visual_feature_extractor.py`. This kind of path drift is a common contributor to duplicated logic (people re-create code in a “guessed” location).

## Proposed Approval Workflow (Fix-by-Fix)

If you want, I can propose small PR-sized patches, each gated behind “approve DUP-00X”:
1. DUP-001/DUP-002: consolidate rosbag sqlite helpers under `tools/`
2. DUP-004/DUP-006: consolidate QoS + duplicate-stamp helpers under `fl_slam_poc/common/`
3. DUP-003: unify OpReport publishing helper to remove divergence risk
4. DUP-005: add parity tests for `se3_numpy` vs `se3_jax` APIs
