# Geometric Compositional SLAM v2 configuration

**Single source of truth: `gc_unified.yaml`**

Launch loads `gc_backend.ros__parameters` from this file. No overrides in launch file or run scripts.

- **gc_unified.yaml** — Canonical config (hub + backend). Contains extrinsics from Kimera calibration.
- **calibration/** — Reference extrinsics from Kimera_Data/calibration (6D format).
- Backend also consumes `/gc/sensors/visual_features` from `visual_feature_node` (C++).
