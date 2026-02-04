# Pipeline order and evidence

Corrected spine for Geometric Compositional SLAM v2 per-scan pipeline: evidence uses the pre-update map; visual pose evidence is linearized at an IMU+odom-informed pose; map update uses the post-recompose pose.

## Corrected spine (order)

1. **Predict** — Belief prediction (diffusion).
2. **Deskew** — Constant-twist deskew.
3. **IMU+odom evidence → z_lin** — Build odom + IMU evidence (odom quadratic, vMF gravity, gyro, preintegration, planar priors, odom twist) from **belief_pred**. Fuse only IMU+odom: L_imu_odom, h_imu_odom. Solve (L_pred + L_imu_odom)^{-1}(h_pred + h_imu_odom) → **z_lin** (22D). Extract **z_lin_pose** (first 6 components) as linearization point for visual evidence.
4. **Surfel extraction + camera batch** — Measurement batch (surfels + camera splats).
5. **Map view M_{t-1}** — Extract PrimitiveMapView from current map; **do not mutate map yet**.
6. **Association** — MA hex candidate generation → cost → unbalanced Sinkhorn → (π, candidate_indices).
7. **Visual pose evidence** — `visual_pose_evidence(M_{t-1}, measurement_batch, π, z_lin_pose)` → L_pose, h_pose. Evidence is evaluated at **z_lin** (IMU+odom-informed), not raw prediction.
8. **Fuse evidence** — L_evidence = L_imu_odom + L_pose, h_evidence = h_imu_odom + h_pose. Apply fusion scale, InfoFusionAdditive, Frobenius recompose → **z_t** (updated belief).
9. **Map update with z_t** — Use **z_t** (post-recompose pose): R_t, t_t = mean_world_pose(z_t). Transform measurement primitives to world with (R_t, t_t). Run primitive_map_fuse (associated), primitive_map_insert (unassociated).
10. **Cull / forget / merge_reduce** — Map maintenance after update.
11. **Anchor drift, output** — Anchor drift update; emit belief, trajectory, map.

## Invariants

1. **Evidence uses pre-update map.** Visual pose evidence is computed from M_{t-1} and association (π). No fuse/insert before visual_pose_evidence.
2. **z_lin is IMU+odom-informed.** The linearization point for visual evidence is the posterior mean after fusing only IMU+odom with the predicted belief, not the raw prediction.

## Depth contract

**camera_depth is the authoritative depth sensor (RGB-D).** LiDAR is fused into it via `lidar_depth_evidence` in `splat_prep_fused` (Product-of-Experts: Λf = Λc + Λ_ell, θf = θc + θ_ell).

- **No separate use** of raw camera_depth for a different likelihood.
- **No double-counting:** one fused depth path only (camera Λc, θc from feature meta + LiDAR Λ_ell, θ_ell from `lidar_depth_evidence`).

**Code alignment:**

- **Frontend:** `splat_prep_fused` (frontend/sensors/splat_prep.py) uses camera depth from feature meta (`depth_Lambda_c`, `depth_theta_c`) and `lidar_depth_evidence(u, v, ...)` from `lidar_camera_depth_fusion.py`; fused depth is the single 3D lift for splats.
- **Backend:** Camera batch is built from RGB + depth; no second path consumes camera_depth for a separate term. Primitives (surfels + camera splats) are the single measurement representation.

## References

- Pipeline and data flow: `docs/IMU_BELIEF_MAP_AND_FUSION.md`
- Geometric Compositional spec: `docs/GC_SLAM.md`
