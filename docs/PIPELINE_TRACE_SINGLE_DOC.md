# Pipeline Trace: One Document — Value as Object, Mechanism and Causality

One document. We treat each **value as an object** and follow it through the pipeline. Like putting a radioactive signature on a physical thing: we see where the raw value goes at every step and how it **contaminates** (contributes to) downstream outputs. The process is **deterministic** — just math in sequence. No branching on data; one path per operator.

**Trigger:** One LiDAR scan runs `process_scan_single_hypothesis`. It uses the **latest** odom and the **last M** buffered IMU samples. We trace: (1) the pipeline’s fixed step order (the spine), (2) each object (IMU message 5, Odom message 5, one LiDAR point) through that spine to final outputs, (3) **preintegration** (all steps P1–P8 plus body-frame outputs), (4) **belief and 6D pose Hessian** (22D information form, fusion, recompose), and (5) **adaptive noise** (Inverse-Wishart: where Q and Sigma_* come from and when they are updated).

All units are stated. Concrete numbers use **raw message 5** from the IMU and odom CSVs; LiDAR uses one representative point and the same config.

---

# Part 1: The pipeline spine (deterministic step order)

When `on_lidar` runs, it executes this sequence. Every scan runs the same steps in the same order.

| Step | What happens | Code / location |
|------|--------------|------------------|
| **L1** | Read `msg.header.stamp` → `t_scan`. | `on_lidar`: stamp_sec |
| **L2** | Parse PointCloud2: x, y, z, timebase, time_offset, ring, tag; per-point timestamps; range-based weights. Non-finite x,y,z → sentinel. | `parse_pointcloud2_vectorized()` |
| **L3** | Transform points to base: `p_base = R_base_lidar @ p_lidar + t_base_lidar`. | `on_lidar`: pts_base, points |
| **L4** | Scan bounds and `dt_sec = t_scan - t_prev_scan`. | `on_lidar`: scan_start_time, scan_end_time, dt_sec |
| **L5** | Copy last M IMU samples from buffer → `imu_stamps`, `imu_gyro`, `imu_accel`. | `on_lidar`: tail → imu_stamps_j, imu_gyro_j, imu_accel_j |
| **L6** | Call `process_scan_single_hypothesis(...)` with raw_points, raw_timestamps, raw_weights, odom_pose, odom_cov_se3, imu_*, scan times, dt_sec, Q, bin_atlas, map_stats, config. | `on_lidar` |
| **L7** | **Step 1 – PointBudgetResample:** Cap points; resample (ring, tag preserved). | pipeline: point_budget_resample() |
| **L8** | **Step 2 – PredictDiffusion:** belief_pred from previous belief, Q, dt_sec. | pipeline: predict_diffusion() |
| **L9** | **Step 3 – DeskewConstantTwist:** IMU preintegration over scan window → delta_pose_scan, xi_body; deskew points. deskewed_covs = zeros. | pipeline: preintegrate_imu_relative_pose_jax(), deskew_constant_twist() |
| **L10** | **Step 4 – BinSoftAssign:** Point directions → soft assignments to bin atlas. | pipeline: bin_soft_assign() |
| **L11** | **Step 5 – ScanBinMomentMatch:** Weighted moment match per bin (responsibilities, weights, point_covariances=zeros). | pipeline: scan_bin_moment_match() |
| **L12** | **Step 6 – KappaFromResultant:** Inside ScanBinMomentMatch. | (same) |
| **L13** | **Step 7 – WahbaSVD:** Scan-vs-map rotation → R_hat. | pipeline: wahba_svd_rotation_jax() |
| **L14** | **Step 8 – TranslationWLS:** Scan-vs-map translation → t_hat, t_cov. | pipeline: translation_wls_jax() |
| **L15** | **Step 9 – Evidence:** Odom (Gaussian pose); IMU vMF gravity; IMU gyro rotation; IMU preintegration factor; LiDAR quadratic. Combined additively (L_evidence, h_evidence). | pipeline: odom_quadratic_evidence(), imu_*_evidence(), imu_preintegration_factor(), lidar_quadratic_evidence() |
| **L16** | **Step 10 – FusionScaleFromCertificates:** Scale evidence by alpha. | pipeline: fusion_scale_from_certificates_jax() |
| **L17** | **Step 11 – InfoFusionAdditive:** Posterior info = prior_info + scaled_evidence. | pipeline: info_fusion_additive() |
| **L18** | **Step 12 – PoseUpdateFrobeniusRecompose:** Tangent update → SE(3) belief; anchor update. | pipeline: pose_update_frobenius_recompose() |
| **L19** | **Step 13 – PoseCovInflationPushforward:** Map update from scan stats with pose covariance inflation. | pipeline: pos_cov_inflation_pushforward() |
| **L20** | **Step 14 – AnchorDriftUpdate:** Reanchor on belief_recomposed. | pipeline: anchor_drift_update() |
| **L21** | IW sufficient-statistics accumulation; hypothesis and map_stats updated. | on_lidar: accum_*, hypothesis, map_update |

So: **Raw PointCloud2** → parse → T_base_lidar → budget → predict → deskew (IMU) → bin → moment match → Wahba/WLS → evidence (odom + IMU + LiDAR) → fusion → recompose → map update → anchor drift. One sequence; every value we trace below flows through this spine.

---

# Part 2: Object 1 — IMU message 5 (stamp, gyro, accel)

**Source:** Row 6 of `docs/raw_sensor_dump/imu_raw_first_300.csv` (5th message).

## 2.1 Raw object (bag / CSV)

| Field | Value | Unit |
|-------|--------|------|
| stamp_sec | 1732437229.419865 | s |
| gyro_x, gyro_y, gyro_z | 0.00348, -0.01219, -0.00835 | rad/s (IMU frame) |
| accel_x, accel_y, accel_z | -0.4655, -0.02375, 0.8787 | **g** (IMU frame) |

Message covariances (orientation, angular_velocity, linear_acceleration) are **not** used.

## 2.2 Where the object goes before the spine (on_imu)

**Step 1 – Scale accel g → m/s²**

- `accel_imu_mps2 = accel_raw * GC_IMU_ACCEL_SCALE` (9.81 m/s² per g).
- Object: **[-4.566, -0.233, 8.620] m/s²** (IMU frame). Still **specific force** (reaction to gravity).

**Step 2 – Rotate into base frame**

- `gyro_base = R_base_imu @ gyro_imu`, `accel_base = R_base_imu @ accel_imu_mps2`.  
  R_base_imu from T_base_imu rotvec [-0.015586, 0.489293, 0] rad.
- Object: gyro_base = **[-0.000804, -0.01232, -0.00882] rad/s** (base); accel_base = **[0.0217, -0.0869, 9.757] m/s²** (base). No gravity subtraction.

**Step 3 – Buffer**

- Append `(stamp_sec, gyro_base, accel_base)` to `imu_buffer`.  
So message 5’s object lives in the buffer until a LiDAR scan runs and pulls the last M samples (**L5**).

## 2.3 Object in the spine (when a LiDAR scan runs)

- **L5:** Last M samples (including message 5 if still in window) copied to `imu_stamps`, `imu_gyro`, `imu_accel` and passed into `process_scan_single_hypothesis`. No change to values.
- **L9 (Deskew):** Preintegration over (scan_start, scan_end) uses these samples. See **2.4** for the exact preintegration steps. Output: delta_pose_scan, xi_body → deskewed_points. So **message 5’s gyro/accel** contribute to where LiDAR points end up.
- **L15 (Evidence):** Same IMU arrays used for (a) preintegration over (t_last_scan, t_scan) → delta_pose_int, delta_p_int, delta_v_int; (b) gyro evidence; (c) vMF gravity evidence (accel_base vs predicted gravity); (d) preintegration factor (delta_p_int, delta_v_int vs predicted position/velocity). So **message 5** contaminates L_odom+L_imu+L_gyro+L_imu_preint+L_lidar → **L_evidence, h_evidence**.
- **L17–L20:** Evidence → fusion → posterior belief → recompose → map update → anchor drift. So **message 5** contaminates **belief_final** and thus **trajectory** (and indirectly map via belief pose).

## 2.4 Preintegration: exact steps (object = one sample i, e.g. message 5)

Preintegration receives: `imu_stamps`, `imu_gyro`, `imu_accel` (base frame), weights, `rotvec_start_WB`, `gyro_bias`, `accel_bias`, `gravity_W = (0, 0, -9.81)` m/s². For **one sample i**:

| Step | Formula | What we get |
|------|---------|-------------|
| **P1** | dt_i = t_{i+1} - t_i, dt_eff = w_i * dt_i (s) | Effective time step |
| **P2** | omega = gyro_i - gyro_bias; dR = Exp(omega * dt_eff); R_next = R_k @ dR | Rotation update (no gravity). Object (gyro) → R_next. |
| **P3** | a_body = accel_i - accel_bias (m/s²) | Bias-corrected **specific force** (body). For msg 5: ≈ (0.02, -0.09, 9.76). |
| **P4** | a_world_nog = R_k @ a_body (m/s²) | Specific force in world (still not linear accel). |
| **P5** | **a_world = a_world_nog + gravity_W** | **Only place gravity is subtracted.** a_world = linear accel (world). For msg 5 (level): ≈ (0.02, -0.09, -0.05) ≈ 0. |
| **P6** | v_next = v_k + a_world * dt_eff (m/s) | Velocity (world). Uses a_world (gravity already out). |
| **P7** | p_next = p_k + v_k*dt_eff + 0.5*a_world*dt_eff² (m) | Position (world). Uses a_world. |
| **P8** | Accumulate sum_a_body, sum_a_world_nog, sum_a_world | For diagnostics / IW. |

**After the full window:** Relative pose and body-frame outputs: **delta_R** = R_start^T @ R_end (relative rotation); **p_body_frame** = R_start^T @ p_end (world displacement in start body frame, m); **v_body_frame** = R_start^T @ v_end (velocity in start body frame, m/s); **delta_pose** (6,) = [p_body_frame, rotvec(delta_R)] for SE(3) relative pose. So **message 5’s accel** (after P5) contributes to **delta_pose_scan** (deskew), **delta_pose_int**, **delta_p_int**, **delta_v_int** → deskew twist, gyro evidence, preintegration factor → evidence → fusion → **trajectory**. Full formulas and units: `PREINTEGRATION_STEP_BY_STEP.md`.

## 2.5 Contamination summary (IMU message 5)

- **Deskew:** Via delta_pose_scan / xi_body → deskewed_points.
- **Gyro evidence:** delta_pose_int[3:6] → L_gyro, h_gyro.
- **vMF gravity:** accel_base (specific force direction) → L_imu, h_imu.
- **Preintegration factor:** delta_p_int, delta_v_int → L_imu_preint, h_imu_preint.
- **Fusion:** All of the above in L_evidence, h_evidence → posterior → **trajectory**.

---

# Part 3: Object 2 — Odom message 5 (pose, covariance)

**Source:** Row 6 of `docs/raw_sensor_dump/odom_raw_first_300.csv` (5th message). **Twist (vx,vy,vz,wx,wy,wz) is never read.**

## 3.1 Raw object (bag / CSV)

| Field | Value | Unit |
|-------|--------|------|
| stamp_sec | 1732437229.607023716 | s |
| x, y, z | 3.07225, 3.96221, 29.96797 | m (parent frame) |
| qx, qy, qz, qw | -0, 0, 0.66237, -0.74918 | — |
| pose covariance (diagonal) | 0.001, 0.001, 1e6, 1e6, 1e6, 1000 | m², m², m², rad², rad², rad² [x,y,z,roll,pitch,yaw] |

First odom (message 1): trans **[3.07019, 3.97681, 29.99595]** m, rotvec **[0, 0, -1.41998]** rad (stored as reference).

## 3.2 Where the object goes before the spine (on_odom)

**Step 1 – Absolute pose from message 5**

- Position **[3.07225, 3.96221, 29.96797]** m; quat → rotvec **[0, 0, -1.44796]** rad.  
  odom_pose_absolute = SE3(trans, rotvec).

**Step 2 – Relative to first odom**

- `last_odom_pose = first_odom_inv ∘ odom_pose_absolute`.  
  Object: **last_odom_pose** trans **[0.01474, -0.00015, -0.02798]** m, rotvec **[0, 0, -0.02798]** rad.

**Step 3 – Covariance**

- `last_odom_cov_se3 = reshape(msg.pose.covariance, (6,6))`. Unchanged. So z information = 1/1e6 = **1e-6** (very weak).

## 3.3 Object in the spine

- **L6:** `last_odom_pose` and `last_odom_cov_se3` passed into `process_scan_single_hypothesis` as odom_pose, odom_cov_se3.
- **L15 (Evidence):** **odom_quadratic_evidence:** residual = log(pred⁻¹ ∘ odom_pose) in tangent space (6D); L_odom = inv(cov_psd); h_odom = L_odom @ delta_z_star. So **message 5’s pose and cov** set the odom measurement and its information (z very weak: 1e-6).
- **L17–L20:** L_odom, h_odom in L_evidence, h_evidence → fusion → posterior → **trajectory**.

## 3.4 Contamination summary (Odom message 5)

- **Evidence:** last_odom_pose and last_odom_cov_se3 → L_odom, h_odom.
- **Fusion:** → posterior belief → **trajectory**. Twist never used.

---

# Part 4: Object 3 — LiDAR (one representative point)

**Source:** No dump of “message 5” point cloud; we trace one point through the formulas. Config: T_base_lidar = [-0.011, 0, 0.778] m, R_base_lidar = I; range weight 0.25, 0.5, 50 m.

## 4.1 Raw point (example)

- p_lidar = **(1.0, 0.2, 0.5)** m (LiDAR frame); time_offset, ring, tag from message.

## 4.2 Parse and transform (before spine)

- t = timebase_sec + time_offset×1e-9 (s). dist = sqrt(x²+y²+z²) = **1.135** m; weight w from sigmoid range formula → **w ≈ 1**.
- **p_base = p_lidar + t_base_lidar** = (1, 0.2, 0.5) + (-0.011, 0, 0.778) = **(0.989, 0.2, 1.278)** m.

## 4.3 Object in the spine

- **L2–L3:** Parse and transform (above).
- **L7:** Budget → object may be kept or dropped; ring, tag preserved.
- **L9:** Deskew: constant twist (from IMU) applied using per-point t → deskewed point.
- **L10–L11:** Soft assign to bins; moment match → scan_bins (centroids, directions, Sigma_p, etc.). Object contributes to bin statistics.
- **L13–L14:** Wahba → R_hat; TranslationWLS → **t_hat (3D)**, t_cov (3×3). Object’s contribution is part of **t_hat** (including **t_hat[2]** = z).
- **L15:** LiDAR evidence from R_hat, t_hat, t_cov (full 3D; no z-downweighting). This point **contaminates** t_hat → L_lidar (translation block including z) → L_evidence, h_evidence.
- **L17–L20:** Fusion → posterior → **trajectory** and **map** (map update uses R_hat, t_hat / belief pose).

## 4.4 Contamination summary (LiDAR point)

- **Scan bins** → R_hat, t_hat (3D) → LiDAR evidence (full 3D trans) → fusion → **trajectory** and **map**.

---

# Part 5: Combined flow — raw objects → evidence → final outputs

1. **IMU message 5:** raw (g, rad/s) → scale → rotate → buffer (specific force). When scan runs: **L5** → preintegration **P1–P8**; at **P5** gravity subtracted (a_world = a_world_nog + (0,0,-9.81)) → linear accel → delta_pose, delta_p, delta_v. These feed **L9** deskew, **L15** gyro evidence, vMF gravity, preintegration factor → L_evidence, h_evidence → **L17** fusion → **belief_updated** → **trajectory**.
2. **Odom message 5:** raw pose + cov → relative pose, cov unchanged → last_odom_pose, last_odom_cov_se3 → **L6** → **L15** odom_evidence → L_odom, h_odom → fusion → **trajectory**.
3. **LiDAR point:** raw → parse → base → **L7** budget → **L9** deskew (using IMU) → **L10–L11** bin → **L13–L14** Wahba/WLS → R_hat, **t_hat (3D)** → **L15** LiDAR evidence → fusion → **trajectory** and **map**.

**Trajectory pose (including z)** gets contributions from: (1) odom pose (z very weak, 1e-6); (2) LiDAR t_hat (full 3D, strong z); (3) IMU via delta_p, delta_v (gravity already subtracted in P5) and vMF/gyro. A large z in the trajectory comes from **LiDAR translation evidence** and map–scan feedback (see `TRACE_Z_EVIDENCE_AND_TRAJECTORY.md`), not from forgetting to subtract gravity in preintegration. Full preintegration step detail: `PREINTEGRATION_STEP_BY_STEP.md`.

---

# Part 6: Belief and 22D information (6D pose Hessian)

The pipeline keeps state and belief in **information form** on a **22D tangent space**. Every evidence term and the fusion/recompose steps operate on this same representation.

## 6.1 State dimension and blocks (D_Z = 22)

| Slice | Block | Dimension | Meaning |
|-------|--------|-----------|---------|
| 0:3 | trans | 3 | Translation (m) |
| 3:6 | rot | 3 | Rotation (rotvec, rad) |
| 6:9 | vel | 3 | Velocity (m/s) |
| 9:12 | bg | 3 | Gyro bias (rad/s) |
| 12:15 | ba | 3 | Accel bias (m/s²) |
| 15:16 | dt | 1 | Time offset (s) |
| 16:22 | ex | 6 | LiDAR–IMU extrinsic (6D pose) |

**Pose block** = 0:6 = [trans, rot]. This is the 6D block that becomes SE(3) pose; its **information matrix** is **L[0:6, 0:6]** (the 6×6 Hessian of the NLL in tangent space).

## 6.2 Belief representation

- **Belief** = (L, h, z_lin, X_anchor, …). **L** = (22, 22) information matrix (= Hessian of negative log-likelihood at z_lin). **h** = (22,) information vector; mean in tangent space is **μ = L⁻¹ h** (when L is invertible). **z_lin** = (22,) linearization point (chart origin).
- **Covariance** (when needed): Σ = L⁻¹ (e.g. for map inflation, diagnostics). Pose covariance = Σ[0:6, 0:6].

## 6.3 Where the 22D / 6D Hessian appears in the spine

| Step | What happens to L / belief |
|------|----------------------------|
| **L8 (PredictDiffusion)** | belief_prev (L_prev, h_prev) → convert to cov_prev = L_prev⁻¹ → OU diffusion: cov_pred = f(cov_prev, Q, dt_sec) → invert to L_pred, h_pred; output belief_pred. **Q** (22×22) from process-noise IW state. |
| **L15 (Evidence)** | Each evidence term produces (L_*, h_*) (22×22, 22). Odom: L_odom[0:6,0:6] = inv(odom_cov_se3), h_odom[0:6] = L_odom @ delta_z_star. LiDAR: L_lidar from R_hat, t_hat, t_cov (pose block + optional extrinsic block). IMU gyro: rotation block. IMU vMF: rotation block. IMU preint: position and velocity blocks (0:3, 6:9). All combined: **L_evidence** = L_lidar + L_odom + L_imu + L_gyro + L_imu_preint, **h_evidence** = sum of h_*. |
| **L16 (FusionScale)** | Scale evidence by alpha (from certificates): L_ev_scaled, h_ev_scaled. Excitation scaling can scale prior L/h on extrinsic block 16:22. |
| **L17 (InfoFusionAdditive)** | **L_post = L_prior + α L_evidence**, **h_post = h_prior + α h_evidence**. Posterior belief in information form; pose block L_post[0:6,0:6] is the updated 6D pose Hessian. |
| **L18 (PoseUpdateFrobeniusRecompose)** | **delta_z** = L_post⁻¹ @ h_post (MAP increment, 22D). **delta_pose_z** = delta_z[0:6]. Frobenius BCH correction on pose → delta_pose_corrected. New world pose: X_new = X_anchor ∘ Exp(delta_pose_corrected). Then shift chart: z_lin_new = z_lin - shift (pose part), h_new = h - L @ shift so non-pose state is preserved. Output belief_recomposed. |
| **L20 (AnchorDriftUpdate)** | Optional reanchor (rho); updates z_lin and h, same 22D form. |

So: **6D pose Hessian** = L[0:6, 0:6] at every step; it is updated by predict (L8), then by additive evidence (L15–L17), then the pose part is used in recompose (L18) to get the SE(3) trajectory pose.

---

# Part 7: Adaptive noise (Inverse-Wishart)

Noise is **not** fixed: **Q** (process), **Sigma_g** (gyro), **Sigma_a** (accel), **Sigma_meas** (LiDAR) come from **IW state** and are updated every scan from sufficient statistics. So raw values also “contaminate” the **noise** used on the next scan.

## 7.1 Where IW state is read (start of scan, before L6)

- **backend_node** (once per scan, before looping over hypotheses):  
  - **config.Sigma_meas** = measurement_noise_mean_jax(measurement_noise_state, idx=2) (LiDAR).  
  - **config.Sigma_g** = measurement_noise_mean_jax(measurement_noise_state, idx=0) (gyro).  
  - **config.Sigma_a** = measurement_noise_mean_jax(measurement_noise_state, idx=1) (accel).  
  - **Q_scan** = process_noise_state_to_Q_jax(process_noise_state) (22×22).  
- These are passed into `process_scan_single_hypothesis` as **config** and **Q**. So **this scan** uses the IW state **from the end of the previous scan**.

## 7.2 Where adaptive noise is used in the spine

| Step | Uses |
|------|------|
| **L8 (PredictDiffusion)** | **Q_scan** (22×22) in OU diffusion. |
| **L15 (Evidence)** | **Sigma_g** in gyro evidence and (via dt) in gyro rotation covariance; **Sigma_a** in preintegration factor and vMF/accel IW; **Sigma_meas** (or per-bucket IW) in LiDAR translation WLS and LiDAR evidence. Odom uses **message** covariance (not IW). |
| **L21 (IW accumulation)** | During pipeline, each hypothesis returns **iw_process_dPsi, iw_process_dnu**, **iw_meas_dPsi, iw_meas_dnu**, **iw_lidar_bucket_dPsi, iw_lidar_bucket_dnu**. These are accumulated (weighted by hypothesis weights). |

## 7.3 Where IW state is updated (after hypotheses combined)

- **After** all hypotheses run and are combined, **backend_node** applies IW updates **once per scan** (no gating; readiness is a weight on sufficient stats):  
  - **process_noise_state** ← process_noise_iw_apply_suffstats_jax(accum_dPsi, accum_dnu, dt_sec).  
  - **measurement_noise_state** ← measurement_noise_apply_suffstats_jax(accum_meas_dPsi, accum_meas_dnu).  
  - **lidar_bucket_noise_state** ← lidar_bucket_iw_apply_suffstats_jax(accum_lidar_bucket_dPsi, accum_lidar_bucket_dnu).  
- Then **Q** and **config.Sigma_g, Sigma_a, Sigma_meas** are updated from the new state for the **next** scan.

So: **adaptive noise** = read IW state at scan start → use Q and Sigma_* in L8 and L15 → accumulate sufficient stats in L21 → apply IW update after scan → next scan uses updated noise. Raw residuals (odom, IMU, LiDAR) from this scan therefore influence the **noise** used for the next scan.

---

# Units summary

| Quantity | Unit |
|----------|------|
| Time, stamp_sec | s |
| Gyro | rad/s |
| Accel (raw Livox) | g |
| Accel (after scale) | m/s² |
| gravity_W | m/s² |
| Position, translation | m |
| Rotation (rotvec) | rad |
| Pose covariance (x,y,z) | m² |
| Pose covariance (roll,pitch,yaw) | rad² |
| Sigma_meas (LiDAR) | m² |
| Weights | — |

---

This is the single trace document: one mechanism, deterministic math, values as objects followed from raw inputs to final outputs.

---

# Part 8: Does the pipeline trace explain bad performance? (Results vs trace)

**Short answer: yes.** Recent runs (e.g. `results/gc_20260128_105746`) show very poor metrics: **ATE translation RMSE ~47 m**, **ATE rotation RMSE ~116°**, **RPE @ 1m ~12.5 m/1m** (translation), and estimated trajectory **z** drifting to **-50 m to -80 m** (ground truth is planar, z ≈ 0.86 m). The pipeline trace and design-gaps docs explain *why* this happens.

## 8.1 Z drift (huge z in trajectory) — and where we're screwing up the z calculation

**Yes, the trace explains z massive accumulation.** It’s not one wrong formula; it’s a chain of design choices that are wrong for a planar robot.

### Where exactly we're screwing up z

| Where in the pipeline | What we do (the screw‑up) | Consequence |
|------------------------|----------------------------|-------------|
| **Odom evidence (L15)** | Use **cov[2,2] = 1e6** m² for z → **L_odom[2,2] = 1e-6**. | We *tell* the filter “z is almost unobserved” from odom, but we still feed odom_z. So z is barely pulled toward odom; any other z source dominates. |
| **TranslationWLS (L14)** | Solve for **t_hat (3D)** with **isotropic Sigma_meas** (0.01 m² for x,y,**z**). No z-downweighting, no planar prior. | **t_hat[2]** is estimated with the **same** precision as t_hat[0], t_hat[1]. So we treat vertical translation as just as observable as horizontal. |
| **LiDAR evidence (L15)** | **L_lidar[0:3,0:3] = inv(t_cov)** — full 3D, including **L_lidar[2,2]** from t_cov[2,2]. | We add **strong** z evidence from LiDAR every scan. For a planar robot we should not (or we should inflate t_cov[2,2] / zero out z). |
| **Map update (L19)** | Map centroids = **belief pose** ∘ scan. So **map z = belief_z + (R @ p_scan)_z**. | If belief_z is wrong, the **map’s z is wrong**. Next scan aligns to that map → **t_hat[2]** matches the wrong map z → we feed that back as strong evidence → **belief_z** moves further wrong. **Feedback loop.** |
| **Fusion (L17)** | **L_pose[2,2] = L_odom[2,2] + L_lidar[2,2] + …** ≈ **L_lidar[2,2]** (odom is 1e-6). | Z is **dominated by LiDAR**. So whatever t_hat[2] says (including errors from map feedback) gets fused with high weight. |
| **Process / velocity (L8, L15)** | **Q** trans block same for x,y,z; **no vel_z = 0**; preintegration gives **delta_p_int[2]**, **delta_v_int[2]** with full weight. | We never **damp** z. Once z (or vel_z) is in the state, it can **accumulate** every scan; nothing pulls it back to planar. |

So the **z calculation** is “wrong” in this sense: we **compute** z correctly (no single sign/axis bug), but we **use** z as if the robot were 3D. We give z strong evidence from LiDAR, we put belief_z into the map and then reinforce it, and we never constrain z to a plane. That **is** the screw‑up: **treating z like x and y** + **map–scan feedback on z**.

### How the accumulation grows (step by step)

1. **Seed:** A small **t_hat[2]** appears (e.g. centroid mismatch scan vs map, or odom’s weak 1e-6 pull, or numerical noise). That gets fused with **strong** L_lidar[2,2] → **belief_z** moves a bit.
2. **Map:** Map is updated with that belief → **c_map** now has that z offset.
3. **Next scan:** TranslationWLS fits **t_hat** to align scan to **c_map** → **t_hat[2]** is consistent with the wrong map z → we add strong z evidence again → **belief_z** moves further.
4. **Repeat:** Every scan, **belief_z → map z → t_hat[2] → L_lidar z → belief_z**. Z **accumulates** in the same direction (e.g. negative) because we keep reinforcing it. Process/velocity don’t oppose it.

So **yes, the trace explains z massive accumulation**: it’s the designed behaviour of (1) strong LiDAR z evidence, (2) map built from belief so map z = belief_z, (3) next t_hat[2] aligned to that map z and fed back as strong evidence, (4) no planar prior or z-downweighting anywhere. Fixes: downweight or zero z in LiDAR evidence (e.g. planar prior, inflate t_cov[2,2], or don’t use t_hat[2]); and/or add planar process/prior (smaller Q for z, vel_z = 0); and/or constrain map/TranslationWLS so z isn’t reinforced (see TRACE_Z_EVIDENCE_AND_TRAJECTORY end).

## 8.2 X,Y and rotation errors (ATE ~47 m, rotation ~116°)

- **Trace (PIPELINE_DESIGN_GAPS):** We **do not use odom twist** (vx, vy, vz, wx, wy, wz). Pose and twist are **kinematically linked** (dp/dt = R @ v, dR/dt = R @ ω̂); we treat the 6D pose as a snapshot with **inverse(message covariance)** and no link to velocity or yaw rate. We do not model pose–twist coupling or use twist to constrain pose change. So:
  - No observation that “pose change should match integrated twist.”
  - No velocity / yaw-rate observation from odom.
  - Evidence is effectively: odom 6D pose (x,y strong; z very weak) + IMU (gyro, vMF, preint) + LiDAR (R_hat, t_hat full 3D). If **LiDAR R_hat** or **t_hat** is wrong (e.g. first-scan frame, map mismatch, Wahba sign/frame issues), that wrong evidence is fused with the same weight structure and can dominate.
- **Result:** X,Y and yaw can drift because we never tie them to odom twist or kinematics. Large roll/pitch/yaw errors (e.g. ~86°, ~50°, ~96° RMSE) are consistent with orientation being wrong or flipped (e.g. Wahba/first-scan frame issues, or LiDAR rotation evidence dominating and pulling away from odom/IMU). The trace explains that **underuse of odom twist** and **no pose–twist coupling** leave x,y and rotation under-constrained and vulnerable to wrong LiDAR evidence.

## 8.3 Map–scan feedback amplifies any error

- **Trace (Part 4, Part 5):** Map is built from **belief pose** and scan; next scan’s Wahba/WLS align to that map → **R_hat, t_hat** → LiDAR evidence → fusion → new belief. So **any** error in belief (z, x, y, or rotation) gets baked into the map and then **reinforced** by the next scan’s alignment. There is no separate “ground truth” correction; the pipeline is causal and deterministic.
- **Result:** Initial errors (e.g. wrong R_hat from first scan or calibration) propagate and amplify. The trace explains why errors grow over time rather than being corrected.

## 8.4 Summary: trace ↔ performance

| Observed result | Explained by trace |
|-----------------|--------------------|
| Z drift to -50 m … -80 m | Odom z = 1e-6; LiDAR full 3D t_hat + map–scan feedback; no planar prior (Part 3–5, TRACE_Z_EVIDENCE_AND_TRAJECTORY). |
| ATE trans ~47 m, RPE ~12.5 m/1m | No odom twist; no pose–twist coupling; 6D pose evidence = inverse(message cov) only; wrong LiDAR R_hat/t_hat can dominate (PIPELINE_DESIGN_GAPS). |
| ATE rotation ~116°, roll/pitch/yaw huge | Same: no kinematic coupling; LiDAR rotation (Wahba) and fusion structure; possible Wahba/frame/sign issues. |
| Errors grow over time | Map–scan feedback: belief → map → R_hat/t_hat → evidence → belief (Part 4, 5). |

So the pipeline trace **does** explain why performance is so bad: it is the **expected outcome** of the current design (underuse of odom twist, full 3D LiDAR evidence with no planar constraint, no pose–twist coupling, map–scan feedback). Improving performance would require addressing those gaps (see PIPELINE_DESIGN_GAPS §4).
