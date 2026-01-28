# Pipeline Design Gaps (Known Limitations)

This doc records **known design gaps** in the Golden Child SLAM v2 pipeline, based on the raw-measurements audit, message trace, and covariance inspection. The pipeline underuses available information and treats measurements as more independent than they are; the 6D pose evidence structure does not reflect kinematics or twist.

**References:** `docs/RAW_MEASUREMENTS_VS_PIPELINE.md`, `docs/PIPELINE_TRACE_SINGLE_DOC.md`, `docs/TRACE_Z_EVIDENCE_AND_TRAJECTORY.md`, `tools/inspect_odom_covariance.py`.

---

## 1. We are not using a lot of available information

| Source | What we use | What we leave on the table |
|--------|-------------|----------------------------|
| **Odom** | Pose (x,y,z, quat) + pose covariance (6×6). **Twist (vx, vy, vz, wx, wy, wz) + twist covariance (6×6)** — **now used** (velocity factor, yaw-rate factor, pose–twist kinematic consistency; see Phase 2 odom twist evidence). | — (twist is now read and fused.) |
| **IMU** | Gyro, accel (scaled, rotated to base); preintegration; vMF gravity; gyro evidence; preint factor. | **Message covariances** (orientation, angular_velocity, linear_acceleration) — not read in backend; we use IW only. **Orientation** (if present) — not used. No explicit forward/lateral decomposition. |
| **LiDAR** | x, y, z, timebase, time_offset, ring, tag; range-based weights. | **Intensity (reflectivity)** — parsed for stride but not returned or used for weighting/features. **Per-point covariance** — message has none; we pass zeros. |

So we still underuse IMU (no message cov, no orientation) and LiDAR (no intensity). Odom twist is now used.

---

## 2. We treat measurements as more independent than they are

Physically, **pose and twist are related** by kinematics:

- **dp/dt = R @ v** (world velocity = R × body velocity)
- **dR/dt = R @ ω̂** (orientation rate from body angular rate)

So if we are moving forward (vx > 0) and yawing (wz ≠ 0), we **should** see motion in x and y over time. The odom message gives us both **pose** and **twist** at the same time; they are **not** independent. **We now use twist** (Phase 2: odom velocity evidence, yaw-rate evidence, pose–twist kinematic consistency). Remaining: we do not yet feed gyro ∫ vs odom Δyaw agreement into the fusion model (diagnostics only). Previously: we never said “this pose change is consistent (or not) with the reported velocity and yaw rate.”
Similarly, **odom pose** and **IMU preintegration** are related: integrated gyro should match odom yaw change; integrated accel + gravity should be consistent with odom position change. We log dyaw_gyro / dyaw_odom / dyaw_wahba for diagnostics but **do not** feed that consistency (or inconsistency) into the fusion model. So we treat odom pose, IMU gyro, and IMU accel as separate evidence streams without modeling their dependence.

---

## 3. The 6D pose evidence (odom) is poorly designed for what we have

Current design:

- **Input:** One 6D pose (trans + rotvec) and one 6×6 pose covariance from the message.
- **Math:** Residual = log(pred⁻¹ ∘ odom_pose) in tangent space (6D). Information = inverse(covariance). Evidence = Gaussian on that 6D residual with that information matrix.
- **Covariance:** Whatever the bag publishes (in our bag: diagonal 0.001, 0.001, 1e6, 1e6, 1e6, 1000 in [x,y,z,roll,pitch,yaw] with units m² and rad²; later messages can switch to different diagonals, e.g. yaw 1e-9). We do **not**:
  - Use twist to shape or validate the pose residual (e.g. “pose change should align with integrated twist”).
  - Model pose–twist coupling (e.g. joint observation on [pose; twist] or pose given twist).
  - Enforce planar structure (e.g. z, roll, pitch weakly or structurally constrained) beyond what the message covariance says.
  - Use forward/lateral decomposition (vx, vy, wz) so that motion in x/y is tied to “moving forward while turning.”

So the “pose 6 matrix of evidence” is just **inverse(message 6×6 pose covariance)** applied to a 6D pose error. It is not designed around:

- Kinematic coupling (pose ↔ twist).
- Use of twist (vx, vy, vz, wx, wy, wz).
- Consistency with IMU (gyro ∫ vs Δyaw; velocity vs integrated accel).

That is what we mean by “poorly designed”: it uses only part of the odom message and does not account for how pose and twist (and IMU) depend on each other.

---

## 4. What we are not doing that we need for a better design

1. **Odom twist (DONE)** — We now read vx, vy, vz, wx, wy, wz and twist covariance; feed them as velocity factor, yaw-rate factor, and pose–twist kinematic consistency. Remaining: use gyro ∫ vs odom Δyaw inside the model.

2. **Model pose–twist dependence**
   - Either: joint observation on [pose; twist] with a covariance that reflects kinematics, or
   - Pose observation conditioned on twist (e.g. “pose change given twist” residual), so we don’t treat pose as independent of how the robot is moving.

3. **Reflect kinematics in the evidence**
   - “Moving forward + yaw” should imply motion in x and y; the 6D evidence (or a richer observation model) should be structured so that translation and rotation are consistent with velocity and yaw rate (e.g. dp/dt = R @ v, dR/dt = R @ ω̂).

4. **Use forward/lateral structure**
   - Decompose motion into forward (vx), lateral (vy), and yaw (wz) from odom; optionally from IMU (ax, ay in body). Fuse in a way that respects that structure (e.g. separate or coupled blocks for forward vs lateral vs yaw) instead of one opaque 6D pose Gaussian.

5. **Consistency between sensors**
   - Use dyaw_gyro vs dyaw_odom vs dyaw_wahba (and, if added, velocity consistency) inside the model: e.g. soft constraints, or weighting that depends on agreement, not only diagnostics.

6. **Use more of the raw message**
   - Odom: twist + twist covariance — **now used**.
   - IMU: message covariances (orientation, angular_velocity, linear_acceleration) — not read in backend; use IW only.
   - LiDAR: intensity — parsed for stride but not returned or used for weighting/features.

---

## 5. Summary

- **Odom:** Twist and twist covariance **are now used**. **Underuse of info:** We do not use IMU message covariances or orientation; we do not use LiDAR intensity for weighting or features.
- **Independence assumption:** We treat odom pose, IMU evidence, and LiDAR evidence as separate; we do not model kinematic coupling (pose ↔ twist) or consistency (e.g. gyro ∫ vs odom Δyaw).
- **6D pose evidence:** It is “pose residual + inverse(message 6×6 covariance)” with no twist, no pose–twist coupling, and no design for planar or forward/lateral structure. So the pose 6 matrix of evidence is poorly matched to what we need.

These gaps should be addressed when redesigning the observation model and evidence structure (pose + twist, kinematics, and use of all relevant fields from the raw messages).

---

## 6. Operator-by-operator improvement plan

Below is a concrete improvement plan that (1) fixes the known structural failures in the current single-hypothesis pipeline, (2) extends it cleanly to **MHT**, and (3) shows where **2nd/3rd-order tensors, higher derivatives, and information-geometry / Hessian / Monge–Ampère ideas** can be used—separating **production-safe** from **high-risk research**.

### 6.0 What the pipeline currently is (and why it fails)

**Key structural facts (from this doc and the trace):**

- The pipeline is **fixed-cost and branch-free**: every LiDAR scan runs the same steps in order.
- State is a **22D tangent chart** with blocks [t, θ, v, b_g, b_a, dt, ex].
- **Odom twist is available but not used anywhere.**
- Accel evidence is a **single vMF on mean direction**, Laplace-approximated and **PSD-projected**, placed only in the rotation block (pitch/roll only).
- LiDAR translation WLS is done in **full 3D**, with isotropic measurement noise, and the resulting **z** is fused strongly—then baked into the map, creating a feedback loop.
- Fusion is **additive info-form** with an **α** computed from pose-block conditioning/support + excitation scaling on dt/extrinsic, then PSD projection.

**Failure modes implied (model-class errors, not “bugs”):**

1. **Z instability is designed behavior:** z is treated like x,y; LiDAR injects strong z; map stores belief_z; next scan aligns to that map and reinforces it.
2. **Dynamics are underconstrained** because the strongest kinematic measurements (odom twist) are ignored.
3. **Nonlinear likelihoods are forced into a quadratic mold** (vMF → Laplace + PSD projection). That is workable only when the approximation is locally valid; otherwise it produces overconfident wrong pulls.

---

### 6.1 Production-safe pipeline upgrades (do these first)

#### 6.1.1 Make the estimator match the platform: SE(2.5) not SE(3)

**Goal:** Keep full SO(3) (tilt is needed), but treat translation as planar unless vertical observability is truly available.

**Changes:**

- **TranslationWLS in (x,y) only** (or equivalently, inflate t_cov[2,2] → huge; zero-out z rows/cols before building L_lidar). The trace explicitly calls out isotropic 3D WLS as the z screw-up.
- Add a **planar process prior:** set Q_z small, enforce v_z ≈ 0 (soft), and/or add a direct factor z = z_0 if the platform is level. The trace says we never damp z / never enforce v_z = 0.
- **Map update must not reinforce vertical drift:** store map in a local tangent plane; or project bin centroids onto the plane before accumulation; or maintain a separate map-z gauge that does not equal belief_z. The trace shows “map z = belief_z + …” as the feedback loop.

**Acceptance test:** z stays bounded near the true robot height instead of running to -50 to -80 m (as in the trace).

#### 6.1.2 Add the odom twist factors (currently throwing away the best constraints)

**Add 3 new evidence operators:**

1. **Body velocity factor** (child frame): \( r_v = v_b - v^{\text{odom}}_b \). Inject into the **vel block [6:9]**, with cov from wheel model (or learned IW).
2. **Yaw-rate factor:** \( r_{\omega_z} = (\omega_{b,z} - b_{g,z}) - \omega^{\text{odom}}_z \). Inject into **gyro-bias / rotation dynamics** (helps stabilize yaw under turning).
3. **Pose–twist kinematic consistency factor** across scan dt: \( \text{Log}(X_k^{-1} X_{k+1}) \approx [R_k v_b \Delta t; \omega_b \Delta t] \). This directly repairs “pose snapshots with no dynamic linkage.”

**Net effect:** Stop asking LiDAR to do everything; reduce sensitivity to wrong scan-matching.

#### 6.1.3 Replace “single accel mean vMF” with time-resolved, consistency-weighted tilt evidence

**Current model:** One vMF on mean direction → Laplace at δθ = 0 → PSD projection; only constrains pitch/roll.

**Keep vMF, but fix the data model:** Instead of collapsing all samples to one mean direction, compute a reliability weight per IMU sample (or short window) from gyro–accel transport consistency: \( e_k = \dot f_{b,k} + \omega_{b,k} \times f_{b,k} \). Aggregate a **weighted directional likelihood:** high weight when |e_k| small (gravity-dominant), low weight when dynamic/slip/vibration. This gives continuous soft weighting instead of hoping the single mean vector represents “gravity.”

**Also fix the approximation:** Stop PSD-projecting as the default repair. If curvature is too high, use either a robust likelihood (Student-t directional) or a better local approximation (see 3rd-order section).

#### 6.1.4 Make evidence strength depend on measurement quality

Fusion is additive info-form and depends heavily on L magnitudes. If any operator produces “constant strength regardless of reliability,” it will dominate incorrectly.

**Rule:** Every evidence operator must output L = Σ⁻¹ where Σ is actually tied to: dt window length, IMU consistency residuals, scan alignment residuals / ESS, IW-adapted noise state. The trace describes IW feedback (residuals update noise used next scan).

---

### 6.2 Turning this into MHT (multiple hypothesis tracking)

The current pipeline is branch-free. MHT means introducing branching and weight management correctly.

#### 6.2.1 What a “hypothesis” is

Each hypothesis must contain **everything that affects future evidence:** belief (L, h, z_lin, X_anchor, …), **map_stats** (map drives next scan’s Wahba/WLS), **noise state (IW)** (affects Q and Σ in future steps), any latent calibration that varies (dt, extrinsic). So a hypothesis is **pose + map + noise + calibration**, not just pose.

#### 6.2.2 When to branch (practical triggers)

Branch when there is **structural ambiguity:** scan matching has multiple comparable solutions (symmetry/corridor), Wahba has competing maxima, translation WLS is ill-conditioned, evidence conditioning is bad (α already uses conditioning).

#### 6.2.3 How to score and prune hypotheses (log-evidence in information form)

For each hypothesis j, maintain weight w_j ∝ p(data | H_j). A practical scoring rule: compute the incremental negative log-likelihood at the fused solution; include the Gaussian normalization term (½ log det Σ) or (-½ log det L) in information form. This prevents “overconfident wrong” hypotheses from surviving.

#### 6.2.4 Keep hypothesis count bounded (MHT discipline)

Use a strict cap B (e.g. 4–16). **Prune** by posterior weight; **merge** near-duplicate hypotheses on SE(3) using a proper distance (e.g. geodesic on SO(3) + Mahalanobis on translation in tangent space); **delay commitment:** keep multiple for K scans, then collapse.

#### 6.2.5 How to merge hypotheses without destroying geometry

Moment match in tangent space at a chosen reference (SE(3) log map), but only when modes are close. Use OT barycenters (see Monge–Ampère section) to avoid mode collapse artifacts.

---

### 6.3 Where 2nd- and 3rd-order tensors actually improve estimation

Use higher-order objects only where they fix a known failure (linearization, curvature, multimodality).

#### 6.3.1 Second-order: metric (Hessian/Fisher) should drive the step (not just be inverted)

L is already “the Hessian of NLL” in the tangent chart. **Upgrade:** Use **Riemannian trust-region** steps on SE(3): solve for δ using L but accept/reject with a trust-region ratio; retract via SE(3) exp map. This prevents “flat subspace flinging” when conditioning is bad (which α/conditioning logic is trying to detect).

#### 6.3.2 Third-order: correct the Laplace approximation pathologies

vMF conversion uses an approximate Hessian and then PSD projection. When PSD projection is needed, it is often because we are outside the regime where the 2nd-order approximation is reliable.

**Upgrade options:**

- **Cubic-regularized Newton** (uses 3rd-order Lipschitz bound rather than explicit tensor): stabilizes steps when curvature changes fast; reduces reliance on PSD projection.
- **Explicit 3rd-order correction** (research): compute the 3rd derivative tensor of the directional log-likelihood w.r.t. δθ; apply a cubic Taylor correction to the local model. Use only for accel vMF (and maybe Wahba residual) where curvature is high and approximation stress is already seen.

---

### 6.4 High-risk research (not production-safe)

#### 6.4.1 Monge–Ampère / Optimal-Transport map updates (⚠ high risk)

**Idea:** Treat map update as transport of a distribution of scan-bin Gaussians into map-bin Gaussians using OT (transport map T = ∇φ, convex potential φ, Jacobian satisfying Monge–Ampère). Practically, approximate using Gaussian OT barycenters per bin. **Why it might help:** Reduces map contamination feedback; principled merge for MHT. **Why it’s not safe:** Expensive, brittle under outliers, hard to tune online, risk of artificial “mass conservation” that fights real motion.

#### 6.4.2 Full Hessian geometry (dually flat) fusion via convex potentials (⚠ medium–high risk)

Choose a convex potential φ and perform mirror descent in natural coordinates; treat sensor fusion as movement along e-/m-geodesics. **Why it might help:** Fusion less sensitive to chart choice and α hacks. **Why it’s risky:** Must pick the right potential and ensure numerical stability; mismatch between true likelihoods (vMF + scan matching) and assumed exponential family can backfire.

#### 6.4.3 Third-order Amari–Chentsov tensor for bias dynamics (⚠ high risk)

Use the 3rd-order information tensor to model skewness/non-Gaussianity in bias evolution or scan residuals. **Why it might help:** Can outperform Student-t in some non-Gaussian regimes. **Why it’s not safe:** Estimation extremely sensitive to modeling assumptions; easy to overfit curvature to noise.

#### 6.4.4 “Jerk-informed” slip classifiers driving hypothesis branching (⚠ medium risk)

Use 2nd/3rd derivatives of accel (ḟ, f̈, f⃛) to detect wheel slip, impacts, nonrigid mounting; then branch hypotheses with different motion models/noise. **Why it might help:** Principled trigger for MHT branching. **Why it’s risky:** Numerical differentiation is noisy; false positives create hypothesis explosion.

---

### 6.5 Implementation checklist (mapped to pipeline steps)

| Step | Change |
|------|--------|
| **Step 2 (PredictDiffusion)** | Make Q anisotropic for planar vehicle: constrain z, optionally v_z. If keeping SE(3), enforce a gauge (z prior or v_z = 0 soft factor). |
| **Step 3 (DeskewConstantTwist)** | Use odom twist as alternative or complementary deskew twist when IMU is inconsistent. In MHT: branch if IMU-vs-odom twist disagreement exceeds a robust threshold. |
| **Step 8 (TranslationWLS)** | Solve translation in (x,y) only; or inflate t_cov[2,2] massively before building LiDAR evidence. |
| **Step 9 (Evidence)** | Add: odom twist (v_b, ω_b) factors; gravity vMF time-resolved + consistency-weighted (replace single mean vMF); planar constraints (z, roll/pitch) as soft priors if physically valid. |
| **Steps 10–11 (α + additive fusion)** | Keep additive fusion; compute α per-subspace (translation vs rotation) so one weak axis does not nuke everything; replace PSD projection “repair” with robust weighting upstream (Student-t weights per factor/per bin). |
| **Step 13 (PoseCovInflationPushforward + map update)** | Stop writing belief_z into map z directly. Maintain a planar map frame, or store map points in a ground-referenced frame. |
| **Hypothesis layer (new, wraps Steps 2–13)** | Per scan: run Steps 2–13 per hypothesis; compute log-evidence increment; normalize weights; prune/merge to keep B bounded; optionally delay map updates until a hypothesis survives K steps (prevents map poisoning). |

---

### 6.6 Recommended ROI sequence

1. **Planarize z** (TranslationWLS + map update + process model).
2. **Use odom twist** (new factors) to restore kinematic observability.
3. **Fix accel evidence** (time-resolved, consistency weighted) to avoid fragile vMF–Laplace pulls.
4. Add **MHT with strict caps and delayed map commit**.

Everything “Monge–Ampère / higher-order IG” becomes meaningful only after 1–3 stop the current runaway feedback loops.

**Patch plan:** A literal patch plan keyed to code entrypoints (e.g. `translation_wls`, `imu_vmf_gravity_evidence`, `odom_quadratic_evidence`, fusion operators)—including which blocks in the 22D state each new factor writes into, and how to compute log-evidence per hypothesis from (L, h)—can be derived from this section and the pipeline trace.
