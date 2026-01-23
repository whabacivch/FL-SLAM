# Golden Child SLAM v2 — Strict Interface Spec (Branch-Free, Fixed-Cost, Local-Chart) (2026-01-23)

This document is the **strict interface + budget spec** for the “Golden Child SLAM Method (v2)”, rewritten to eliminate **all `if/else` gates and regime switches**. Every operator is a **total function** (always runs), with **continuous influence scalars** whose effect may smoothly go to ~0. Any numerical/domain stabilization is **declared**, **always applied in the same way**, and **accounted for** in certificates.

Scope:

* Concrete field names, dimensions, invariants, and fixed-cost budgets.
* Every approximation is an **ApproxOp** returning `(result, certificate, expected_effect)` and must be logged.
* No silent fallbacks, no multipaths, no hidden iteration (data-dependent solver loops) inside a single operator call.
* Local charts (anchors) are mandatory and **continuous** (no threshold gating).

Non-goals:

* This does not prescribe ROS message schemas; it specifies the internal library contracts that ROS nodes must call.

---

## 1) Global Invariants (Hard)

### 1.1 Chart convention + state ordering (fail-fast)

* `chart_id = "GC-RIGHT-01"` is the global chart convention for all beliefs and evidence.
* SE(3) perturbation is **right**: `X(δξ) = X · Exp(δξ)`.
* Tangent ordering is fixed:

|               Slice | Symbol       | Dim | Indices (0-based) |
| ------------------: | ------------ | --: | ----------------- |
|               SO(3) | δθ           |   3 | 0..2              |
|                   t | δt           |   3 | 3..5              |
|                   v | δv           |   3 | 6..8              |
|           gyro bias | δbg          |   3 | 9..11             |
|          accel bias | δba          |   3 | 12..14            |
|         time offset | δΔt          |   1 | 15                |
| LiDAR–IMU extrinsic | δξLI (se(3)) |   6 | 16..21            |

* Augmented tangent dimension: `D_Z = 22`.
* Deskew tangent dimension: `D_DESKEW = 22`.
* Any `chart_id` mismatch is a hard error.
* Any dimensional mismatch is a hard error.

### 1.2 Local charts (anchors) are mandatory (hard)

* `chart_id` specifies the convention; **local charts are specified by anchors**.
* Every belief and evidence carries:

  * `anchor_id: str` (local chart instance id; stable within a hypothesis stream)
  * `X_anchor: SE3` (anchor pose in module-local/world frame)
* No operator is allowed to branch on anchor logic. Anchor evolution is continuous and always applied by `AnchorDriftUpdate` (ApproxOp).

### 1.3 Fixed-cost budgets (compile-time constants)

All are hard constants:

* `K_HYP = 4`
* `HYP_WEIGHT_FLOOR = 0.01 / K_HYP`
* `B_BINS = 48`
* `T_SLICES = 5`
* `SIGMA_POINTS = 2 * D_DESKEW + 1 = 45`
* `N_POINTS_CAP = 8192`

Any reduction of these budgets must be done by an explicit ApproxOp and logged.

### 1.4 No multipaths / no fallbacks (hard)

* No “GPU if available else CPU”.
* No alternate runtime implementations in codepaths.
* Backend choices are explicit in configuration and in the runtime manifest.

### 1.5 No hidden iteration (hard, precisely defined)

Disallowed inside any single operator call:

* data-dependent solver loops (Newton/CG/LS until tolerance, line search, adaptive iterations).

Allowed:

* repeating the fixed pipeline over time (time advances).
* fixed-size loops with compile-time constant bounds (`B_BINS`, `SIGMA_POINTS`, `T_SLICES`) with no early exit.

---

## 2) Core Data Structures (float64)

### 2.1 Gaussian belief on augmented tangent (information form)

```text
BeliefGaussianInfo:
  chart_id: str            # must be "GC-RIGHT-01"
  anchor_id: str           # local chart instance id (stable)
  X_anchor: SE3            # anchor pose
  stamp_sec: float

  z_lin: (D_Z,)            # linearization point in chart coordinates
  L: (D_Z, D_Z)            # symmetric PSD/PD information matrix
  h: (D_Z,)                # information vector

  cert: CertBundle
```

Interpretation:

* MAP increment is defined by a **fixed lifted SPD solve** (no conditional SPD checks):
  [
  \delta z^* = (L + \varepsilon_{lift} I)^{-1} h
  ]
  where `eps_lift` is a manifest constant. This is the *declared solve*.

### 2.2 Certificates + expected effect (required everywhere)

```text
CertBundle:
  chart_id: str
  anchor_id: str

  exact: bool
  approximation_triggers: list[str]     # always present for ApproxOps (may be empty for ExactOps)
  frobenius_applied: bool               # True iff any trigger magnitude is nonzero

  conditioning:
    eig_min: float
    eig_max: float
    cond: float
    near_null_count: int

  support:
    ess_total: float
    support_frac: float                 # retained_mass / total_mass (continuous)

  mismatch:
    nll_per_ess: float                  # continuous proxy; may be 0 if undefined
    directional_score: float            # continuous proxy; may be 0 if undefined

  excitation:
    dt_effect: float                    # continuous
    extrinsic_effect: float             # continuous

  influence:
    lift_strength: float                # eps_lift * D_Z (always)
    psd_projection_delta: float         # ||M_proj - sym(M)||_F (always computed)
    mass_epsilon_ratio: float           # eps_mass / (mass + eps_mass) (continuous)
    anchor_drift_rho: float             # continuous reanchor strength in [0,1]
    dt_scale: float                     # in [0,1]
    extrinsic_scale: float              # in [0,1]
    trust_alpha: float                  # fusion scale alpha in [alpha_min, alpha_max]

ExpectedEffect:
  objective_name: str
  predicted: float
  realized: float | null
```

Downstream rule:

* Consumers may only **scale** influence continuously using certificates (e.g., `trust_alpha`, `dt_scale`), never branch/skip.

---

## 3) Branch-Free Numeric Primitives (Library Contracts)

All of these are **total functions** (always run) and return a certificate magnitude that can be exactly zero if no change occurred.

### 3.1 ApproxOp: `Symmetrize`

Always compute:

* `M_sym = 0.5*(M + M^T)`
* magnitude `sym_delta = ||M_sym - M||_F`

Trigger is present always; `frobenius_applied = (sym_delta > 0)`.

### 3.2 ApproxOp: `DomainProjectionPSD`

Always compute:

* `M_sym = Symmetrize(M)`
* eigendecomp of `M_sym`
* `vals_clamped = max(vals, eps_psd)` with fixed `eps_psd=1e-12`
* `M_psd = V diag(vals_clamped) V^T`
* projection magnitude:
  [
  \Delta_{psd} = |M_{psd} - M_{sym}|_F
  ]
* conditioning from `vals_clamped`

No conditional “only if needed”; clamp always executed.

### 3.3 ExactOp: `SPDCholeskySolveLifted`

Always solve:

* ((L + eps_lift I)x = b), with fixed `eps_lift` from manifest.
* `lift_strength = eps_lift * D_Z` recorded always.

No alternate solvers.

### 3.4 Total function: `InvMass`

Always compute:

* `inv_mass(m) = 1 / (m + eps_mass)` with fixed `eps_mass` in manifest.
* `mass_epsilon_ratio = eps_mass / (m + eps_mass)` recorded always.

This removes all division-by-zero gating.

### 3.5 Total function: `Clamp`

Always compute:

* `Clamp(x, lo, hi) = min(max(x, lo), hi)`
* `clamp_delta = |Clamp(x)-x|` recorded always.

No conditional execution.

---

## 4) Core Structures for Mapping and Binning

### 4.1 Bin atlas

```text
BinAtlas:
  dirs: (B_BINS, 3)        # fixed unit vectors
```

### 4.2 Map bin stats (additive sufficient stats)

```text
MapBinStats:
  S_dir: (3,)
  N_dir: float
  N_pos: float
  sum_p: (3,)
  sum_ppT: (3, 3)
```

Derived values are computed using **InvMass** (no if):

* `inv_N_dir = 1/(N_dir + eps_mass)`
* `Rbar = ||S_dir|| * inv_N_dir` (well-defined)
* `mu_dir = S_dir / (||S_dir|| + eps_mass)` (well-defined)
* `inv_N_pos = 1/(N_pos + eps_mass)`
* `c = sum_p * inv_N_pos`
* `Sigma_c_raw = (sum_ppT * inv_N_pos) - c c^T`
* `Sigma_c = DomainProjectionPSD(Sigma_c_raw).M_psd` (always computed)

### 4.3 Deskewed point

```text
DeskewedPoint:
  p_mean: (3,)
  p_cov: (3, 3)
  time_sec: float
  weight: float
```

### 4.4 Scan bin stats

```text
ScanBinStats:
  N: (B_BINS,)
  s_dir: (B_BINS, 3)
  p_bar: (B_BINS, 3)
  Sigma_p: (B_BINS, 3, 3)
  kappa_scan: (B_BINS,)
```

No `support` boolean is used for control. Zero-mass bins contribute weight 0 automatically through `N[b]`.

---

## 5) ApproxOps and ExactOps (All Fixed-Cost, Branch-Free)

All operators return `(result, cert, expected_effect)`.

### 5.1 ApproxOp: `PointBudgetResample`

Always produce an output of size `<= N_POINTS_CAP` by deterministic resampling.

* If `N_raw <= N_POINTS_CAP`, resampling returns the same set (identity outcome) with `support_frac=1`.
* If `N_raw > N_POINTS_CAP`, resampling drops points and renormalizes weights so retained mass is preserved.

Certificate:

* `support_frac` computed always
* `approximation_triggers` always includes `"PointBudgetResample"`
* `frobenius_applied` determined by `1 - support_frac`

ExpectedEffect:

* `objective_name="predicted_mass_retention"`
* `predicted=support_frac`

### 5.2 ExactOp: `PredictDiffusion`

Always predicts between timestamps at fixed cost.

Inputs:

* `belief_prev`
* PSD diffusion `Q: (22,22)` from process noise module
* `dt_sec`

Construction (always):

1. Convert info to moments:

   * Solve `mu = (L + eps_lift I)^{-1} h` using `SPDCholeskySolveLifted`
   * Compute `Sigma` as ((L + eps_lift I)^{-1}) via multiple solves (fixed cost `D_Z` solves)
2. Predict:

   * `mu_pred = mu`
   * `Sigma_pred_raw = Sigma + Q * dt_sec`
3. `Sigma_pred = DomainProjectionPSD(Sigma_pred_raw).M_psd` (always)
4. Convert back:

   * `L_pred_raw = inverse(Sigma_pred)` computed by Cholesky solves (fixed cost)
   * `L_pred = DomainProjectionPSD(L_pred_raw).M_psd` (always)
   * `h_pred = L_pred @ mu_pred`

Certificate includes:

* lift_strength
* psd_projection_delta for `Sigma_pred` and `L_pred`
* conditioning

ExpectedEffect:

* `objective_name="predicted_trace_increase"`
* `predicted=trace(Q)*dt_sec`

### 5.3 ApproxOp: `DeskewUTMomentMatch`

Always uses exactly `T_SLICES*SIGMA_POINTS` evaluations.

Inputs:

* `belief_pred`
* points (budgeted)
* timing model

Outputs:

* `deskewed_points` with `(p_mean, p_cov)` for each point
* excitation scalars computed continuously from UT contributions:

  * `dt_effect >= 0`
  * `extrinsic_effect >= 0`
* also output `ut_cache` (required) containing per-slice, per-sigma transforms and deltas for reuse downstream (fixed size)

Certificate:

* triggers include `"DeskewUTMomentMatch"`
* excitation fields filled always

ExpectedEffect:

* `objective_name="predicted_deskew_cov_trace"`
* `predicted=weighted_mean_i trace(p_cov_i)`

### 5.4 ApproxOp: `BinSoftAssign`

Always compute responsibilities:
[
r_{i,b} = \frac{\exp(\langle u_i, d_b\rangle/\tau)}{\sum_{b'}\exp(\langle u_i, d_{b'}\rangle/\tau)}
]
where `tau = tau_soft_assign` is constant.

Certificate:

* directional_score = mean responsibility entropy (continuous)

ExpectedEffect:

* `objective_name="predicted_assignment_entropy"`
* `predicted=directional_score`

### 5.5 ApproxOp: `ScanBinMomentMatch`

Always compute:

* `N[b] = Σ_i w_i r_{i,b}`
* `p_bar[b] = (Σ_i w_i r_{i,b} p_mean_i) * InvMass(N[b])`
* `Sigma_p[b] = within-bin scatter + Σ_i w_i r_{i,b} p_cov_i`, then `DomainProjectionPSD` always applied to each `Sigma_p[b]`

Certificate includes:

* `support.ess_total` always
* `support.support_frac` based on effective mass distribution (continuous)
* `psd_projection_delta` aggregated

ExpectedEffect:

* `objective_name="predicted_scatter_trace"`
* `predicted=Σ_b N[b] trace(Sigma_p[b]) / (Σ_b N[b] + eps_mass)`

### 5.6 ApproxOp: `KappaFromResultant:v2_single_formula`

No piecewise, no iteration.

Input:

* `Rbar_raw = ||S_dir|| * InvMass(N_dir)`

Always compute:

* `Rbar = Clamp(Rbar_raw, 0, 1 - eps_r)` with fixed `eps_r`
* `den = 1 - Rbar^2 + eps_den` with fixed `eps_den`
* `kappa = Rbar * (3 - Rbar^2) / den`

Certificate:

* clamp_delta recorded in influence fields
* `frobenius_applied` based on clamp_delta + denom regularization magnitude

ExpectedEffect:

* `objective_name="predicted_kappa_magnitude"`
* `predicted=kappa`

### 5.7 ExactOp: `WahbaSVD`

Always compute Wahba matrix:
[
M = \sum_b w_b, \mu_{map}[b]\mu_{scan}[b]^\top,\quad
w_b = N[b]\kappa_{map}[b]\kappa_{scan}[b]
]
No bin skipping; if `N[b]=0`, contribution is exactly zero.

Output `R_hat` from SVD.

Certificate:

* conditioning based on singular values

ExpectedEffect:

* `objective_name="predicted_wahba_rank_proxy"`
* `predicted=σ_2/σ_1` (continuous)

### 5.8 ExactOp: `TranslationWLS`

Always compute per-bin:

* `Sigma_b_raw = Sigma_c_map[b] + R Sigma_p[b] R^T + Sigma_meas`
* `Sigma_b = DomainProjectionPSD(Sigma_b_raw).M_psd` always
* WLS normal equations accumulated over all bins with weight `w_b` (zero weight bins contribute nothing)
* solve for `t_hat` by SPDCholeskySolveLifted on `L_tt` (always lifted)

Outputs:

* `t_hat`, `L_tt`

Certificate:

* aggregated PSD projection deltas and conditioning

ExpectedEffect:

* `objective_name="predicted_translation_info_trace"`
* `predicted=trace(L_tt)`

### 5.9 ApproxOp: `LidarQuadraticEvidence`

Produces `BeliefGaussianInfo evidence` on full 22D tangent at fixed cost, reusing `ut_cache`.

Inputs:

* `belief_pred`
* `scan_bins`, `map_bins`
* `R_hat`, `t_hat`
* `ut_cache` from DeskewUTMomentMatch
* constants `c_dt`, `c_ex` from manifest

Outputs:

* `evidence` with same chart/anchor/z_lin
* `L_lidar` PSD (DomainProjectionPSD always)
* `h_lidar`

Branch-free coupling rule:

* define continuous excitation scales:
  [
  s_{dt} = \frac{dt_effect}{dt_effect + c_{dt}},\quad
  s_{ex} = \frac{extrinsic_effect}{extrinsic_effect + c_{ex}}
  ]
* `dt_scale = s_dt`, `extrinsic_scale = s_ex` recorded always
* apply scales by multiplication (always):

  * blocks involving index 15 multiplied by `s_dt`
  * blocks involving indices 16..21 multiplied by `s_ex`

Quadratic construction (fixed cost, deterministic):

1. Compute target pose increment `δξ_pose*` from `(R_hat, t_hat)` in right perturbation at `z_lin`.
2. Build `δz*` by:

   * pose slice = `δξ_pose*`
   * all other slices start at 0
   * compute continuous least-squares coupling for `[δΔt, δξLI]` using fixed design derived from `ut_cache`:

     * form `J_u` (pose residual sensitivity to those slices) by closed-form regression over all UT samples (one normal equation solve with lift)
     * solve `δu* = (J_u^T J_u + eps_lift I)^{-1} J_u^T δξ_pose*`
     * insert δu* into indices 15 and 16..21
3. Build `L_lidar_raw` from UT regression with fixed feature map:

   * Use all UT deltas `δz^(s)` to form a fixed quadratic feature vector φ(δz) containing:

     * all 22 linear terms
     * all 253 symmetric quadratic terms (upper triangular of δzδz^T)
   * Solve normal equations once (lifted) to get quadratic coefficients, assemble symmetric `L_lidar_raw`
4. `L_lidar = DomainProjectionPSD(L_lidar_raw).M_psd` always
5. Apply excitation scaling to relevant blocks (always)
6. Set `h_lidar = L_lidar @ δz*`

Certificate:

* mismatch proxies computed continuously from residual surrogates in UT regression
* excitation, dt_scale, extrinsic_scale always filled
* psd_projection_delta always filled

ExpectedEffect:

* `objective_name="predicted_quadratic_nll_decrease"`
* `predicted=0.5 * δz*^T L_lidar δz*`

### 5.10 ApproxOp: `FusionScaleFromCertificates`

Always computes `alpha` (trust) as a continuous function.

Inputs:

* `cert_evidence`, `cert_belief`
* constants `alpha_min`, `alpha_max`, `c0`, `kappa_scale`

Always compute:
[
s = \exp(-nll_per_ess)\cdot \frac{1}{1 + cond/c0}
]
[
\alpha = Clamp(kappa_scale \cdot s,\ alpha_min,\ alpha_max)
]
Record `trust_alpha = alpha` always.

ExpectedEffect:

* `objective_name="predicted_alpha"`
* `predicted=alpha`

### 5.11 ExactOp: `InfoFusionAdditive`

Always compute:

* `L_post_raw = L_pred + alpha * L_evidence`
* `h_post = h_pred + alpha * h_evidence`
* `L_post = DomainProjectionPSD(L_post_raw).M_psd` always (projection magnitude recorded)
* return belief_post

ExpectedEffect:

* `objective_name="predicted_info_trace_increase"`
* `predicted=trace(alpha * L_evidence)`

### 5.12 ApproxOp: `PoseUpdateFrobeniusRecompose`

Always recomposes with a continuous Frobenius strength.

Inputs:

* `belief_post`
* `frobenius_strength = min(1, total_trigger_magnitude / (total_trigger_magnitude + c_frob))`
  where `total_trigger_magnitude` is the sum of `psd_projection_delta`, clamp deltas, mass eps ratios, etc., and `c_frob` is constant.

Always:

1. Solve `δz_map = (L + eps_lift I)^{-1} h`
2. Apply right perturbation update to pose using BCH3 correction blended by `frobenius_strength`:

   * compute BCH3 corrected increment `δξ_BCH3`
   * apply `δξ_apply = (1 - frobenius_strength) * δξ_linear + frobenius_strength * δξ_BCH3`
3. Euclidean slices add: `u += δu` (always)

Certificate:

* `anchor_drift_rho` unaffected here
* record `frobenius_strength` as continuous effect magnitude

ExpectedEffect:

* `objective_name="predicted_step_norm"`
* `predicted=||δz_map||`

### 5.13 ApproxOp: `PoseCovInflationPushforward`

Always updates map increments with continuous inflation computed from pose covariance.

Inputs:

* belief_post (convert to covariance once with lifted solves)
* scan bins

Always:

* inflate scan covariances by first-order pose covariance contribution
* update `ΔMapBinStats` additively
* apply DomainProjectionPSD to any derived covariance used internally (always)

ExpectedEffect:

* `objective_name="predicted_map_update_norm"`
* `predicted=norm(ΔMapBinStats)` (fixed norm)

### 5.14 ApproxOp: `AnchorDriftUpdate` (Continuous local chart maintenance)

Replaces threshold “AnchorPromote”. Always runs, no discrete switching.

Inputs:

* `belief_post`
* constants: `m0`, `a`, `c0`, `b`, `s0`, `d` (manifest)
* uses current mean increment norm and cert fields

Always compute mean pose increment norm:

* `mu = (L + eps_lift I)^{-1} h`
* `pose_norm = ||mu[0:6]||`

Compute continuous reanchor strength:
[
\rho = \sigma(a(pose_norm - m0)) \cdot \sigma(b(\log(cond) - \log(c0))) \cdot \sigma(d(s0 - support_frac))
]
with `σ(x)=1/(1+exp(-x))`.

Always update anchor on SE(3):

* `X_mean = X_anchor · Exp(mu_pose)`
* `Δ = Log(X_anchor^{-1} X_mean)`
* `X_anchor_next = X_anchor · Exp( ρ * Δ )`

Always transport belief to the new anchor using first-order adjoint pushforward (no gating):

* update `X_anchor` to `X_anchor_next`
* set `z_lin` to zero vector
* transport `(L,h)` accordingly with a fixed first-order `ChartTransportRightSE3Continuous` step (internal to this op)

Certificate:

* `anchor_drift_rho = ρ`
* expected effect:

  * `objective_name="predicted_linearization_error_reduction_proxy"`
  * `predicted=ρ * pose_norm`

### 5.15 ApproxOp: `HypothesisBarycenterProjection`

Always produces a single belief for publishing.

Inputs:

* hypotheses `{(w_j, belief_j)}`

Always enforce weight floor continuously:

* `w'_j = max(w_j, HYP_WEIGHT_FLOOR)` then renormalize
* record floor adjustment magnitude in certificate

Always barycenter in info form:

* `L_out_raw = Σ w'_j L_j`
* `h_out = Σ w'_j h_j`
* `L_out = DomainProjectionPSD(L_out_raw).M_psd` always

ExpectedEffect:

* `objective_name="predicted_projection_spread_proxy"`
* `predicted` computed as weighted variance of mean increments (fixed formula with InvMass)

---

## 6) Runtime Manifest (Required)

```text
RuntimeManifest:
  chart_id: "GC-RIGHT-01"

  D_Z: 22
  D_DESKEW: 22
  K_HYP: 4
  HYP_WEIGHT_FLOOR: 0.0025
  B_BINS: 48
  T_SLICES: 5
  SIGMA_POINTS: 45
  N_POINTS_CAP: 8192

  tau_soft_assign: float

  eps_psd: 1e-12
  eps_lift: float
  eps_mass: float
  eps_r: float
  eps_den: float

  alpha_min: float
  alpha_max: float
  kappa_scale: float
  c0_cond: float

  c_dt: float
  c_ex: float
  c_frob: float

  anchor_drift_params:
    m0: float
    a: float
    c0: float
    b: float
    s0: float
    d: float

  backends: dict[str, str]      # explicit and singular (no fallback)
```

---

## 7) Deterministic Per-Scan Execution Order (Per Hypothesis)

1. `PointBudgetResample`
2. `PredictDiffusion`
3. `DeskewUTMomentMatch` (produces `ut_cache`)
4. `BinSoftAssign`
5. `ScanBinMomentMatch`
6. `KappaFromResultant:v2_single_formula` (map and scan)
7. `WahbaSVD`
8. `TranslationWLS`
9. `LidarQuadraticEvidence` (reuses `ut_cache`)
10. `FusionScaleFromCertificates` (alpha)
11. `InfoFusionAdditive`
12. `PoseUpdateFrobeniusRecompose` (continuous Frobenius strength)
13. `PoseCovInflationPushforward`
14. `AnchorDriftUpdate` (continuous reanchoring; always)

After all hypotheses:
15. `HypothesisBarycenterProjection`

All steps run every time; influence may go to ~0 smoothly. No gates.

---

## 8) How This Solves the Original Backend Audit Issues (by construction)

1. Delta accumulation drift: no frame-to-frame delta integration updates; effects enter via fixed operators with continuous scaling.
2. Double counting: evidence enters exactly once through additive info fusion; no separate prior inflation with the same covariance.
3. Non-SPD joint: no block joint construction; PSD projection is a declared ApproxOp always applied; solves always lifted.
4. `R_imu` unused: IMU uncertainty enters deskew UT and induced point covariances, and therefore the quadratic evidence.
5. Round-trip instability: conversions are explicit, fixed, and accounted for with projection deltas and conditioning.
6. JAX NaN propagation: DomainProjectionPSD + lifted solve defines the domain; NaNs are prevented by construction and logged via projection deltas.
7. Unbounded growth: prediction is dt-scaled diffusion; no unscaled accumulation.
8. Residual accumulation: process noise adaptation is required to be forgetful; all “mass eps” and trust effects are continuous and logged.
9. Framework mix: moment matching is confined to deskew/binning operators; fusion is pure info addition; no conflation of posterior spread with measurement noise.

---

Th