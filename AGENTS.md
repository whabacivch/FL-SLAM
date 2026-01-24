# Impact Project_v1: Agent Instructions

These rules apply only to this project. Other projects have their own rules.

## Project Intent
- Build a Frobenius–Legendre compositional inference backend for dynamic SLAM.
- Implement and verify **Golden Child SLAM v2** as a strict, branch-free, fixed-cost SLAM backend.

## Target Endstate (GC v2 Full Implementation)

The current pipeline is LiDAR-only with fixed noise parameters. The target endstate implements the full spec from `docs/GOLDEN_CHILD_INTERFACE_SPEC.md` including:

### Adaptive Noise (Inverse-Wishart)
- **Process noise Q**: Per-block Inverse-Wishart states (rot, trans, vel, biases, dt, extrinsic) with conjugate updates from innovation residuals.
- **Measurement noise Σ**: IW states for gyro, accel, odom, and LiDAR measurement noise.
- **Initialization**: Datasheet priors with low pseudocounts (ν = p + 0.5) for fast adaptation.
  - IMU (ICM-40609): gyro σ² ≈ 8.7e-7 rad²/s², accel σ² ≈ 9.5e-5 m²/s⁴
  - LiDAR (Mid-360): range σ ≈ 2cm, angular σ ≈ 0.15°, combined ~1e-3 m²/axis
  - Odometry: use real 2D-aware covariances from `/odom` message (0.001 for x,y; 1e6 for z/roll/pitch)

### Sensor Evidence (IMU + Odom)
- **Accelerometer direction**: von Mises-Fisher (vMF) likelihood on S² with random concentration κ.
- **Gyro integration**: Gaussian likelihood with IW-adaptive Σg.
- **Odom partial observation**: Constrain `[x, y, yaw]` strongly, `[z, roll, pitch]` weakly; use message covariance or IW-adaptive.
- **Time offset warp**: Soft membership kernel based on Δt uncertainty (no hard window boundaries).
- **Per-scan evidence**: IMU + Odom + LiDAR evidence fused additively before fusion scale.

### Likelihood-Based Evidence (Laplace/I-Projection)
- **Replace UT regression**: Build explicit likelihood terms (vMF directional + Gaussian translational).
- **Laplace approximation**: Compute (g, H) at z_lin via JAX autodiff or closed-form exponential family Hessians.
- **Exponential family closed forms**: Gaussian H = Σ⁻¹, vMF H = κ(I - μμᵀ).

### Invariants Preserved
- No gating: κ adaptation is continuous via resultant statistics, not threshold-based.
- No fixed constants: all noise parameters are IW random variables with weak priors.
- Branch-free: IW updates happen every scan regardless of "convergence".

## Canonical References (Do Not Drift)
- Golden Child strict interface/spec anchor: `docs/GOLDEN_CHILD_INTERFACE_SPEC.md`
- Self-adaptive constraints: `docs/Self-Adaptive Systems Guide.md`
- Math reference: `docs/Comprehensive Information Geometry.md`
- Development log (required): `CHANGELOG.md`

## Quickstart and Validation
- Workspace: `fl_ws/` (ROS 2), package: `fl_ws/src/fl_slam_poc/`, tools: `tools/`
- Build: `cd fl_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select fl_slam_poc && source install/setup.bash`
- GC eval (primary): `bash tools/run_and_evaluate_gc.sh` (artifacts under `results/`)
- Legacy eval (if needed): `bash tools/run_and_evaluate.sh` (artifacts under `results/`)

## Non-Negotiable Invariants (GC v2)
- Closed-form-first: prefer analytic operators; only use solvers when no closed-form exists.
- Associative, order-robust fusion: when evidence is in-family and product-of-experts applies, fusion must be commutative/associative.
- Soft association only: no heuristic gating; use responsibilities from a declared generative model.
- Loop closure is late evidence: recomposition only (no iterative global optimization); any scope reduction must be an explicit approximation operator with an internal objective + predicted vs realized effect.
- Local modularity: state is an atlas of local modules; updates stay local by construction.
- Core must be Jacobian-free; Jacobians allowed only in sensor→evidence extraction and must be logged as `Linearization` (approx trigger) with Frobenius correction.
- Self-adaptive rules are hard constraints: no hard gates; startup is not a mode; constants are priors/budgets; approximate operators return (result, certificate, expected_effect) with no accept/reject branching.
- No hidden iteration: disallow data-dependent solver loops inside a single operator call (fixed-size loops only).
- Fail-fast on contract violations: chart id mismatches, dimension mismatches, and missing configured backends/sensors are hard errors.

## No Fallbacks / No Multi-Paths (Required)

The root failure mode to prevent is: *multiple math paths silently coexist*, making it impossible to know what behavior is actually running.

**Hard rules (enforced in review):**
- One runtime implementation per operator: delete duplicates or move them under `archive/` (not importable by installed entrypoints).
- No fallbacks: no environment-based selection, no `try/except ImportError` backends, no “GPU if available else CPU”.
- If variants are unavoidable, selection is explicit (`*_backend` param) and the node fails-fast at startup if unavailable.
- Nodes must emit a runtime manifest (log + status topic) listing resolved topics, enabled sensors, and selected backends/operators; tests must assert it.

## Frobenius Correction Policy (Mandatory When Applicable)
- If any approximation is introduced, Frobenius third-order correction MUST be applied.
  - Approximation triggers: linearization, mixture reduction, or out-of-family likelihood approximation.
  - Implementation rule: `approximation_triggered => apply_frobenius_retraction`.
  - Log each trigger and correction with the affected module id and operator name.
- If an operation is exact and in-family (e.g., Gaussian info fusion), correction is not applied.

## Evidence Fusion Rules
- Fusion/projection use Bregman barycenters (closed-form when available; otherwise geometry-defined solvers only).
- All sensor evidence is constructed via Laplace/I-projection at z_lin: compute (g, H) of joint NLL, project H to SPD.
- Noise covariances (Σg, Σa, Σlidar) are IW random variables, not fixed constants; use IW posterior mean as plug-in estimate.

## Implementation Conventions (Project-Specific)
- `fl_slam_poc/common/`: pure Python utilities (no ROS imports).
- `fl_slam_poc/frontend/`: sensor I/O + evidence extraction + utility nodes.
- `fl_slam_poc/backend/`: inference + fusion + kernels.

## Operator Reporting (Required)
- Every operator returns `(result, CertBundle, ExpectedEffect)` per `docs/GOLDEN_CHILD_INTERFACE_SPEC.md`.
- `CertBundle` must report: `exact`, `approximation_triggers`, `family_in/out` (where applicable), `closed_form`, `solver_used`, `frobenius_applied`.
- Enforcement rule: `approximation_triggers != ∅` ⇒ `frobenius_applied == True` (no exceptions).

## No Heuristics (Hard)
- No gating: no threshold/branch that changes model structure, association, or evidence inclusion.
- Domain constraints are allowed only as explicit DomainProjection logs (positivity/SPD/stable inversion safeguards).
- Compute budgeting is allowed only via explicit approximation operators; any mass drop must be preserved by renormalization or explicitly logged.
- Expected vs realized benefit is logged only in internal objectives (divergence/ELBO/etc.), never external metrics (ATE/RPE).

## Review Checklist (Use Before Merging Changes)
- Does the change preserve the non-negotiable invariants?
- Did the change introduce any new backend/operator variant or fallback path? If yes, remove it or move it under `archive/` and enforce explicit selection + fail-fast.
- Did the change introduce any approximation? If yes, is Frobenius correction applied and logged?
- Is evidence fusion performed by barycenters (closed-form when available)?
- Are loop closures handled by recomposition (not iterative global optimization)?
- Are responsibilities used for association (no gating)?

## Development Log (Required)
- Add a brief, timestamped entry to `CHANGELOG.md` for any material change in scope, assumptions, sensors, or model fidelity.
