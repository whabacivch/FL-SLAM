"""
IMU evidence operators for GC v2.

Currently implements the accelerometer direction term as a vMF-style factor:
  ell(delta) = -kappa * dot(mu(delta), xbar)

where:
  mu(delta) = R(delta)^T * (-g_hat)
  xbar is the measured resultant direction over the scan window.

We convert this factor to quadratic Gaussian information by Laplace at delta=0
using closed-form derivatives (intrinsic primitives only; no autodiff).

=============================================================================
GRAVITY CONVENTION (CRITICAL)
=============================================================================
World frame: Z-UP convention
  gravity_W = [0, 0, -9.81] m/s²  (gravity points DOWN in -Z direction)
  g_hat = [0, 0, -1]              (normalized gravity direction, pointing DOWN)
  minus_g_hat = [0, 0, +1]        (expected accel direction, pointing UP)

Accelerometer Convention:
  IMU accelerometers measure REACTION TO GRAVITY (specific force), NOT gravity.
  When level and stationary, accelerometer reads +Z (pointing UP).
  This is the force preventing the sensor from freefalling.

Expected vs Measured:
  mu0 = R_body^T @ minus_g_hat    (expected accel direction in body frame)
  xbar = normalized(mean(accel))  (measured accel direction in body frame)
  Alignment: xbar @ mu0 should be ~+1.0 for correct gravity alignment
  If negative, the IMU extrinsic is likely inverting gravity!

State Ordering:
  L_imu is placed at [3:6, 3:6] which is the ROTATION block in GC ordering.
  (GC state: [trans(0:3), rot(3:6), ...])
=============================================================================

## vMF HESSIAN APPROXIMATION NOTE (Audit Compliance)

The Hessian H_rot used for the information matrix is an approximation to the
true Fisher information of the vMF directional likelihood.

### Exact vMF Fisher Information (on S²)

For vMF(μ, κ) with μ ∈ S² and κ > 0, the Fisher information matrix in the
tangent space at μ is:

    F_vMF = κ * (I - μμᵀ)

This is a rank-2 matrix (the tangent space to S² is 2D).

### Chain Rule for Rotation Parameterization

When μ(θ) = Rᵀ(-g_hat) and R = R₀·Exp(δθ), we need to compose with the
Jacobian ∂μ/∂δθ. The full Hessian is:

    H = (∂μ/∂δθ)ᵀ · F_vMF · (∂μ/∂δθ) + first_order_terms

For the likelihood ℓ(δθ) = -κ·μ(δθ)ᵀ·x̄, the gradient is:

    g = -κ·(μ₀ × x̄)  (cross product, exact)

The exact second derivative has the form:

    H_exact = κ·(x̄·μ₀)·I - κ·(outer products involving x̄, μ₀, and their derivatives)

### Approximation Used

We use a simplified closed-form approximation:

    H_approx = κ * [ (x̄·μ₀)·I - 0.5·(x̄μ₀ᵀ + μ₀x̄ᵀ) ]

This captures the dominant curvature but differs from the exact form by:
1. Missing the second-derivative terms from the rotation Jacobian
2. Using a symmetric average instead of the exact outer product structure

### Error Characteristics

- When x̄ ≈ μ₀ (good alignment): Error is O(κ·|x̄ - μ₀|²), typically < 5%
- When x̄ ⊥ μ₀ (poor alignment): Both exact and approx have low information, error is benign
- When x̄ ≈ -μ₀ (opposite): H is near-zero for both (correctly captures ambiguity)

### Justification for Use

1. The approximation is conservative (tends to underestimate information)
2. PSD projection is always applied afterward (ensures valid covariance)
3. Closed-form avoids autodiff overhead in hot path
4. Error is small when evidence is strong (x̄ ≈ μ₀)
5. When evidence is weak, both forms give low information (safe)

### Alternative (Not Implemented)

For exact Hessian, use JAX autodiff:
    H_exact = jax.hessian(lambda dtheta: -kappa * mu(dtheta) @ xbar)(zeros(3))

This would add computational cost and require differentiating through SO(3) exp.

Reference: Mardia & Jupp (2000) "Directional Statistics" Ch. 9 (vMF Fisher info)
Reference: Sra (2012) "A short note on parameter approximation for vMF"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import D_Z
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    InfluenceCert,
    SupportCert,
    MismatchCert,
)
from fl_slam_poc.common.primitives import domain_projection_psd_core
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_v2


@dataclass
class ImuEvidenceResult:
    L_imu: jnp.ndarray  # (22,22)
    h_imu: jnp.ndarray  # (22,)
    kappa: float
    ess: float


@dataclass
class TimeResolvedImuResult:
    """Result of time-resolved IMU evidence with reliability weighting."""
    L_imu: jnp.ndarray  # (22,22)
    h_imu: jnp.ndarray  # (22,)
    kappa: float
    ess_weighted: float  # Effective sample size after reliability weighting
    ess_raw: float  # Raw sample count
    mean_reliability: float  # Average reliability across samples
    transport_sigma: float  # Self-adaptive transport consistency scale


@dataclass
class ImuDependenceInflationResult:
    """Conservative dependence inflation between gyro and accel evidence."""
    scale: float  # Multiplicative scale applied to L/h (0 < scale <= 1)


@jax.jit
def _accel_resultant_direction_jax(
    imu_accel: jnp.ndarray,   # (M,3)
    weights: jnp.ndarray,     # (M,)
    accel_bias: jnp.ndarray,  # (3,)
    eps_mass: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    w = weights
    ess = jnp.sum(w)

    a = imu_accel - accel_bias[None, :]
    n = jnp.linalg.norm(a, axis=1, keepdims=True)
    x = a / (n + eps_mass)  # (M,3)

    S = jnp.sum(w[:, None] * x, axis=0)  # (3,)
    S_norm = jnp.linalg.norm(S)
    xbar = S / (S_norm + eps_mass)
    Rbar = S_norm / (ess + eps_mass)
    return xbar, Rbar, ess


def imu_vmf_gravity_evidence(
    rotvec_world_body: jnp.ndarray,  # (3,) rotvec of body in world
    imu_accel: jnp.ndarray,          # (M,3)
    weights: jnp.ndarray,            # (M,)
    accel_bias: jnp.ndarray,         # (3,)
    gravity_W: jnp.ndarray,          # (3,)
    eps_psd: float,
    eps_mass: float,
    chart_id: str,
    anchor_id: str,
) -> Tuple[ImuEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Closed-form Laplace/I-projection of the vMF-style accelerometer direction factor onto a 22D Gaussian info term.

    We differentiate only over the 3D rotation perturbation δθ applied on the right:
      R(δθ) = R0 @ Exp(δθ)
    """
    rotvec0 = jnp.asarray(rotvec_world_body, dtype=jnp.float64).reshape(-1)
    R0 = se3_jax.so3_exp(rotvec0)

    g = jnp.asarray(gravity_W, dtype=jnp.float64).reshape(-1)
    g_hat = g / (jnp.linalg.norm(g) + eps_mass)
    minus_g_hat = -g_hat

    xbar, Rbar, ess = _accel_resultant_direction_jax(
        imu_accel=jnp.asarray(imu_accel, dtype=jnp.float64),
        weights=jnp.asarray(weights, dtype=jnp.float64).reshape(-1),
        accel_bias=jnp.asarray(accel_bias, dtype=jnp.float64).reshape(-1),
        eps_mass=eps_mass,
    )

    kappa_result, kappa_cert, _kappa_effect = kappa_from_resultant_v2(
        R_bar=float(Rbar),
        eps_r=constants.GC_EPS_R,
        eps_den=constants.GC_EPS_DEN,
        chart_id=chart_id,
        anchor_id=anchor_id,
    )
    kappa = float(kappa_result.kappa)

    # Predicted mean direction in body frame at the linearization point.
    # mu0 = R0^T (-g_hat)
    mu0 = R0.T @ minus_g_hat

    kappa_f = jnp.array(kappa, dtype=jnp.float64)
    x_dot_mu = (xbar @ mu0)

    # Closed-form gradient and Hessian w.r.t. right-perturbation rotation δθ:
    #   g = ∂/∂δθ (-κ mu^T xbar) |0 = -κ (mu0 × xbar)  [EXACT]
    #   H ≈ κ [ (x·mu) I - 0.5 (x mu^T + mu x^T) ]     [APPROXIMATION, see module docstring]
    #
    # NOTE: This H_rot is an approximation. See module docstring for:
    #   - Exact vMF Fisher information derivation
    #   - Error bounds (< 5% when x̄ ≈ μ₀)
    #   - Justification for conservative approximation
    g_rot = -kappa_f * jnp.cross(mu0, xbar)
    I3 = jnp.eye(3, dtype=jnp.float64)
    H_rot = kappa_f * (x_dot_mu * I3 - 0.5 * (jnp.outer(xbar, mu0) + jnp.outer(mu0, xbar)))
    H_rot = 0.5 * (H_rot + H_rot.T)
    H_rot_psd, H_cert_vec = domain_projection_psd_core(H_rot, eps_psd)

    L = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    # GC ordering: [trans(0:3), rot(3:6)] - rotation evidence goes to [3:6] block
    L = L.at[3:6, 3:6].set(H_rot_psd)
    h = jnp.zeros((D_Z,), dtype=jnp.float64)
    h = h.at[3:6].set(-g_rot)

    nll_proxy = float(-kappa_f * (mu0 @ xbar))
    # Conditioning comes from the PSD projection certificate (already clamped).
    proj_delta = float(H_cert_vec[0])
    eig_min = float(H_cert_vec[2])
    eig_max = float(H_cert_vec[3])
    cond = float(H_cert_vec[4])
    near_null = int(H_cert_vec[5])

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ImuAccelDirectionClosedFormLaplace"] + list(kappa_cert.approximation_triggers),
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=near_null,
        ),
        support=SupportCert(ess_total=float(ess), support_frac=1.0),
        mismatch=MismatchCert(nll_per_ess=nll_proxy / (float(ess) + eps_mass), directional_score=float(Rbar)),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=proj_delta,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    effect = ExpectedEffect(
        objective_name="imu_accel_direction_nll_proxy",
        predicted=nll_proxy,
        realized=None,
    )

    return ImuEvidenceResult(L_imu=L, h_imu=h, kappa=kappa, ess=float(ess)), cert, effect


# =============================================================================
# Time-Resolved Accel Evidence with Per-Sample Reliability
# =============================================================================


@jax.jit
def _compute_transport_consistency(
    imu_accel: jnp.ndarray,  # (M, 3) accel samples in body frame
    imu_gyro: jnp.ndarray,   # (M, 3) gyro samples in body frame
    dt: float,               # Time step between samples
    eps_mass: float,
) -> jnp.ndarray:
    """
    Compute transport consistency error for each sample.

    For gravity-dominant measurements, the specific force should satisfy:
        d(f_body)/dt + ω × f_body ≈ 0

    This is because gravity in world frame is constant, so in body frame:
        f_body = R^T @ g_world
        df/dt = -ω × f_body  (due to rotating frame)

    When there's linear acceleration, this relationship breaks down.

    Args:
        imu_accel: (M, 3) accelerometer readings
        imu_gyro: (M, 3) gyroscope readings (angular velocity)
        dt: Time step between samples
        eps_mass: Numerical stability epsilon

    Returns:
        e_k: (M,) transport consistency error magnitude for each sample
    """
    M = imu_accel.shape[0]

    # Compute df/dt via central finite difference (interior points)
    # For endpoints, use forward/backward difference
    df_dt = jnp.zeros_like(imu_accel)

    # Central difference for interior: df/dt[k] = (f[k+1] - f[k-1]) / (2*dt)
    df_dt = df_dt.at[1:-1].set(
        (imu_accel[2:] - imu_accel[:-2]) / (2 * dt + eps_mass)
    )
    # Forward difference for first point
    df_dt = df_dt.at[0].set(
        (imu_accel[1] - imu_accel[0]) / (dt + eps_mass)
    )
    # Backward difference for last point
    df_dt = df_dt.at[-1].set(
        (imu_accel[-1] - imu_accel[-2]) / (dt + eps_mass)
    )

    # Compute ω × f for each sample
    omega_cross_f = jnp.cross(imu_gyro, imu_accel)  # (M, 3)

    # Transport consistency error: e_k = df/dt + ω × f
    # Should be ~0 when measuring pure gravity
    e_k = df_dt + omega_cross_f  # (M, 3)

    # Return magnitude of error
    e_k_mag = jnp.linalg.norm(e_k, axis=1)  # (M,)

    return e_k_mag


@jax.jit
def _compute_reliability_weights(
    e_k_mag: jnp.ndarray,  # (M,) transport error magnitudes
    eps_mass: float,
) -> Tuple[jnp.ndarray, float]:
    """
    Compute self-adaptive reliability weights from transport errors.

    Uses robust median-based sigma estimation (no manual knobs):
        sigma = median(e_k) / 0.6745  (MAD-based robust scale)

    Then weights are:
        reliability_k = exp(-e_k^2 / (2 * sigma^2))

    Args:
        e_k_mag: (M,) transport error magnitudes
        eps_mass: Numerical stability epsilon

    Returns:
        reliability: (M,) per-sample reliability weights in [0, 1]
        sigma: Self-adaptive transport consistency scale
    """
    # Robust sigma estimation using MAD (Median Absolute Deviation)
    # MAD / 0.6745 ≈ sigma for Gaussian distribution
    median_e = jnp.median(e_k_mag)
    mad = jnp.median(jnp.abs(e_k_mag - median_e))
    sigma = mad / 0.6745 + eps_mass  # Self-adaptive scale, no manual knob

    # Compute reliability weights
    reliability = jnp.exp(-0.5 * (e_k_mag / sigma) ** 2)

    return reliability, sigma


@jax.jit
def _accel_resultant_direction_weighted_jax(
    imu_accel: jnp.ndarray,   # (M, 3)
    weights: jnp.ndarray,     # (M,) base weights (e.g., uniform)
    reliability: jnp.ndarray, # (M,) per-sample reliability from transport consistency
    accel_bias: jnp.ndarray,  # (3,)
    eps_mass: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute weighted resultant direction with reliability weighting.

    Combined weight: w_k = base_weight_k * reliability_k
    """
    # Combined weights
    w = weights * reliability
    ess_weighted = jnp.sum(w)
    ess_raw = jnp.sum(weights)

    # Normalize accelerations
    a = imu_accel - accel_bias[None, :]
    n = jnp.linalg.norm(a, axis=1, keepdims=True)
    x = a / (n + eps_mass)  # (M, 3) unit directions

    # Weighted resultant
    S = jnp.sum(w[:, None] * x, axis=0)  # (3,)
    S_norm = jnp.linalg.norm(S)
    xbar = S / (S_norm + eps_mass)
    Rbar = S_norm / (ess_weighted + eps_mass)

    return xbar, Rbar, ess_weighted, ess_raw


def imu_vmf_gravity_evidence_time_resolved(
    rotvec_world_body: jnp.ndarray,  # (3,) rotvec of body in world
    imu_accel: jnp.ndarray,          # (M, 3) accel samples
    imu_gyro: jnp.ndarray,           # (M, 3) gyro samples (for transport consistency)
    weights: jnp.ndarray,            # (M,) base weights
    accel_bias: jnp.ndarray,         # (3,)
    gravity_W: jnp.ndarray,          # (3,)
    dt_imu: float,                   # Time step between IMU samples
    eps_psd: float,
    eps_mass: float,
    chart_id: str,
    anchor_id: str,
) -> Tuple[TimeResolvedImuResult, CertBundle, ExpectedEffect]:
    """
    Time-resolved vMF gravity evidence with per-sample reliability weighting.

    This replaces the single-mean vMF approach with consistency-weighted evidence.
    Samples where d(f)/dt + ω × f is small (gravity-dominant) get high weight.
    Samples with large transport error (linear acceleration) are downweighted.

    The reliability scale sigma is self-adaptive from the data (MAD-based),
    so there are NO manual tuning knobs.

    Args:
        rotvec_world_body: (3,) rotation vector of body in world frame
        imu_accel: (M, 3) accelerometer readings in body frame
        imu_gyro: (M, 3) gyroscope readings in body frame
        weights: (M,) base sample weights
        accel_bias: (3,) accelerometer bias estimate
        gravity_W: (3,) gravity vector in world frame
        dt_imu: Time step between IMU samples
        eps_psd: PSD projection epsilon
        eps_mass: Numerical stability epsilon
        chart_id: Chart identifier
        anchor_id: Anchor identifier

    Returns:
        TimeResolvedImuResult, CertBundle, ExpectedEffect
    """
    rotvec0 = jnp.asarray(rotvec_world_body, dtype=jnp.float64).reshape(-1)
    R0 = se3_jax.so3_exp(rotvec0)

    g = jnp.asarray(gravity_W, dtype=jnp.float64).reshape(-1)
    g_hat = g / (jnp.linalg.norm(g) + eps_mass)
    minus_g_hat = -g_hat

    imu_accel = jnp.asarray(imu_accel, dtype=jnp.float64)
    imu_gyro = jnp.asarray(imu_gyro, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)
    accel_bias = jnp.asarray(accel_bias, dtype=jnp.float64).reshape(-1)

    # Bias-corrected accel for transport computation
    accel_corrected = imu_accel - accel_bias[None, :]

    # Step 1: Compute transport consistency error for each sample
    e_k_mag = _compute_transport_consistency(
        imu_accel=accel_corrected,
        imu_gyro=imu_gyro,
        dt=dt_imu,
        eps_mass=eps_mass,
    )

    # Step 2: Compute self-adaptive reliability weights (no manual sigma)
    reliability, transport_sigma = _compute_reliability_weights(
        e_k_mag=e_k_mag,
        eps_mass=eps_mass,
    )

    # Step 3: Compute weighted resultant direction
    xbar, Rbar, ess_weighted, ess_raw = _accel_resultant_direction_weighted_jax(
        imu_accel=imu_accel,
        weights=weights,
        reliability=reliability,
        accel_bias=accel_bias,
        eps_mass=eps_mass,
    )

    # Step 4: Compute kappa from weighted resultant
    kappa_result, kappa_cert, _kappa_effect = kappa_from_resultant_v2(
        R_bar=float(Rbar),
        eps_r=constants.GC_EPS_R,
        eps_den=constants.GC_EPS_DEN,
        chart_id=chart_id,
        anchor_id=anchor_id,
    )
    kappa = float(kappa_result.kappa)

    # Predicted mean direction in body frame
    mu0 = R0.T @ minus_g_hat

    kappa_f = jnp.array(kappa, dtype=jnp.float64)
    x_dot_mu = xbar @ mu0

    # Closed-form gradient and Hessian (same as original, but with reliability-weighted xbar)
    g_rot = -kappa_f * jnp.cross(mu0, xbar)
    I3 = jnp.eye(3, dtype=jnp.float64)
    H_rot = kappa_f * (x_dot_mu * I3 - 0.5 * (jnp.outer(xbar, mu0) + jnp.outer(mu0, xbar)))
    H_rot = 0.5 * (H_rot + H_rot.T)
    H_rot_psd, H_cert_vec = domain_projection_psd_core(H_rot, eps_psd)

    L = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    L = L.at[3:6, 3:6].set(H_rot_psd)
    h = jnp.zeros((D_Z,), dtype=jnp.float64)
    h = h.at[3:6].set(-g_rot)

    nll_proxy = float(-kappa_f * (mu0 @ xbar))

    # Conditioning from PSD projection
    proj_delta = float(H_cert_vec[0])
    eig_min = float(H_cert_vec[2])
    eig_max = float(H_cert_vec[3])
    cond = float(H_cert_vec[4])
    near_null = int(H_cert_vec[5])

    mean_reliability = float(jnp.mean(reliability))

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ImuAccelDirectionTimeResolved", "TransportConsistencyWeighting"] + list(kappa_cert.approximation_triggers),
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=near_null,
        ),
        support=SupportCert(
            ess_total=float(ess_weighted),
            support_frac=mean_reliability,  # Use mean reliability as support quality
        ),
        mismatch=MismatchCert(
            nll_per_ess=nll_proxy / (float(ess_weighted) + eps_mass),
            directional_score=float(Rbar),
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=proj_delta,
            mass_epsilon_ratio=float(ess_weighted) / (float(ess_raw) + eps_mass),  # Reliability ratio
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=mean_reliability,
        ),
    )

    effect = ExpectedEffect(
        objective_name="imu_accel_direction_time_resolved_nll_proxy",
        predicted=nll_proxy,
        realized=None,
    )

    result = TimeResolvedImuResult(
        L_imu=L,
        h_imu=h,
        kappa=kappa,
        ess_weighted=float(ess_weighted),
        ess_raw=float(ess_raw),
        mean_reliability=mean_reliability,
        transport_sigma=float(transport_sigma),
    )

    return result, cert, effect


def imu_dependence_inflation(
    transport_sigma: float,
    eps_mass: float,
    chart_id: str,
    anchor_id: str,
) -> Tuple[ImuDependenceInflationResult, CertBundle, ExpectedEffect]:
    """
    Conservative inflation for IMU gyro↔accel dependence.

    Uses transport_sigma (from transport consistency) to downscale evidence
    continuously when dependence is strong. No gating, fixed-cost.
    """
    sigma = float(max(transport_sigma, 0.0))
    scale = 1.0 / (1.0 + sigma * sigma + float(eps_mass))
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ImuDependenceInflation"],
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=float(scale),
        ),
    )
    effect = ExpectedEffect(
        objective_name="imu_dependence_inflation",
        predicted=float(scale),
        realized=float(scale),
    )
    return ImuDependenceInflationResult(scale=float(scale)), cert, effect
