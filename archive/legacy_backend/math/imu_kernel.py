"""
IMU Batched Projection Kernel.

Implements the core computation for Hellinger-Dirichlet IMU integration:
1. Batched sigma-support sampling across all anchors
2. IMU residual computation (exact Forster et al. model)
3. Hellinger-tilted weight computation
4. Global moment matching (e-projection)

Key implementation notes:
- Uses manifold retraction for pose components (SE(3) ⊕), not raw vector addition.
- Uses spherical-radial cubature weights (non-negative sigma-support weights),
  because sigma points are later treated as explicit mixture support.

Reference: Forster et al. (2017), Hellinger hierarchical construction
"""

# Import JAX from common initialization module (ensures single initialization)
from fl_slam_poc.common.jax_init import jax, jnp
from jax import lax
from jax.scipy.linalg import cholesky, solve_triangular
from typing import Tuple

# NOTE: JAX is already initialized and configured (GPU + x64) by common.jax_init.
# GPU availability check is deferred to backend_node._check_gpu_availability()
# to avoid checking at module import time.

from fl_slam_poc.common.geometry.se3_jax import so3_exp, so3_log, se3_plus
from fl_slam_poc.common.constants import (
    COV_REGULARIZATION_MIN as COV_REGULARIZATION,
    MIN_MIXTURE_WEIGHT,
    HELLINGER_TILT_WEIGHT,
)


def _apply_delta_state(xbar: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """
    Apply a tangent-space delta to a 15D state.

    Layout:
      [p(3), rotvec(3), v(3), b_g(3), b_a(3)]

    Pose uses SE(3) right-composition retraction; remaining terms are Euclidean.
    """
    pose = se3_plus(xbar[:6], delta[:6])
    rest = xbar[6:] + delta[6:]
    return jnp.concatenate([pose, rest], axis=0)


# =============================================================================
# IMU Residual Model (Contract B: raw IMU integration)
# =============================================================================


def _integrate_raw_imu(
    imu_stamps: jnp.ndarray,
    imu_accel: jnp.ndarray,
    imu_gyro: jnp.ndarray,
    imu_valid: jnp.ndarray,
    bias_g: jnp.ndarray,
    bias_a: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Integrate raw IMU measurements into (delta_p, delta_v, delta_rotvec).
    """

    def step(carry, inputs):
        delta_R, delta_v, delta_p, t_prev, prev_valid = carry
        t, a, g, valid = inputs

        dt = jnp.where(prev_valid & valid, t - t_prev, 0.0)
        dt = jnp.maximum(dt, 0.0)

        omega = (g - bias_g) * dt
        delta_R = delta_R @ so3_exp(omega)
        accel_corr = a - bias_a
        accel_rot = delta_R @ accel_corr
        # CRITICAL: Position update uses OLD velocity (tangent-space correct)
        # This is the Forster preintegration model: predict-then-retract
        delta_p = delta_p + delta_v * dt + 0.5 * accel_rot * dt * dt
        delta_v = delta_v + accel_rot * dt

        t_prev = jnp.where(valid, t, t_prev)
        prev_valid = valid
        return (delta_R, delta_v, delta_p, t_prev, prev_valid), None

    t0 = imu_stamps[0]
    carry0 = (jnp.eye(3, dtype=imu_stamps.dtype), jnp.zeros(3), jnp.zeros(3), t0, imu_valid[0])
    inputs = (imu_stamps[1:], imu_accel[1:], imu_gyro[1:], imu_valid[1:])
    carry_out, _ = lax.scan(step, carry0, inputs)
    delta_R, delta_v, delta_p, _, _ = carry_out
    delta_rotvec = so3_log(delta_R)
    return delta_p, delta_v, delta_rotvec


def imu_residual_from_raw(
    xi_anchor: jnp.ndarray,
    xi_current: jnp.ndarray,
    imu_stamps: jnp.ndarray,
    imu_accel: jnp.ndarray,
    imu_gyro: jnp.ndarray,
    imu_valid: jnp.ndarray,
    gravity: jnp.ndarray,
    dt_total: float,
) -> jnp.ndarray:
    """
    Compute IMU residual from raw IMU segment (Contract B).
    """
    p_i = xi_anchor[:3]
    rotvec_i = xi_anchor[3:6]
    v_i = xi_anchor[6:9]
    b_g_i = xi_anchor[9:12]
    b_a_i = xi_anchor[12:15]

    p_j = xi_current[:3]
    rotvec_j = xi_current[3:6]
    v_j = xi_current[6:9]

    delta_p_meas, delta_v_meas, delta_rotvec_meas = _integrate_raw_imu(
        imu_stamps, imu_accel, imu_gyro, imu_valid, b_g_i, b_a_i
    )

    R_i = so3_exp(rotvec_i)
    R_j = so3_exp(rotvec_j)

    delta_p_pred = R_i.T @ (p_j - p_i - v_i * dt_total - 0.5 * gravity * dt_total**2)
    delta_v_pred = R_i.T @ (v_j - v_i - gravity * dt_total)
    delta_R_pred = R_i.T @ R_j
    delta_rotvec_pred = so3_log(delta_R_pred)

    r_p = delta_p_meas - delta_p_pred
    r_v = delta_v_meas - delta_v_pred
    r_omega = delta_rotvec_meas - delta_rotvec_pred

    # IMPORTANT: Return residual in state-order tangent layout:
    # [p(3), rotvec(3), v(3)] to match the top-left 9x9 block of the 15D state.
    return jnp.concatenate([r_p, r_omega, r_v], axis=0)


# =============================================================================
# Hellinger Distance
# =============================================================================


def hellinger_squared_gaussian(
    mu1: jnp.ndarray,
    cov1: jnp.ndarray,
    mu2: jnp.ndarray,
    cov2: jnp.ndarray,
) -> jnp.ndarray:
    """
    Squared Hellinger distance H²(N₁, N₂).

    H²(p, q) = 1 - BC where BC = ∫√(p q) dx

    For Gaussians: BC = |Σ₁|^{1/4} |Σ₂|^{1/4} |Σ_avg|^{-1/2} exp(-⅛ Δμᵀ Σ_avg⁻¹ Δμ)

    Args:
        mu1, cov1: First Gaussian parameters
        mu2, cov2: Second Gaussian parameters

    Returns:
        H²: Squared Hellinger distance in [0, 1]
    """
    cov_avg = 0.5 * (cov1 + cov2)

    # Regularize for numerical stability
    n = cov1.shape[0]
    I = jnp.eye(n, dtype=cov1.dtype)
    cov_avg_reg = cov_avg + I * COV_REGULARIZATION
    cov1_reg = cov1 + I * COV_REGULARIZATION
    cov2_reg = cov2 + I * COV_REGULARIZATION

    # Cholesky factorizations
    L_avg = cholesky(cov_avg_reg, lower=True)
    L1 = cholesky(cov1_reg, lower=True)
    L2 = cholesky(cov2_reg, lower=True)

    # Log determinants
    logdet_avg = 2.0 * jnp.sum(jnp.log(jnp.diag(L_avg)))
    logdet1 = 2.0 * jnp.sum(jnp.log(jnp.diag(L1)))
    logdet2 = 2.0 * jnp.sum(jnp.log(jnp.diag(L2)))

    # Mahalanobis distance
    diff = mu1 - mu2
    y = solve_triangular(L_avg, diff, lower=True)
    mahal_sq = jnp.dot(y, y)

    # Bhattacharyya coefficient
    log_bc = 0.25 * logdet1 + 0.25 * logdet2 - 0.5 * logdet_avg - 0.125 * mahal_sq
    bc = jnp.exp(log_bc)
    return jnp.maximum(0.0, 1.0 - bc)


# =============================================================================
# Main IMU Projection Kernel
# =============================================================================


@jax.jit
def imu_batched_projection_kernel(
    anchor_mus: jnp.ndarray,
    anchor_covs: jnp.ndarray,
    current_mu: jnp.ndarray,
    current_cov: jnp.ndarray,
    routing_weights: jnp.ndarray,
    imu_stamps: jnp.ndarray,
    imu_accel: jnp.ndarray,
    imu_gyro: jnp.ndarray,
    imu_valid: jnp.ndarray,
    R_imu: jnp.ndarray,
    R_nom: jnp.ndarray,
    gravity: jnp.ndarray,
    dt_total: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Batched IMU projection kernel.

    Args:
        anchor_mus: (M, 15) anchor states (15D)
        anchor_covs: (M, 15, 15) anchor covariances
        current_mu: (15,) current state
        current_cov: (15, 15) current covariance
        routing_weights: (M,) responsibility weights (Dirichlet)
        imu_stamps, imu_accel, imu_gyro, imu_valid: IMU segment data
        R_imu: (9, 9) preintegration covariance
        R_nom: (9, 9) nominal covariance (unused placeholder)
        gravity: (3,) gravity vector
        dt_total: segment duration

    Returns:
        joint_mean: (30,) joint state [anchor_0, current] for Schur marginalization
        joint_cov: (30, 30) joint covariance
        diagnostics: dict with hellinger_mean, degenerate_weights, etc.
    """
    M = anchor_mus.shape[0]

    # Normalize routing weights and enforce minimum
    routing_weights = routing_weights / (jnp.sum(routing_weights) + MIN_MIXTURE_WEIGHT)
    routing_weights = jnp.maximum(routing_weights, MIN_MIXTURE_WEIGHT)
    routing_weights = routing_weights / jnp.sum(routing_weights)

    # Compute per-anchor residuals and Hellinger weights
    def residual_for_anchor(anchor_mu, anchor_cov, weight):
        r = imu_residual_from_raw(
            anchor_mu,
            current_mu,
            imu_stamps,
            imu_accel,
            imu_gyro,
            imu_valid,
            gravity,
            dt_total,
        )

        # IMU residual covariance (9x9) from linearized model
        # Use anchor covariance position/velocity/bias blocks
        cov_r = anchor_cov[:9, :9] + current_cov[:9, :9]

        # Hellinger distance between residual and zero-mean
        h2 = hellinger_squared_gaussian(r, cov_r, jnp.zeros_like(r), cov_r)
        w = jnp.exp(-HELLINGER_TILT_WEIGHT * h2)
        return r, w

    residuals, h_weights = jax.vmap(residual_for_anchor)(anchor_mus, anchor_covs, routing_weights)

    # Combine Hellinger and routing weights
    weights = routing_weights * h_weights
    weights = weights / (jnp.sum(weights) + MIN_MIXTURE_WEIGHT)

    # Moment matching (weighted mean + covariance) in the 9D tangent subspace:
    # [p(3), rotvec(3), v(3)]
    mean_delta9 = jnp.sum(weights[:, None] * residuals, axis=0)
    mean_delta15 = jnp.concatenate([mean_delta9, jnp.zeros((6,), dtype=mean_delta9.dtype)], axis=0)
    delta_mu = _apply_delta_state(current_mu, mean_delta15)

    centered = residuals - mean_delta9
    cov_delta = jnp.einsum("i,ij,ik->jk", weights, centered, centered)

    # Apply delta covariance to current covariance
    delta_cov = jnp.zeros_like(current_cov)
    delta_cov = delta_cov.at[:9, :9].set(cov_delta)
    new_cov = current_cov + delta_cov

    # Build joint state for Schur marginalization
    # Joint = [anchor_0 (15D), current (15D)] = 30D
    # For single anchor case, use first anchor
    joint_mean = jnp.concatenate([anchor_mus[0], delta_mu], axis=0)
    
    # =========================================================================
    # INFORMATION-FORM FACTOR CONSTRUCTION (Principled)
    # =========================================================================
    # The IMU preintegration creates a binary factor connecting anchor and current.
    # The measurement model (linearized) is:
    #   r = xi_current[:9] - (xi_anchor[:9] + delta_preint)
    # 
    # Jacobians (allowed in sensor→evidence extraction per AGENTS.md):
    #   J_anchor = -I_9  (residual decreases when anchor increases)
    #   J_current = +I_9 (residual increases when current increases)
    #
    # Information matrix for joint state from this factor:
    #   L_factor = [[J_a.T @ R_inv @ J_a, J_a.T @ R_inv @ J_c],
    #               [J_c.T @ R_inv @ J_a, J_c.T @ R_inv @ J_c]]
    #            = [[R_inv, -R_inv],
    #               [-R_inv, R_inv]]
    #
    # This is added to the prior joint information matrix.
    # The cross-information is -R_inv, NOT an ad-hoc 0.5*cov_delta.
    # =========================================================================
    
    # Compute IMU factor precision (R_imu_inv) from moment-matched covariance
    # cov_delta is the 9x9 covariance of the moment-matched residual
    # Regularize and invert to get precision
    n9 = 9
    I9 = jnp.eye(n9, dtype=cov_delta.dtype)
    cov_delta_reg = cov_delta + I9 * COV_REGULARIZATION
    
    # Cholesky-based inversion for precision
    L_cov = cholesky(cov_delta_reg, lower=True)
    R_inv_9 = solve_triangular(L_cov, I9, lower=True)
    R_inv_9 = R_inv_9.T @ R_inv_9  # R_inv = (L^-1)^T @ L^-1
    
    # Embed 9x9 precision into 15x15 (only affects p, rotvec, v, not biases)
    R_inv = jnp.zeros((15, 15), dtype=current_cov.dtype)
    R_inv = R_inv.at[:9, :9].set(R_inv_9)
    
    # Convert prior covariances to information form
    # Lambda = Sigma^{-1}
    I15 = jnp.eye(15, dtype=current_cov.dtype)
    
    anchor_cov_reg = anchor_covs[0] + I15 * COV_REGULARIZATION
    L_anc = cholesky(anchor_cov_reg, lower=True)
    Lambda_anchor = solve_triangular(L_anc, I15, lower=True)
    Lambda_anchor = Lambda_anchor.T @ Lambda_anchor
    
    current_cov_reg = new_cov + I15 * COV_REGULARIZATION
    L_cur = cholesky(current_cov_reg, lower=True)
    Lambda_current = solve_triangular(L_cur, I15, lower=True)
    Lambda_current = Lambda_current.T @ Lambda_current
    
    # Joint information matrix: prior + IMU factor
    # Prior (assuming independent): [[Lambda_anchor, 0], [0, Lambda_current]]
    # IMU factor: [[R_inv, -R_inv], [-R_inv, R_inv]]
    # Joint = Prior + Factor
    joint_info = jnp.zeros((30, 30), dtype=current_cov.dtype)
    
    # Prior blocks
    joint_info = joint_info.at[:15, :15].set(Lambda_anchor)
    joint_info = joint_info.at[15:, 15:].set(Lambda_current)
    
    # Add IMU factor contribution (principled cross-information structure)
    joint_info = joint_info.at[:15, :15].set(joint_info[:15, :15] + R_inv)
    joint_info = joint_info.at[15:, 15:].set(joint_info[15:, 15:] + R_inv)
    joint_info = joint_info.at[:15, 15:].set(joint_info[:15, 15:] - R_inv)  # Cross-info = -R_inv
    joint_info = joint_info.at[15:, :15].set(joint_info[15:, :15] - R_inv)  # Symmetric
    
    # Convert back to covariance form for downstream Schur marginalization
    # joint_cov = joint_info^{-1}
    I30 = jnp.eye(30, dtype=current_cov.dtype)
    joint_info_reg = joint_info + I30 * COV_REGULARIZATION
    L_joint = cholesky(joint_info_reg, lower=True)
    joint_cov = solve_triangular(L_joint, I30, lower=True)
    joint_cov = joint_cov.T @ joint_cov

    # Return JAX arrays for diagnostics (convert to Python types outside JIT)
    hellinger_mean = jnp.mean(h_weights)
    degenerate = jnp.any(weights < MIN_MIXTURE_WEIGHT * 0.1)
    # Count anchors with significant weight (above 1% threshold)
    valid_anchors = jnp.sum(weights > 0.01)
    # Effective sample size for weight distribution
    ess = 1.0 / (jnp.sum(weights ** 2) + 1e-10)
    # Weight entropy
    weight_entropy = -jnp.sum(weights * jnp.log(weights + 1e-10))
    
    # Pack diagnostics as JAX arrays (dict is returned outside JIT-traced code)
    # The caller will convert these to Python types
    diagnostics = {
        "hellinger_mean": hellinger_mean,
        "hellinger_weights": h_weights,
        "routing_weights": routing_weights,
        "degenerate_weights": degenerate,
        "valid_anchors": valid_anchors,
        "ess": ess,
        "weight_entropy": weight_entropy,
    }

    return joint_mean, joint_cov, diagnostics
