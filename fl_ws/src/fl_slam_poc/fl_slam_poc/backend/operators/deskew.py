"""
DeskewUTMomentMatch operator for Golden Child SLAM v2.

Unscented transform deskewing with moment matching.
Produces ut_cache for reuse by LidarQuadraticEvidence.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import (
    BeliefGaussianInfo,
    D_Z,
    SLICE_POSE,
    pose_z_to_se3_delta,
)
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ExcitationCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_inverse_lifted,
)
from fl_slam_poc.common.geometry import se3_jax


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class UTCache:
    """
    Cache of UT evaluations for reuse by LidarQuadraticEvidence.
    
    Contains all sigma point evaluations so they don't need to be
    recomputed downstream.
    """
    sigma_points: jnp.ndarray  # (SIGMA_POINTS, D_Z) sigma points in tangent space
    weights_mean: jnp.ndarray  # (SIGMA_POINTS,) weights for mean
    weights_cov: jnp.ndarray  # (SIGMA_POINTS,) weights for covariance
    pose_deltas: jnp.ndarray  # (T_SLICES, SIGMA_POINTS, 6) pose deltas per slice
    dt_contributions: jnp.ndarray  # (SIGMA_POINTS,) time offset contributions
    extrinsic_contributions: jnp.ndarray  # (SIGMA_POINTS, 6) extrinsic contributions


@dataclass
class DeskewResult:
    """Result of DeskewUTMomentMatch operator."""
    p_mean: jnp.ndarray  # (N, 3) deskewed point means
    p_cov: jnp.ndarray  # (N, 3, 3) deskewed point covariances
    timestamps: jnp.ndarray  # (N,) point timestamps (sec)
    weights: jnp.ndarray  # (N,) point weights
    ut_cache: UTCache
    dt_effect: float  # Continuous excitation from time offset
    extrinsic_effect: float  # Continuous excitation from extrinsic


# =============================================================================
# Sigma Point Generation (JAX)
# =============================================================================


@jax.jit
def _generate_sigma_points(
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    kappa: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate sigma points using unscented transform.
    
    Args:
        mean: Mean vector (n,)
        cov: Covariance matrix (n, n)
        kappa: Scaling parameter (typically 0 or 3-n)
        
    Returns:
        Tuple of (sigma_points, weights_mean, weights_cov)
    """
    n = mean.shape[0]
    lambda_ = kappa  # Simplified: alpha=1, beta=0
    
    # Compute square root of scaled covariance
    # Using Cholesky: L @ L.T = (n + lambda) * cov
    scale = n + lambda_
    scaled_cov = scale * cov
    
    # Add small regularization for numerical stability
    scaled_cov = scaled_cov + 1e-12 * jnp.eye(n)
    L = jnp.linalg.cholesky(scaled_cov)
    
    # Generate 2n+1 sigma points (vectorized, no Python loop)
    # Central point at index 0, positive directions at 1:n+1, negative at n+1:2n+1
    L_T = L.T  # (n, n) - columns become rows for easy broadcasting
    
    sigma_points = jnp.concatenate([
        mean[None, :],           # (1, n) - central point
        mean[None, :] + L_T,     # (n, n) - positive directions
        mean[None, :] - L_T,     # (n, n) - negative directions
    ], axis=0)  # (2n+1, n)
    
    # Weights
    w0 = lambda_ / (n + lambda_)
    wi = 1.0 / (2.0 * (n + lambda_))
    
    weights_mean = jnp.full(2 * n + 1, wi)
    weights_mean = weights_mean.at[0].set(w0)
    
    weights_cov = jnp.full(2 * n + 1, wi)
    weights_cov = weights_cov.at[0].set(w0)
    
    return sigma_points, weights_mean, weights_cov


@jax.jit
def _transform_point_by_pose(
    point: jnp.ndarray,
    pose_delta: jnp.ndarray,
) -> jnp.ndarray:
    """
    Transform a 3D point by a 6D pose delta.
    
    Args:
        point: 3D point (3,)
        pose_delta: 6D pose delta [trans, rotvec]
        
    Returns:
        Transformed point (3,)
    """
    trans = pose_delta[:3]
    rotvec = pose_delta[3:6]
    
    # Rotation matrix from rotation vector
    R = se3_jax.so3_exp(rotvec)
    
    # Apply: R @ point + trans
    return R @ point + trans


@jax.jit
def _pose_z_to_se3_delta_batch(delta_pose_z: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized pose convention conversion (GC -> se3_jax ordering).

    Input:  (K, 6) [rot(3), trans(3)]
    Output: (K, 6) [trans(3), rot(3)]
    """
    delta_pose_z = jnp.asarray(delta_pose_z, dtype=jnp.float64)
    rot = delta_pose_z[:, 0:3]
    trans = delta_pose_z[:, 3:6]
    return jnp.concatenate([trans, rot], axis=1)


@jax.jit
def _project_psd_batch_3x3(M: jnp.ndarray, eps_psd: float) -> jnp.ndarray:
    """
    Batch PSD projection for stacks of 3x3 matrices.

    Always symmetrizes and eigen-clamps to >= eps_psd.
    """
    M = jnp.asarray(M, dtype=jnp.float64)
    M_sym = 0.5 * (M + jnp.swapaxes(M, -1, -2))
    eigvals, eigvecs = jax.vmap(jnp.linalg.eigh)(M_sym)
    vals_clamped = jnp.maximum(eigvals, eps_psd)
    # Reconstruct: V diag(vals) V^T
    return jnp.einsum("nij,nj,nkj->nik", eigvecs, vals_clamped, eigvecs)


@jax.jit
def _deskew_points_ut_batch(
    points: jnp.ndarray,
    alphas: jnp.ndarray,
    pose_sigma_se3: jnp.ndarray,
    weights_mean: jnp.ndarray,
    weights_cov: jnp.ndarray,
    eps_psd: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Deskew all points via UT moment matching in a single batched computation.

    Args:
        points: (N, 3)
        alphas: (N,) in [0, 1]
        pose_sigma_se3: (S, 6) sigma-point pose increments in se3_jax ordering [trans, rot]
        weights_mean: (S,)
        weights_cov: (S,)
        eps_psd: PSD clamp epsilon

    Returns:
        Tuple of (p_mean (N,3), p_cov_psd (N,3,3))
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    alphas = jnp.asarray(alphas, dtype=jnp.float64).reshape(-1)
    pose_sigma_se3 = jnp.asarray(pose_sigma_se3, dtype=jnp.float64)
    weights_mean = jnp.asarray(weights_mean, dtype=jnp.float64).reshape(-1)
    weights_cov = jnp.asarray(weights_cov, dtype=jnp.float64).reshape(-1)

    trans_sigma = pose_sigma_se3[:, 0:3]  # (S,3)
    rot_sigma = pose_sigma_se3[:, 3:6]  # (S,3)

    def _one_point(point: jnp.ndarray, alpha: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        rotvecs = alpha * rot_sigma  # (S,3)
        trans = alpha * trans_sigma  # (S,3)
        R = jax.vmap(se3_jax.so3_exp)(rotvecs)  # (S,3,3)
        transformed = jnp.einsum("sij,j->si", R, point) + trans  # (S,3)
        p_mean = jnp.sum(weights_mean[:, None] * transformed, axis=0)  # (3,)
        delta = transformed - p_mean[None, :]  # (S,3)
        p_cov = jnp.einsum("s,si,sj->ij", weights_cov, delta, delta)  # (3,3)
        return p_mean, p_cov

    p_mean, p_cov = jax.vmap(_one_point)(points, alphas)
    p_cov_psd = _project_psd_batch_3x3(p_cov, eps_psd)
    return p_mean, p_cov_psd


# =============================================================================
# Main Operator
# =============================================================================


def deskew_ut_moment_match(
    belief_pred: BeliefGaussianInfo,
    points: jnp.ndarray,
    timestamps: jnp.ndarray,
    weights: jnp.ndarray,
    scan_start_time: float,
    scan_end_time: float,
    T_SLICES: int = constants.GC_T_SLICES,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> Tuple[DeskewResult, CertBundle, ExpectedEffect]:
    """
    Deskew points using unscented transform with moment matching.
    
    Uses exactly T_SLICES * SIGMA_POINTS pose evaluations.
    Produces ut_cache for reuse by LidarQuadraticEvidence.
    
    Args:
        belief_pred: Predicted belief
        points: Raw point positions (N, 3)
        timestamps: Per-point timestamps (N,)
        weights: Per-point weights (N,)
        scan_start_time: Scan start timestamp
        scan_end_time: Scan end timestamp
        T_SLICES: Number of time slices (default from constants)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        
    Returns:
        Tuple of (DeskewResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.3
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    timestamps = jnp.asarray(timestamps, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    
    # Get mean and covariance from belief
    mean, cov, _ = belief_pred.to_moments(eps_lift)
    
    # Project covariance to PSD
    cov_psd_result = domain_projection_psd(cov, eps_psd)
    cov_psd = cov_psd_result.M_psd
    
    # Generate sigma points (D_Z dimensional)
    sigma_points, weights_mean, weights_cov = _generate_sigma_points(mean, cov_psd)
    n_sigma = sigma_points.shape[0]  # 2 * D_Z + 1 = 45
    
    # Time slices for deskewing
    slice_times = jnp.linspace(scan_start_time, scan_end_time, T_SLICES)
    
    # Compute pose deltas for each sigma point at each time slice
    # Shape: (T_SLICES, SIGMA_POINTS, 6)
    pose_sigma_z = sigma_points[:, SLICE_POSE]  # (S,6) [rot, trans]
    pose_sigma_se3 = _pose_z_to_se3_delta_batch(pose_sigma_z)  # (S,6) [trans, rot]
    slice_alphas = (slice_times - scan_start_time) / (scan_end_time - scan_start_time + 1e-12)
    pose_deltas = slice_alphas[:, None, None] * pose_sigma_se3[None, :, :]  # (T,S,6)
    
    # Compute excitation metrics from time offset and extrinsic contributions
    # Time offset is at index 15
    dt_contributions = sigma_points[:, 15]
    dt_effect = float(jnp.std(dt_contributions))
    
    # Extrinsic is at indices 16:22
    extrinsic_contributions = sigma_points[:, 16:22]
    extrinsic_effect = float(jnp.std(jnp.linalg.norm(extrinsic_contributions, axis=1)))
    
    # Build UT cache
    ut_cache = UTCache(
        sigma_points=sigma_points,
        weights_mean=weights_mean,
        weights_cov=weights_cov,
        pose_deltas=pose_deltas,
        dt_contributions=dt_contributions,
        extrinsic_contributions=extrinsic_contributions,
    )
    
    # Deskew all points using moment matching (batched, no per-point host sync)
    alphas_pts = (timestamps - scan_start_time) / (scan_end_time - scan_start_time + 1e-12)
    alphas_pts = jnp.clip(alphas_pts, 0.0, 1.0)
    p_mean, p_cov_psd = _deskew_points_ut_batch(
        points=points,
        alphas=alphas_pts,
        pose_sigma_se3=pose_sigma_se3,
        weights_mean=weights_mean,
        weights_cov=weights_cov,
        eps_psd=eps_psd,
    )

    # Predicted covariance trace metric (weighted)
    total_cov_trace = jnp.sum(jnp.trace(p_cov_psd, axis1=1, axis2=2) * weights)
    total_weight = jnp.sum(weights)
    predicted_cov_trace = float(total_cov_trace / (total_weight + constants.GC_EPS_MASS))
    
    # Build result
    result = DeskewResult(
        p_mean=p_mean,
        p_cov=p_cov_psd,
        timestamps=timestamps,
        weights=weights,
        ut_cache=ut_cache,
        dt_effect=dt_effect,
        extrinsic_effect=extrinsic_effect,
    )
    
    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
        triggers=["DeskewUTMomentMatch"],
        excitation=ExcitationCert(
            dt_effect=dt_effect,
            extrinsic_effect=extrinsic_effect,
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=cov_psd_result.projection_delta,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_deskew_cov_trace",
        predicted=predicted_cov_trace,
        realized=None,
    )
    
    return result, cert, expected_effect
