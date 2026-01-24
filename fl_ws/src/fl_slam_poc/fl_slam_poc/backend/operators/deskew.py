"""
DeskewUTMomentMatch operator for Golden Child SLAM v2.

Unscented transform deskewing with moment matching.
Produces ut_cache for reuse by LidarQuadraticEvidence.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

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
class DeskewedPoint:
    """Single deskewed point with uncertainty."""
    p_mean: jnp.ndarray  # (3,) mean position
    p_cov: jnp.ndarray  # (3, 3) position covariance
    time_sec: float
    weight: float


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
    deskewed_points: List[DeskewedPoint]
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
    
    # Generate 2n+1 sigma points
    sigma_points = jnp.zeros((2 * n + 1, n))
    
    # Central point
    sigma_points = sigma_points.at[0].set(mean)
    
    # Points along positive/negative sqrt directions
    for i in range(n):
        sigma_points = sigma_points.at[1 + i].set(mean + L[:, i])
        sigma_points = sigma_points.at[1 + n + i].set(mean - L[:, i])
    
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
    
    n_points = points.shape[0]
    
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
    pose_deltas = jnp.zeros((T_SLICES, n_sigma, 6))
    
    # Extract pose slice from each sigma point
    for s in range(n_sigma):
        pose_s_z = sigma_points[s, SLICE_POSE]  # (6,) GC ordering: [rot, trans]
        pose_s = pose_z_to_se3_delta(pose_s_z)  # (6,) se3_jax ordering: [trans, rot]
        for t in range(T_SLICES):
            # Interpolation factor for this time slice
            alpha = (slice_times[t] - scan_start_time) / (scan_end_time - scan_start_time + 1e-12)
            # Pose delta at this time (linear interpolation in tangent space)
            pose_deltas = pose_deltas.at[t, s].set(alpha * pose_s)
    
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
    
    # Deskew each point using moment matching
    deskewed_points = []
    total_cov_trace = 0.0
    
    for i in range(n_points):
        point = points[i]
        time_i = float(timestamps[i])
        weight_i = float(weights[i])
        
        # Find the closest time slice
        alpha = (time_i - scan_start_time) / (scan_end_time - scan_start_time + 1e-12)
        alpha = jnp.clip(alpha, 0.0, 1.0)
        
        # Interpolate pose for each sigma point at this time
        transformed_points = jnp.zeros((n_sigma, 3))
        for s in range(n_sigma):
            # Interpolate pose delta for this sigma point
            pose_s_z = sigma_points[s, SLICE_POSE]
            pose_s = pose_z_to_se3_delta(pose_s_z)
            pose_at_t = alpha * pose_s
            
            # Transform point
            transformed_points = transformed_points.at[s].set(
                _transform_point_by_pose(point, pose_at_t)
            )
        
        # Moment matching: compute mean and covariance of transformed points
        p_mean = jnp.sum(weights_mean[:, None] * transformed_points, axis=0)
        
        # Covariance
        p_cov = jnp.zeros((3, 3))
        for s in range(n_sigma):
            delta = transformed_points[s] - p_mean
            p_cov = p_cov + weights_cov[s] * jnp.outer(delta, delta)
        
        # Project covariance to PSD
        p_cov_psd = domain_projection_psd(p_cov, eps_psd).M_psd
        
        deskewed_points.append(DeskewedPoint(
            p_mean=p_mean,
            p_cov=p_cov_psd,
            time_sec=time_i,
            weight=weight_i,
        ))
        
        total_cov_trace += float(jnp.trace(p_cov_psd)) * weight_i
    
    # Normalize by total weight
    total_weight = float(jnp.sum(weights))
    predicted_cov_trace = total_cov_trace / (total_weight + constants.GC_EPS_MASS)
    
    # Build result
    result = DeskewResult(
        deskewed_points=deskewed_points,
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
