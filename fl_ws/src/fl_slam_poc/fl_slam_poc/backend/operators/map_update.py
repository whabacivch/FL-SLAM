"""
PoseCovInflationPushforward operator for Golden Child SLAM v2.

Push scan statistics into map frame with pose covariance inflation.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.13
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import BeliefGaussianInfo, SLICE_POSE
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_inverse_lifted,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class MapUpdateResult:
    """Result of PoseCovInflationPushforward operator."""
    # Increments to add to map sufficient statistics
    delta_S_dir: jnp.ndarray  # (B_BINS, 3)
    delta_N_dir: jnp.ndarray  # (B_BINS,)
    delta_N_pos: jnp.ndarray  # (B_BINS,)
    delta_sum_p: jnp.ndarray  # (B_BINS, 3)
    delta_sum_ppT: jnp.ndarray  # (B_BINS, 3, 3)
    inflation_magnitude: float  # Total covariance inflation


# =============================================================================
# Main Operator
# =============================================================================


def pos_cov_inflation_pushforward(
    belief_post: BeliefGaussianInfo,
    scan_N: jnp.ndarray,
    scan_s_dir: jnp.ndarray,
    scan_p_bar: jnp.ndarray,
    scan_Sigma_p: jnp.ndarray,
    R_hat: jnp.ndarray,
    t_hat: jnp.ndarray,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> Tuple[MapUpdateResult, CertBundle, ExpectedEffect]:
    """
    Push scan statistics into map frame with pose covariance inflation.
    
    The pose uncertainty from the belief is propagated into the
    position covariance of each bin.
    
    Args:
        belief_post: Posterior belief
        scan_N: Scan bin masses (B,)
        scan_s_dir: Scan direction resultants (B, 3)
        scan_p_bar: Scan centroids (B, 3)
        scan_Sigma_p: Scan centroid covariances (B, 3, 3)
        R_hat: Estimated rotation (3, 3)
        t_hat: Estimated translation (3,)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        
    Returns:
        Tuple of (MapUpdateResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.13
    """
    scan_N = jnp.asarray(scan_N, dtype=jnp.float64)
    scan_s_dir = jnp.asarray(scan_s_dir, dtype=jnp.float64)
    scan_p_bar = jnp.asarray(scan_p_bar, dtype=jnp.float64)
    scan_Sigma_p = jnp.asarray(scan_Sigma_p, dtype=jnp.float64)
    R_hat = jnp.asarray(R_hat, dtype=jnp.float64)
    t_hat = jnp.asarray(t_hat, dtype=jnp.float64)
    
    n_bins = scan_N.shape[0]
    
    # Extract pose covariance from belief
    _, cov_full, _ = belief_post.to_moments(eps_lift)
    
    # Pose covariance is the 6x6 block
    # [rotation cov (3x3), rot-trans cross (3x3)]
    # [trans-rot cross (3x3), translation cov (3x3)]
    Sigma_pose = cov_full[SLICE_POSE, SLICE_POSE]  # (6, 6)
    
    # For position inflation, we need the translation uncertainty
    # and how rotation uncertainty affects position
    Sigma_trans = Sigma_pose[3:6, 3:6]  # Translation covariance
    Sigma_rot = Sigma_pose[0:3, 0:3]  # Rotation covariance
    
    # Initialize output arrays
    delta_S_dir = jnp.zeros((n_bins, 3), dtype=jnp.float64)
    delta_N_dir = jnp.zeros(n_bins, dtype=jnp.float64)
    delta_N_pos = jnp.zeros(n_bins, dtype=jnp.float64)
    delta_sum_p = jnp.zeros((n_bins, 3), dtype=jnp.float64)
    delta_sum_ppT = jnp.zeros((n_bins, 3, 3), dtype=jnp.float64)
    
    total_inflation = 0.0
    total_psd_projection_delta = 0.0
    
    for b in range(n_bins):
        N_b = scan_N[b]
        s_dir_b = scan_s_dir[b]
        p_bar_b = scan_p_bar[b]
        Sigma_p_b = scan_Sigma_p[b]
        
        # Transform centroid to map frame
        p_map_b = R_hat @ p_bar_b + t_hat
        
        # Transform direction to map frame
        s_dir_map_b = R_hat @ s_dir_b
        
        # Inflate position covariance with pose uncertainty
        # For rotation: J_rot @ Sigma_rot @ J_rot^T where J_rot = -[R @ p]_x
        p_rotated = R_hat @ p_bar_b
        
        # Skew symmetric for cross product
        px, py, pz = p_rotated[0], p_rotated[1], p_rotated[2]
        p_skew = jnp.array([
            [0, -pz, py],
            [pz, 0, -px],
            [-py, px, 0]
        ], dtype=jnp.float64)
        
        # Rotation contribution to position uncertainty
        Sigma_rot_contribution = p_skew @ Sigma_rot @ p_skew.T
        
        # Total position covariance in map frame
        Sigma_map_raw = (
            R_hat @ Sigma_p_b @ R_hat.T  # Transformed measurement cov
            + Sigma_trans  # Translation uncertainty
            + Sigma_rot_contribution  # Rotation uncertainty
        )
        
        # Project to PSD (always)
        Sigma_map_result = domain_projection_psd(Sigma_map_raw, eps_psd)
        Sigma_map_b = Sigma_map_result.M_psd
        total_psd_projection_delta += Sigma_map_result.projection_delta
        
        inflation_b = float(jnp.trace(Sigma_map_b) - jnp.trace(R_hat @ Sigma_p_b @ R_hat.T))
        total_inflation += inflation_b * float(N_b)
        
        # Build increments to sufficient statistics
        delta_N_dir = delta_N_dir.at[b].set(N_b)
        delta_N_pos = delta_N_pos.at[b].set(N_b)
        delta_S_dir = delta_S_dir.at[b].set(s_dir_map_b)
        delta_sum_p = delta_sum_p.at[b].set(N_b * p_map_b)
        
        # For scatter: sum_ppT increment = N * (Sigma + p @ p^T)
        # = N * Sigma + N * p @ p^T
        delta_sum_ppT = delta_sum_ppT.at[b].set(
            N_b * Sigma_map_b + N_b * jnp.outer(p_map_b, p_map_b)
        )
    
    # Build result
    result = MapUpdateResult(
        delta_S_dir=delta_S_dir,
        delta_N_dir=delta_N_dir,
        delta_N_pos=delta_N_pos,
        delta_sum_p=delta_sum_p,
        delta_sum_ppT=delta_sum_ppT,
        inflation_magnitude=total_inflation,
    )
    
    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=belief_post.chart_id,
        anchor_id=belief_post.anchor_id,
        triggers=["PoseCovInflationPushforward"],
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=total_psd_projection_delta,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_inflation_magnitude",
        predicted=total_inflation,
        realized=None,
    )
    
    return result, cert, expected_effect
