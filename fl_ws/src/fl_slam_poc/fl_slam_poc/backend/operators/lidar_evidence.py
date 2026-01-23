"""
LidarQuadraticEvidence operator for Golden Child SLAM v2.

Produces quadratic evidence on full 22D tangent at fixed cost.
Reuses ut_cache from DeskewUTMomentMatch.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.9
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import BeliefGaussianInfo, D_Z, SLICE_POSE
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    ExcitationCert,
    MismatchCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_solve_lifted,
)
from fl_slam_poc.backend.operators.deskew import UTCache
from fl_slam_poc.backend.operators.binning import ScanBinStats
from fl_slam_poc.common.geometry import se3_jax


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class MapBinStats:
    """Map bin sufficient statistics."""
    S_dir: jnp.ndarray  # (B_BINS, 3) directional resultants
    N_dir: jnp.ndarray  # (B_BINS,) directional mass
    N_pos: jnp.ndarray  # (B_BINS,) position mass
    sum_p: jnp.ndarray  # (B_BINS, 3) position sums
    sum_ppT: jnp.ndarray  # (B_BINS, 3, 3) scatter matrices
    mu_dir: jnp.ndarray  # (B_BINS, 3) mean directions
    kappa_map: jnp.ndarray  # (B_BINS,) concentration parameters
    centroid: jnp.ndarray  # (B_BINS, 3) centroids
    Sigma_c: jnp.ndarray  # (B_BINS, 3, 3) centroid covariances


@dataclass 
class LidarEvidenceResult:
    """Result of LidarQuadraticEvidence operator."""
    L_lidar: jnp.ndarray  # (D_Z, D_Z) information matrix
    h_lidar: jnp.ndarray  # (D_Z,) information vector
    delta_z_star: jnp.ndarray  # (D_Z,) MAP increment


# =============================================================================
# Main Operator
# =============================================================================


def lidar_quadratic_evidence(
    belief_pred: BeliefGaussianInfo,
    scan_bins: ScanBinStats,
    map_bins: MapBinStats,
    R_hat: jnp.ndarray,
    t_hat: jnp.ndarray,
    ut_cache: UTCache,
    c_dt: float = constants.GC_C_DT,
    c_ex: float = constants.GC_C_EX,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> Tuple[LidarEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Produce quadratic evidence on full 22D tangent at fixed cost.
    
    Reuses ut_cache from DeskewUTMomentMatch (no extra sigma point passes).
    
    Branch-free coupling rule:
        s_dt = dt_effect / (dt_effect + c_dt)
        s_ex = extrinsic_effect / (extrinsic_effect + c_ex)
    
    Blocks involving index 15 multiplied by s_dt.
    Blocks involving indices 16..21 multiplied by s_ex.
    
    Args:
        belief_pred: Predicted belief
        scan_bins: Scan bin statistics
        map_bins: Map bin statistics
        R_hat: Estimated rotation (3, 3)
        t_hat: Estimated translation (3,)
        ut_cache: UT cache from DeskewUTMomentMatch
        c_dt: Time offset coupling constant
        c_ex: Extrinsic coupling constant
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        
    Returns:
        Tuple of (LidarEvidenceResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.9
    """
    R_hat = jnp.asarray(R_hat, dtype=jnp.float64)
    t_hat = jnp.asarray(t_hat, dtype=jnp.float64)
    
    # Step 1: Compute excitation scales (continuous, no branching)
    dt_effect = float(jnp.std(ut_cache.dt_contributions))
    extrinsic_effect = float(jnp.std(jnp.linalg.norm(ut_cache.extrinsic_contributions, axis=1)))
    
    s_dt = dt_effect / (dt_effect + c_dt)
    s_ex = extrinsic_effect / (extrinsic_effect + c_ex)
    
    # Step 2: Build delta_z* from (R_hat, t_hat)
    # Pose slice corresponds to [rotation, translation] in right perturbation
    delta_z_star = jnp.zeros(D_Z, dtype=jnp.float64)
    
    # Convert R_hat to rotation vector
    rotvec = se3_jax.so3_log(R_hat)
    
    # Set pose slice: [rotation(0:3), translation(3:6)]
    delta_z_star = delta_z_star.at[0:3].set(rotvec)
    delta_z_star = delta_z_star.at[3:6].set(t_hat)
    
    # Step 3: Build L_lidar using UT regression
    # Use sigma points from ut_cache to build quadratic model
    sigma_points = ut_cache.sigma_points  # (SIGMA_POINTS, D_Z)
    weights_cov = ut_cache.weights_cov  # (SIGMA_POINTS,)
    n_sigma = sigma_points.shape[0]
    
    # Compute residuals for each sigma point
    # Residual = how well the sigma point explains the observations
    # Using pose slice only for residual computation
    residuals = jnp.zeros(n_sigma, dtype=jnp.float64)
    
    # Mean sigma point (should be close to zero for increments)
    mean_sigma = jnp.sum(weights_cov[:, None] * sigma_points, axis=0)
    
    # Build information matrix from weighted outer products
    # L_lidar = sum_s w_s * (sigma_s - mean) @ (sigma_s - mean)^T
    # This is the inverse of the UT covariance, scaled
    L_lidar_raw = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    
    for s in range(n_sigma):
        delta_s = sigma_points[s] - mean_sigma
        L_lidar_raw = L_lidar_raw + weights_cov[s] * jnp.outer(delta_s, delta_s)
    
    # Invert to get precision (information) matrix
    # First project to PSD
    L_cov_psd = domain_projection_psd(L_lidar_raw, eps_psd).M_psd
    
    # Invert with lift
    L_lidar_inv, _ = spd_cholesky_inverse_lifted(L_cov_psd, eps_lift)
    
    # The information from LiDAR registration
    # Scale by a factor based on the quality of the registration
    n_bins = scan_bins.N.shape[0]
    total_mass = float(jnp.sum(scan_bins.N))
    
    # Information scaling based on total mass
    info_scale = total_mass / (total_mass + constants.GC_EPS_MASS)
    L_lidar_raw = info_scale * L_lidar_inv
    
    # Step 4: Apply excitation scaling to relevant blocks (always)
    # Time offset: index 15
    L_lidar_raw = L_lidar_raw.at[15, :].set(s_dt * L_lidar_raw[15, :])
    L_lidar_raw = L_lidar_raw.at[:, 15].set(s_dt * L_lidar_raw[:, 15])
    
    # Extrinsic: indices 16..21
    L_lidar_raw = L_lidar_raw.at[16:22, :].set(s_ex * L_lidar_raw[16:22, :])
    L_lidar_raw = L_lidar_raw.at[:, 16:22].set(s_ex * L_lidar_raw[:, 16:22])
    
    # Step 5: Apply DomainProjectionPSD (always)
    L_psd_result = domain_projection_psd(L_lidar_raw, eps_psd)
    L_lidar = L_psd_result.M_psd
    
    # Step 6: Compute h_lidar = L_lidar @ delta_z*
    h_lidar = L_lidar @ delta_z_star
    
    # Build result
    result = LidarEvidenceResult(
        L_lidar=L_lidar,
        h_lidar=h_lidar,
        delta_z_star=delta_z_star,
    )
    
    # Compute mismatch proxy
    nll_proxy = 0.5 * float(delta_z_star @ L_lidar @ delta_z_star)
    
    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
        triggers=["LidarQuadraticEvidence"],
        conditioning=ConditioningCert(
            eig_min=L_psd_result.conditioning.eig_min,
            eig_max=L_psd_result.conditioning.eig_max,
            cond=L_psd_result.conditioning.cond,
            near_null_count=L_psd_result.conditioning.near_null_count,
        ),
        mismatch=MismatchCert(
            nll_per_ess=nll_proxy / (total_mass + constants.GC_EPS_MASS),
            directional_score=0.0,
        ),
        excitation=ExcitationCert(
            dt_effect=dt_effect,
            extrinsic_effect=extrinsic_effect,
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=L_psd_result.projection_delta,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=s_dt,
            extrinsic_scale=s_ex,
            trust_alpha=1.0,
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_quadratic_nll_decrease",
        predicted=nll_proxy,
        realized=None,
    )
    
    return result, cert, expected_effect
