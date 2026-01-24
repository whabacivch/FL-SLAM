"""
TranslationWLS operator for Golden Child SLAM v2.

Weighted least squares translation estimation.
Always uses lifted solve.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.8
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_solve_lifted,
    inv_mass,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TranslationWLSResult:
    """Result of TranslationWLS operator."""
    t_hat: jnp.ndarray  # (3,) estimated translation
    t_cov: jnp.ndarray  # (3, 3) covariance of estimate
    residual_norm: float  # Weighted residual norm


# =============================================================================
# Main Operator
# =============================================================================


def translation_wls(
    c_map: jnp.ndarray,
    Sigma_c_map: jnp.ndarray,
    p_bar_scan: jnp.ndarray,
    Sigma_p_scan: jnp.ndarray,
    R_hat: jnp.ndarray,
    weights: jnp.ndarray,
    Sigma_meas: jnp.ndarray,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[TranslationWLSResult, CertBundle, ExpectedEffect]:
    """
    Weighted least squares translation estimation.
    
    Model: c_map_b = R_hat @ p_bar_scan_b + t + noise
    
    Always uses lifted solve and DomainProjectionPSD.
    
    Args:
        c_map: Map centroids (B, 3)
        Sigma_c_map: Map centroid covariances (B, 3, 3)
        p_bar_scan: Scan centroids (B, 3)
        Sigma_p_scan: Scan centroid covariances (B, 3, 3)
        R_hat: Estimated rotation (3, 3)
        weights: Per-bin weights (B,)
        Sigma_meas: Measurement noise covariance (3, 3)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (TranslationWLSResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.8
    """
    c_map = jnp.asarray(c_map, dtype=jnp.float64)
    Sigma_c_map = jnp.asarray(Sigma_c_map, dtype=jnp.float64)
    p_bar_scan = jnp.asarray(p_bar_scan, dtype=jnp.float64)
    Sigma_p_scan = jnp.asarray(Sigma_p_scan, dtype=jnp.float64)
    R_hat = jnp.asarray(R_hat, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    Sigma_meas = jnp.asarray(Sigma_meas, dtype=jnp.float64)
    
    n_bins = weights.shape[0]
    
    # Build normal equations: A @ t = b
    # where A = Σ w_b * Sigma_b^{-1}, b = Σ w_b * Sigma_b^{-1} @ (c_map_b - R_hat @ p_bar_scan_b)
    
    A = jnp.zeros((3, 3), dtype=jnp.float64)
    b = jnp.zeros(3, dtype=jnp.float64)
    total_lift = 0.0
    total_projection_delta = 0.0
    
    for bin_idx in range(n_bins):
        w_b = weights[bin_idx]
        
        # Skip if weight is essentially zero (continuous by multiplication)
        # No branching - zero weight just contributes zero
        
        # Total covariance for this bin
        # Sigma_b = Sigma_c_map_b + R_hat @ Sigma_p_scan_b @ R_hat^T + Sigma_meas
        Sigma_scan_rotated = R_hat @ Sigma_p_scan[bin_idx] @ R_hat.T
        Sigma_b_raw = Sigma_c_map[bin_idx] + Sigma_scan_rotated + Sigma_meas
        
        # Project to PSD (always)
        Sigma_b_result = domain_projection_psd(Sigma_b_raw, eps_psd)
        Sigma_b = Sigma_b_result.M_psd
        total_projection_delta += Sigma_b_result.projection_delta
        
        # Invert covariance (lifted)
        solve_result = spd_cholesky_solve_lifted(Sigma_b, jnp.eye(3), eps_lift)
        Sigma_b_inv = solve_result.x  # Result is already the inverse
        total_lift += solve_result.lift_strength
        
        # Actually we need to use the inverse properly
        # Let's use the proper lifted solve
        inv_result = domain_projection_psd(Sigma_b, eps_psd)
        L_info = jnp.linalg.inv(inv_result.M_psd + eps_lift * jnp.eye(3))
        
        # Residual for this bin
        p_rotated = R_hat @ p_bar_scan[bin_idx]
        residual_b = c_map[bin_idx] - p_rotated
        
        # Accumulate normal equations
        A = A + w_b * L_info
        b = b + w_b * L_info @ residual_b
    
    # Solve normal equations (always lifted)
    A_psd_result = domain_projection_psd(A, eps_psd)
    A_psd = A_psd_result.M_psd
    total_projection_delta += A_psd_result.projection_delta
    
    solve_result = spd_cholesky_solve_lifted(A_psd, b, eps_lift)
    t_hat = solve_result.x
    total_lift += solve_result.lift_strength
    
    # Covariance of estimate (inverse of A)
    t_cov_raw = jnp.linalg.inv(A_psd + eps_lift * jnp.eye(3))
    t_cov_result = domain_projection_psd(t_cov_raw, eps_psd)
    t_cov = t_cov_result.M_psd
    
    # Compute weighted residual norm
    residual_norm = 0.0
    for bin_idx in range(n_bins):
        w_b = weights[bin_idx]
        p_rotated = R_hat @ p_bar_scan[bin_idx]
        residual_b = c_map[bin_idx] - p_rotated - t_hat
        residual_norm = residual_norm + w_b * float(jnp.dot(residual_b, residual_b))
    residual_norm = float(jnp.sqrt(residual_norm))
    
    # Build result
    result = TranslationWLSResult(
        t_hat=t_hat,
        t_cov=t_cov,
        residual_norm=residual_norm,
    )
    
    # Conditioning info from A
    eig_A = jnp.linalg.eigvalsh(A_psd)
    
    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["TranslationWLS"],
        conditioning=ConditioningCert(
            eig_min=float(jnp.min(eig_A)),
            eig_max=float(jnp.max(eig_A)),
            cond=float(jnp.max(eig_A) / (jnp.min(eig_A) + eps_psd)),
            near_null_count=int(jnp.sum(eig_A < 10 * eps_psd)),
        ),
        influence=InfluenceCert(
            lift_strength=total_lift,
            psd_projection_delta=total_projection_delta,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_translation_residual",
        predicted=residual_norm,
        realized=None,
    )
    
    return result, cert, expected_effect
