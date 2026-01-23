"""
PredictDiffusion operator for Golden Child SLAM v2.

dt-scaled prediction with continuous DomainProjectionPSD.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import BeliefGaussianInfo, D_Z
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_inverse_lifted,
)


# =============================================================================
# Default Process Noise
# =============================================================================


def build_default_process_noise(
    sigma_gyro: float = 0.01,
    sigma_accel: float = 0.1,
    sigma_gyro_bias: float = 1e-4,
    sigma_accel_bias: float = 1e-3,
    sigma_dt: float = 1e-3,
    sigma_extrinsic: float = 1e-4,
) -> jnp.ndarray:
    """
    Build default process noise matrix Q.
    
    Returns:
        Process noise matrix (D_Z, D_Z)
    """
    Q = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    
    # SO(3) rotation (indices 0:3)
    Q = Q.at[0:3, 0:3].set(sigma_gyro**2 * jnp.eye(3))
    
    # Translation (indices 3:6)
    Q = Q.at[3:6, 3:6].set(sigma_accel**2 * jnp.eye(3))
    
    # Velocity (indices 6:9)
    Q = Q.at[6:9, 6:9].set(sigma_accel**2 * jnp.eye(3))
    
    # Gyro bias (indices 9:12)
    Q = Q.at[9:12, 9:12].set(sigma_gyro_bias**2 * jnp.eye(3))
    
    # Accel bias (indices 12:15)
    Q = Q.at[12:15, 12:15].set(sigma_accel_bias**2 * jnp.eye(3))
    
    # Time offset (index 15)
    Q = Q.at[15, 15].set(sigma_dt**2)
    
    # Extrinsic (indices 16:22)
    Q = Q.at[16:22, 16:22].set(sigma_extrinsic**2 * jnp.eye(6))
    
    return Q


# =============================================================================
# Main Operator
# =============================================================================


def predict_diffusion(
    belief_prev: BeliefGaussianInfo,
    Q: jnp.ndarray,
    dt_sec: float,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> Tuple[BeliefGaussianInfo, CertBundle, ExpectedEffect]:
    """
    Predict belief forward with dt-scaled process noise.
    
    Always applies DomainProjectionPSD (even if Q is zero).
    
    Prediction in information form:
    1. Convert to moment form
    2. Add dt * Q to covariance
    3. Convert back to information form
    4. Apply DomainProjectionPSD
    
    Args:
        belief_prev: Previous belief
        Q: Process noise matrix (D_Z, D_Z)
        dt_sec: Time delta in seconds
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        
    Returns:
        Tuple of (predicted_belief, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.2
    """
    Q = jnp.asarray(Q, dtype=jnp.float64)
    dt_sec = float(dt_sec)
    
    # Step 1: Convert to moment form
    mean_prev, cov_prev, lift_prev = belief_prev.to_moments(eps_lift)
    
    # Step 2: Add scaled process noise (always, dt may be 0)
    cov_pred_raw = cov_prev + dt_sec * Q
    
    # Step 3: Project predicted covariance to PSD (always)
    cov_pred_result = domain_projection_psd(cov_pred_raw, eps_psd)
    cov_pred = cov_pred_result.M_psd
    
    # Step 4: Invert to get predicted information matrix
    L_pred, lift_inv = spd_cholesky_inverse_lifted(cov_pred, eps_lift)
    
    # Step 5: Project L to PSD (always)
    L_pred_result = domain_projection_psd(L_pred, eps_psd)
    L_pred_psd = L_pred_result.M_psd
    
    # Step 6: Compute h_pred = L_pred @ mean_prev
    h_pred = L_pred_psd @ mean_prev
    
    # Total projection delta
    total_projection_delta = cov_pred_result.projection_delta + L_pred_result.projection_delta
    
    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=belief_prev.chart_id,
        anchor_id=belief_prev.anchor_id,
        triggers=["PredictDiffusion"],
        conditioning=ConditioningCert(
            eig_min=L_pred_result.conditioning.eig_min,
            eig_max=L_pred_result.conditioning.eig_max,
            cond=L_pred_result.conditioning.cond,
            near_null_count=L_pred_result.conditioning.near_null_count,
        ),
        influence=InfluenceCert(
            lift_strength=lift_prev + lift_inv,
            psd_projection_delta=total_projection_delta,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=dt_sec,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    
    # Build predicted belief
    belief_pred = BeliefGaussianInfo(
        chart_id=belief_prev.chart_id,
        anchor_id=belief_prev.anchor_id,
        X_anchor=belief_prev.X_anchor,
        stamp_sec=belief_prev.stamp_sec + dt_sec,
        z_lin=belief_prev.z_lin,
        L=L_pred_psd,
        h=h_pred,
        cert=cert,
    )
    
    # Expected effect: predicted covariance trace
    predicted_cov_trace = float(jnp.trace(cov_pred))
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_cov_trace",
        predicted=predicted_cov_trace,
        realized=None,
    )
    
    return belief_pred, cert, expected_effect
