"""
IMU preintegration factor for velocity and position evidence.

Provides direct Gaussian evidence on:
  - velocity (state indices 6:9): v_j = v_i + R_i @ Δv_body
  - position (state indices 0:3): p_j = p_i + v_i * dt + R_i @ Δp_body

This complements the gyro rotation evidence by adding constraints on the
translational dynamics from IMU accelerometer integration.

Covariances scale with accelerometer noise:
  - Σ_Δv ∝ Σ_a * dt (velocity from integrated accel)
  - Σ_Δp ∝ Σ_a * dt³ (position from double-integrated accel)

Following the branch-free, gate-free design: when dt_int → 0, evidence → 0
continuously via mass scaling.
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
    MismatchCert,
)
from fl_slam_poc.common.primitives import domain_projection_psd, spd_cholesky_inverse_lifted
from fl_slam_poc.common.geometry import se3_jax


@dataclass
class ImuPreintegrationFactorResult:
    L_imu_preint: jnp.ndarray  # (22, 22)
    h_imu_preint: jnp.ndarray  # (22,)
    r_vel: jnp.ndarray   # (3,) velocity residual
    r_pos: jnp.ndarray   # (3,) position residual


def imu_preintegration_factor(
    # Previous state (treated as known/fixed)
    p_start_world: jnp.ndarray,       # (3,) previous world position
    rotvec_start_WB: jnp.ndarray,     # (3,) previous world orientation
    v_start_world: jnp.ndarray,       # (3,) previous world velocity
    # Predicted current state
    p_end_pred_world: jnp.ndarray,    # (3,) predicted world position
    v_end_pred_world: jnp.ndarray,    # (3,) predicted world velocity
    # IMU preintegration measurements (in start body frame)
    delta_v_body: jnp.ndarray,        # (3,) velocity change from IMU
    delta_p_body: jnp.ndarray,        # (3,) position change from IMU
    # Noise parameters
    Sigma_a: jnp.ndarray,             # (3,3) accel noise covariance proxy
    dt_int: float,                    # IMU integration time
    # Standard parameters
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[ImuPreintegrationFactorResult, CertBundle, ExpectedEffect]:
    """
    Build Gaussian evidence on velocity and position from IMU preintegration.

    The IMU kinematic constraints are:
      v_j = v_i + R_i @ Δv_body
      p_j = p_i + v_i * dt + R_i @ Δp_body

    where Δv_body and Δp_body are the preintegrated quantities in the start body frame.

    Residuals (measurement - prediction form):
      r_v = v_imu - v_pred = (v_i + R_i @ Δv_body) - v_j_pred
      r_p = p_imu - p_pred = (p_i + v_i * dt + R_i @ Δp_body) - p_j_pred

    Covariances scale with accelerometer noise and integration time:
      Σ_v = Σ_a * dt_int (velocity from integrated accel)
      Σ_p = Σ_a * dt_int³ (position from double-integrated accel)

    Continuous mass check: when dt_int → 0, evidence weight → 0 (no gates).
    """
    p_start_world = jnp.asarray(p_start_world, dtype=jnp.float64).reshape(-1)
    rotvec_start_WB = jnp.asarray(rotvec_start_WB, dtype=jnp.float64).reshape(-1)
    v_start_world = jnp.asarray(v_start_world, dtype=jnp.float64).reshape(-1)
    p_end_pred_world = jnp.asarray(p_end_pred_world, dtype=jnp.float64).reshape(-1)
    v_end_pred_world = jnp.asarray(v_end_pred_world, dtype=jnp.float64).reshape(-1)
    delta_v_body = jnp.asarray(delta_v_body, dtype=jnp.float64).reshape(-1)
    delta_p_body = jnp.asarray(delta_p_body, dtype=jnp.float64).reshape(-1)
    Sigma_a = jnp.asarray(Sigma_a, dtype=jnp.float64)

    # Rotation matrix for transforming body-frame quantities to world
    R_start = se3_jax.so3_exp(rotvec_start_WB)

    # IMU-predicted velocity and position in world frame
    delta_v_world = R_start @ delta_v_body
    delta_p_world = R_start @ delta_p_body

    v_imu = v_start_world + delta_v_world
    p_imu = p_start_world + v_start_world * dt_int + delta_p_world

    # Residuals (measurement - prediction form)
    r_vel = v_imu - v_end_pred_world
    r_pos = p_imu - p_end_pred_world

    # Continuous mass scaling (branch-free)
    eps_mass = constants.GC_EPS_MASS
    dt_pos = jnp.maximum(jnp.array(dt_int, dtype=jnp.float64), 0.0)
    dt_eff = dt_pos + eps_mass  # strictly positive
    mass_scale = dt_pos / dt_eff

    # Covariances scale with integration time
    # Velocity: Σ_v = Σ_a * dt (single integration)
    # Position: Σ_p = Σ_a * dt³ (double integration, simplified)
    # Note: The exact covariance involves more complex terms, but this captures
    # the dominant scaling behavior.
    Sigma_v = Sigma_a * dt_eff
    Sigma_p = Sigma_a * (dt_eff ** 3)

    # Project to PSD and invert to get information
    Sigma_v_psd = domain_projection_psd(Sigma_v, eps_psd).M_psd
    Sigma_p_psd = domain_projection_psd(Sigma_p, eps_psd).M_psd
    L_v, lift_v = spd_cholesky_inverse_lifted(Sigma_v_psd, eps_lift)
    L_p, lift_p = spd_cholesky_inverse_lifted(Sigma_p_psd, eps_lift)

    # Scale by continuous mass (branch-free)
    L_v_scaled = mass_scale * L_v
    L_p_scaled = mass_scale * L_p

    # Assemble into 22D state
    # State ordering: [trans(0:3), rot(3:6), vel(6:9), ...]
    L = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    h = jnp.zeros((D_Z,), dtype=jnp.float64)

    # Position evidence at indices 0:3
    L = L.at[0:3, 0:3].set(L_p_scaled)
    h = h.at[0:3].set(L_p_scaled @ r_pos)

    # Velocity evidence at indices 6:9
    L = L.at[6:9, 6:9].set(L_v_scaled)
    h = h.at[6:9].set(L_v_scaled @ r_vel)

    # NLL proxy for diagnostics
    nll_vel = 0.5 * float(r_vel @ L_v @ r_vel)
    nll_pos = 0.5 * float(r_pos @ L_p @ r_pos)
    nll_proxy = nll_vel + nll_pos

    # Conditioning diagnostics (combined)
    eigvals_v = jnp.linalg.eigvalsh(domain_projection_psd(L_v, eps_psd).M_psd)
    eigvals_p = jnp.linalg.eigvalsh(domain_projection_psd(L_p, eps_psd).M_psd)
    eigvals = jnp.concatenate([eigvals_v, eigvals_p])
    eig_min = float(jnp.min(eigvals))
    eig_max = float(jnp.max(eigvals))
    cond = eig_max / max(eig_min, 1e-18)

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ImuPreintegrationVelPos"],
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=int(jnp.sum(eigvals < eps_psd)),
        ),
        mismatch=MismatchCert(nll_per_ess=nll_proxy, directional_score=0.0),
        influence=InfluenceCert(
            lift_strength=float(lift_v + lift_p),
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    effect = ExpectedEffect(
        objective_name="imu_preint_nll_proxy",
        predicted=nll_proxy,
        realized=None,
    )

    return ImuPreintegrationFactorResult(L_imu_preint=L, h_imu_preint=h, r_vel=r_vel, r_pos=r_pos), cert, effect
