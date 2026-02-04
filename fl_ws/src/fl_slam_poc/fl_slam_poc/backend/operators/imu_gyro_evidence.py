"""
Gyro rotation evidence operator (Gaussian on SO(3)) for GC v2.

This is a unary factor on the scan-end orientation, using the scan-start
orientation (from the previous belief) as a known anchor plus IMU preintegrated
relative rotation over the scan window.
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
from fl_slam_poc.common.primitives import (
    domain_projection_psd_core,
    spd_cholesky_inverse_lifted_core,
)
from fl_slam_poc.common.geometry import se3_jax


@dataclass
class ImuGyroEvidenceResult:
    L_gyro: jnp.ndarray  # (22,22)
    h_gyro: jnp.ndarray  # (22,)
    r_rot: jnp.ndarray   # (3,) rotation residual


def _imu_gyro_rotation_evidence_jax(
    rotvec_start_WB: jnp.ndarray,
    rotvec_end_pred_WB: jnp.ndarray,
    delta_rotvec_meas: jnp.ndarray,
    Sigma_g: jnp.ndarray,
    dt_int: jnp.ndarray,  # scalar 0-d
    eps_psd: float,
    eps_lift: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled core: returns (L, h, r_rot, nll_proxy, eig_min, eig_max, near_null_count, lift_strength).
    All outputs are JAX arrays (scalars as 0-d) for JIT compatibility.
    """
    rotvec_start_WB = jnp.asarray(rotvec_start_WB, dtype=jnp.float64).reshape(-1)
    rotvec_end_pred_WB = jnp.asarray(rotvec_end_pred_WB, dtype=jnp.float64).reshape(-1)
    delta_rotvec_meas = jnp.asarray(delta_rotvec_meas, dtype=jnp.float64).reshape(-1)
    Sigma_g = jnp.asarray(Sigma_g, dtype=jnp.float64)
    dt_pos = jnp.maximum(jnp.asarray(dt_int, dtype=jnp.float64), 0.0)

    R_start = se3_jax.so3_exp(rotvec_start_WB)
    R_delta = se3_jax.so3_exp(delta_rotvec_meas)
    R_end_imu = R_start @ R_delta
    R_end_pred = se3_jax.so3_exp(rotvec_end_pred_WB)
    R_diff = R_end_pred.T @ R_end_imu
    r_rot = se3_jax.so3_log(R_diff)

    eps_mass = constants.GC_EPS_MASS
    dt_eff = dt_pos + eps_mass
    mass_scale = dt_pos / dt_eff

    Sigma_rot = Sigma_g * dt_eff
    Sigma_rot_psd, _ = domain_projection_psd_core(Sigma_rot, eps_psd)
    L_rot, lift_strength = spd_cholesky_inverse_lifted_core(Sigma_rot_psd, eps_lift)
    L_rot_scaled = mass_scale * L_rot

    L = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    L = L.at[constants.GC_IDX_ROT, constants.GC_IDX_ROT].set(L_rot_scaled)
    h = jnp.zeros((D_Z,), dtype=jnp.float64)
    h = h.at[constants.GC_IDX_ROT].set(L_rot_scaled @ r_rot)

    nll_proxy = 0.5 * (r_rot @ L_rot @ r_rot)
    L_rot_psd, _ = domain_projection_psd_core(L_rot, eps_psd)
    eigvals = jnp.linalg.eigvalsh(L_rot_psd)
    eig_min = jnp.min(eigvals)
    eig_max = jnp.max(eigvals)
    near_null_count = jnp.sum(eigvals < eps_psd)
    return L, h, r_rot, nll_proxy, eig_min, eig_max, near_null_count, lift_strength


@jax.jit
def _imu_gyro_rotation_evidence_jit(
    rotvec_start_WB: jnp.ndarray,
    rotvec_end_pred_WB: jnp.ndarray,
    delta_rotvec_meas: jnp.ndarray,
    Sigma_g: jnp.ndarray,
    dt_int: jnp.ndarray,
    eps_psd: float,
    eps_lift: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return _imu_gyro_rotation_evidence_jax(
        rotvec_start_WB, rotvec_end_pred_WB, delta_rotvec_meas, Sigma_g,
        dt_int, eps_psd, eps_lift,
    )


def imu_gyro_rotation_evidence(
    rotvec_start_WB: jnp.ndarray,       # (3,) scan-start orientation
    rotvec_end_pred_WB: jnp.ndarray,    # (3,) scan-end predicted orientation
    delta_rotvec_meas: jnp.ndarray,     # (3,) IMU-preintegrated relative rotvec over scan
    Sigma_g: jnp.ndarray,               # (3,3) gyro noise covariance proxy
    dt_int: float,                      # Sum of actual IMU sample intervals (bag-agnostic)
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[ImuGyroEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Build a Gaussian evidence term on the 22D tangent rotation block from gyro preintegration.

    We form a predicted scan-end orientation from the scan-start orientation and IMU delta:
      R_end_imu = R_start * Exp(delta_rotvec_meas)

    Residual on SO(3) (right-perturbation, measurement-target form):
      r = Log( R_end_pred^T * R_end_imu )

    Covariance on r is approximated as:
      Sigma_rot ≈ Sigma_g * dt_int
    where dt_int = Σ_i Δt_i over actual IMU sample intervals (bag-agnostic definition).
    
    Continuous mass check: if dt_int ≈ 0 (no samples), evidence is ~0 without boolean gates.
    Core computation is JIT-compiled; cert/effect are built after one host sync.
    """
    dt_int_j = jnp.array(dt_int, dtype=jnp.float64)
    L, h, r_rot, nll_proxy, eig_min, eig_max, near_null_count, lift_strength = _imu_gyro_rotation_evidence_jit(
        rotvec_start_WB, rotvec_end_pred_WB, delta_rotvec_meas, Sigma_g,
        dt_int_j, eps_psd, eps_lift,
    )
    # Single sync: materialize scalars for cert/effect (no NumPy; no gates).
    nll_p = float(nll_proxy)
    e_min = float(eig_min)
    e_max = float(eig_max)
    cond = e_max / max(e_min, 1e-18)
    n_near = int(near_null_count)
    lift = float(lift_strength)

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ImuGyroRotationGaussian"],
        conditioning=ConditioningCert(
            eig_min=e_min,
            eig_max=e_max,
            cond=cond,
            near_null_count=n_near,
        ),
        mismatch=MismatchCert(nll_per_ess=nll_p, directional_score=0.0),
        influence=InfluenceCert.identity().with_overrides(
            lift_strength=lift,
        ),
    )
    effect = ExpectedEffect(
        objective_name="imu_gyro_rotation_nll_proxy",
        predicted=nll_p,
        realized=None,
    )
    return ImuGyroEvidenceResult(L_gyro=L, h_gyro=h, r_rot=r_rot), cert, effect
