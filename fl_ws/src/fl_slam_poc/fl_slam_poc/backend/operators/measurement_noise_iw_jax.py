"""
Measurement-noise inverse-Wishart operators (arrays-only JAX).

This module provides:
- per-sensor Σ estimates (IW posterior means)
- commutative sufficient-statistics updates from residual outer products
- deterministic IW state updates (once per scan)
"""

from __future__ import annotations

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants as C
from fl_slam_poc.common.primitives import domain_projection_psd_core
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.backend.structures.measurement_noise_iw_jax import (
    MeasurementNoiseIWState,
    MEAS_BLOCK_DIMS,
)


MEAS_BLOCK_MASKS = jnp.eye(3, dtype=jnp.float64)[None, :, :]  # (1,3,3) for clarity; blocks are full 3x3


@jax.jit
def measurement_noise_mean_jax(
    mn_state: MeasurementNoiseIWState,
    idx: int,
) -> jnp.ndarray:
    """IW mean Σ_hat for block idx (0=gyro, 1=accel, 2=lidar)."""
    p = mn_state.block_dims[idx].astype(jnp.float64)
    denom = mn_state.nu[idx] - p - 1.0
    denom = jnp.maximum(denom, 1e-12)
    Sigma = mn_state.Psi_blocks[idx] / denom
    Sigma_psd, _ = domain_projection_psd_core(Sigma, C.GC_EPS_PSD)
    return Sigma_psd


@jax.jit
def measurement_noise_apply_suffstats_jax(
    mn_state: MeasurementNoiseIWState,
    dPsi_blocks: jnp.ndarray,  # (3,3,3)
    dnu: jnp.ndarray,          # (3,)
    eps_psd: float = C.GC_EPS_PSD,
    nu_max: float = 1000.0,
) -> MeasurementNoiseIWState:
    """
    Apply aggregated measurement-noise IW sufficient statistics (once per scan).

    Forgetful retention (per sensor):
      Psi <- rho * Psi + dPsi
      nu  <- rho * nu  + dnu
    """
    rho = jnp.array(
        [C.GC_IW_RHO_MEAS_GYRO, C.GC_IW_RHO_MEAS_ACCEL, C.GC_IW_RHO_MEAS_LIDAR],
        dtype=jnp.float64,
    )

    Psi_raw = rho[:, None, None] * mn_state.Psi_blocks + dPsi_blocks
    Psi_raw = 0.5 * (Psi_raw + jnp.swapaxes(Psi_raw, -1, -2))

    def proj(P):
        P_psd, _ = domain_projection_psd_core(P, eps_psd)
        return P_psd

    Psi_psd = jax.vmap(proj)(Psi_raw)

    dims_f = mn_state.block_dims.astype(jnp.float64)
    nu_raw = rho * mn_state.nu + dnu
    nu_min = dims_f + 1.0 + C.GC_IW_NU_WEAK_ADD
    nu = jnp.clip(jnp.maximum(nu_raw, nu_min), nu_min, nu_max)

    return MeasurementNoiseIWState(nu=nu, Psi_blocks=Psi_psd, block_dims=mn_state.block_dims)


@jax.jit
def lidar_meas_iw_suffstats_from_translation_residuals_jax(
    residuals: jnp.ndarray,  # (B,3)
    weights: jnp.ndarray,    # (B,)
    eps_mass: float = C.GC_EPS_MASS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build commutative IW sufficient statistics for LiDAR translation measurement noise from per-bin residuals.

    We use normalized weights (so dPsi is an average outer product) and dnu=1 per scan.
    """
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)
    residuals = jnp.asarray(residuals, dtype=jnp.float64)

    w_sum = jnp.sum(weights) + eps_mass
    w_norm = weights / w_sum
    # dPsi = Σ_b w_norm[b] * r_b r_b^T
    rrT = jnp.einsum("b,bi,bj->ij", w_norm, residuals, residuals)
    rrT = 0.5 * (rrT + rrT.T)
    rrT_psd, _ = domain_projection_psd_core(rrT, C.GC_EPS_PSD)

    dPsi_blocks = jnp.zeros((3, 3, 3), dtype=jnp.float64)
    dPsi_blocks = dPsi_blocks.at[2].set(rrT_psd)  # lidar block index 2
    dnu = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
    return dPsi_blocks, dnu


@jax.jit
def imu_gyro_meas_iw_suffstats_from_avg_rate_jax(
    imu_gyro: jnp.ndarray,    # (M,3)
    imu_valid: jnp.ndarray,   # (M,)
    weights: jnp.ndarray,     # (M,)
    gyro_bias: jnp.ndarray,   # (3,)
    omega_avg: jnp.ndarray,   # (3,)
    eps_mass: float = C.GC_EPS_MASS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gyro measurement-noise IW sufficient stats from residuals against an average angular rate proxy.

    Residual per sample:
      r_g = (gyro_i - bg) - omega_avg
    """
    imu_gyro = jnp.asarray(imu_gyro, dtype=jnp.float64)
    imu_valid = jnp.asarray(imu_valid).reshape(-1)
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)
    gyro_bias = jnp.asarray(gyro_bias, dtype=jnp.float64).reshape(-1)
    omega_avg = jnp.asarray(omega_avg, dtype=jnp.float64).reshape(-1)

    w = weights * imu_valid.astype(jnp.float64)
    w_sum = jnp.sum(w) + eps_mass
    w_norm = w / w_sum

    r = (imu_gyro - gyro_bias[None, :]) - omega_avg[None, :]
    rrT = jnp.einsum("m,mi,mj->ij", w_norm, r, r)
    rrT = 0.5 * (rrT + rrT.T)
    rrT_psd, _ = domain_projection_psd_core(rrT, C.GC_EPS_PSD)

    dPsi_blocks = jnp.zeros((3, 3, 3), dtype=jnp.float64)
    dPsi_blocks = dPsi_blocks.at[0].set(rrT_psd)  # gyro block index 0
    dnu = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
    return dPsi_blocks, dnu


@jax.jit
def imu_accel_meas_iw_suffstats_from_gravity_dir_jax(
    rotvec_world_body: jnp.ndarray,  # (3,)
    imu_accel: jnp.ndarray,          # (M,3)
    imu_valid: jnp.ndarray,          # (M,)
    weights: jnp.ndarray,            # (M,)
    accel_bias: jnp.ndarray,         # (3,)
    gravity_W: jnp.ndarray,          # (3,)
    eps_mass: float = C.GC_EPS_MASS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Accel measurement-noise IW sufficient stats from directional residuals against predicted gravity direction.

    Predicted direction in body:
      mu = R^T (-g_hat)

    Measured direction (bias-corrected):
      x_i = normalize(accel_i - ba)

    Residual (3D):
      r_a = x_i - mu
    """
    rotvec0 = jnp.asarray(rotvec_world_body, dtype=jnp.float64).reshape(-1)
    R0 = se3_jax.so3_exp(rotvec0)

    g = jnp.asarray(gravity_W, dtype=jnp.float64).reshape(-1)
    g_hat = g / (jnp.linalg.norm(g) + eps_mass)
    mu = R0.T @ (-g_hat)

    imu_accel = jnp.asarray(imu_accel, dtype=jnp.float64)
    imu_valid = jnp.asarray(imu_valid).reshape(-1)
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)
    accel_bias = jnp.asarray(accel_bias, dtype=jnp.float64).reshape(-1)

    w = weights * imu_valid.astype(jnp.float64)
    w_sum = jnp.sum(w) + eps_mass
    w_norm = w / w_sum

    a = imu_accel - accel_bias[None, :]
    n = jnp.linalg.norm(a, axis=1, keepdims=True)
    x = a / (n + eps_mass)

    r = x - mu[None, :]
    rrT = jnp.einsum("m,mi,mj->ij", w_norm, r, r)
    rrT = 0.5 * (rrT + rrT.T)
    rrT_psd, _ = domain_projection_psd_core(rrT, C.GC_EPS_PSD)

    dPsi_blocks = jnp.zeros((3, 3, 3), dtype=jnp.float64)
    dPsi_blocks = dPsi_blocks.at[1].set(rrT_psd)  # accel block index 1
    dnu = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
    return dPsi_blocks, dnu

