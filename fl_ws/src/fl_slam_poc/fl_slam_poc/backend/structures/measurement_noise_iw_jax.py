"""
Measurement-noise inverse-Wishart state containers (arrays-only) for GC v2.

These IW states represent sensor measurement covariances Σ as random variables:
  Σ ~ InvWishart(Psi, nu)

This module is arrays-only and safe for @jax.jit codepaths.
"""

from __future__ import annotations

from typing import NamedTuple

from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common import constants as C


# Measurement-noise blocks (per-sensor, Phase 1 of measurement IW):
# [gyro(3), accel(3), lidar_translation(3)]
MEAS_BLOCK_DIMS = jnp.array([3, 3, 3], dtype=jnp.int32)  # (3,)


class MeasurementNoiseIWState(NamedTuple):
    """Arrays-only measurement-noise IW state (per-sensor)."""

    nu: jnp.ndarray  # (3,) float64
    Psi_blocks: jnp.ndarray  # (3, 3, 3) float64
    block_dims: jnp.ndarray  # (3,) int32


def create_datasheet_measurement_noise_state() -> MeasurementNoiseIWState:
    """
    Initialize measurement-noise IW states from constants.py.

    Uses the same ν convention as process noise:
      nu = p + 1 + nu_extra,  Psi = Sigma_prior * nu_extra
    """
    block_dims = MEAS_BLOCK_DIMS
    p = block_dims.astype(jnp.float64)
    nu_extra = jnp.array(C.GC_IW_NU_WEAK_ADD, dtype=jnp.float64)
    nu = p + 1.0 + nu_extra  # (3,)

    Sigma_gyro = C.GC_IMU_GYRO_NOISE_DENSITY * jnp.eye(3, dtype=jnp.float64)
    Sigma_accel = C.GC_IMU_ACCEL_NOISE_DENSITY * jnp.eye(3, dtype=jnp.float64)
    Sigma_lidar = C.GC_LIDAR_SIGMA_MEAS * jnp.eye(3, dtype=jnp.float64)

    Psi_blocks = jnp.stack(
        [
            Sigma_gyro * nu_extra,
            Sigma_accel * nu_extra,
            Sigma_lidar * nu_extra,
        ],
        axis=0,
    )

    return MeasurementNoiseIWState(nu=nu, Psi_blocks=Psi_blocks, block_dims=block_dims)

