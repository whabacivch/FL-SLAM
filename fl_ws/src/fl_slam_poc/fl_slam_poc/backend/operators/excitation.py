"""
Fisher-derived excitation scaling (Contract 1).

Compute soft-coupling scalars from evidence vs prior information and apply them
by scaling prior strength for dt and extrinsic-related rows/cols.
"""

from __future__ import annotations

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants


@jax.jit
def compute_excitation_scales_jax(
    L_evidence: jnp.ndarray,  # (22,22)
    L_prior: jnp.ndarray,     # (22,22)
    eps: float = constants.GC_EXC_EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute normalized soft-coupling scalars:
      s_dt = e_dt / (e_dt + pi_dt + eps)
      s_ex = e_ex / (e_ex + pi_ex + eps)
    """
    e_dt = L_evidence[constants.GC_IDX_DT, constants.GC_IDX_DT]
    e_ex = jnp.trace(L_evidence[constants.GC_IDX_EX, constants.GC_IDX_EX])
    pi_dt = L_prior[constants.GC_IDX_DT, constants.GC_IDX_DT]
    pi_ex = jnp.trace(L_prior[constants.GC_IDX_EX, constants.GC_IDX_EX])
    s_dt = e_dt / (e_dt + pi_dt + eps)
    s_ex = e_ex / (e_ex + pi_ex + eps)
    return s_dt, s_ex


@jax.jit
def apply_excitation_prior_scaling_jax(
    L_prior: jnp.ndarray,
    h_prior: jnp.ndarray,
    s_dt: jnp.ndarray,
    s_ex: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Scale prior information for dt and extrinsic by (1 - s).

    We scale both:
      - rows/cols in L_prior
      - corresponding entries in h_prior
    """
    Lp = jnp.asarray(L_prior, dtype=jnp.float64)
    hp = jnp.asarray(h_prior, dtype=jnp.float64)

    a_dt = 1.0 - s_dt
    a_ex = 1.0 - s_ex

    # dt index 15
    Lp = Lp.at[constants.GC_IDX_DT, :].set(a_dt * Lp[constants.GC_IDX_DT, :])
    Lp = Lp.at[:, constants.GC_IDX_DT].set(a_dt * Lp[:, constants.GC_IDX_DT])
    hp = hp.at[constants.GC_IDX_DT].set(a_dt * hp[constants.GC_IDX_DT])

    # extrinsic block 16:22
    Lp = Lp.at[constants.GC_IDX_EX, :].set(a_ex * Lp[constants.GC_IDX_EX, :])
    Lp = Lp.at[:, constants.GC_IDX_EX].set(a_ex * Lp[:, constants.GC_IDX_EX])
    hp = hp.at[constants.GC_IDX_EX].set(a_ex * hp[constants.GC_IDX_EX])

    return Lp, hp
