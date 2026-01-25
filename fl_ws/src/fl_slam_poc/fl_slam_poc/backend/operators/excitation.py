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
    e_dt = L_evidence[15, 15]
    e_ex = jnp.trace(L_evidence[16:22, 16:22])
    pi_dt = L_prior[15, 15]
    pi_ex = jnp.trace(L_prior[16:22, 16:22])
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
    Lp = Lp.at[15, :].set(a_dt * Lp[15, :])
    Lp = Lp.at[:, 15].set(a_dt * Lp[:, 15])
    hp = hp.at[15].set(a_dt * hp[15])

    # extrinsic block 16:22
    Lp = Lp.at[16:22, :].set(a_ex * Lp[16:22, :])
    Lp = Lp.at[:, 16:22].set(a_ex * Lp[:, 16:22])
    hp = hp.at[16:22].set(a_ex * hp[16:22])

    return Lp, hp

