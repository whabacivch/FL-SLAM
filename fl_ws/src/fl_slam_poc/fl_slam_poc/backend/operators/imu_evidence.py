"""
IMU evidence operators for GC v2.

Currently implements the accelerometer direction term as a vMF-style factor:
  ell(delta) = -kappa * dot(mu(delta), xbar)

where:
  mu(delta) = R(delta)^T * (-g_hat)
  xbar is the measured resultant direction over the scan window.

We convert this factor to quadratic Gaussian information by Laplace at delta=0
using closed-form derivatives (intrinsic primitives only; no autodiff).
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
    SupportCert,
    MismatchCert,
)
from fl_slam_poc.common.primitives import domain_projection_psd_core
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_v2


@dataclass
class ImuEvidenceResult:
    L_imu: jnp.ndarray  # (22,22)
    h_imu: jnp.ndarray  # (22,)
    kappa: float
    ess: float


@jax.jit
def _accel_resultant_direction_jax(
    imu_accel: jnp.ndarray,   # (M,3)
    imu_valid: jnp.ndarray,   # (M,) bool
    weights: jnp.ndarray,     # (M,)
    accel_bias: jnp.ndarray,  # (3,)
    eps_mass: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    valid_f = imu_valid.astype(jnp.float64)
    w = weights * valid_f
    ess = jnp.sum(w)

    a = imu_accel - accel_bias[None, :]
    n = jnp.linalg.norm(a, axis=1, keepdims=True)
    x = a / (n + eps_mass)  # (M,3)

    S = jnp.sum(w[:, None] * x, axis=0)  # (3,)
    S_norm = jnp.linalg.norm(S)
    xbar = S / (S_norm + eps_mass)
    Rbar = S_norm / (ess + eps_mass)
    return xbar, Rbar, ess


def imu_vmf_gravity_evidence(
    rotvec_world_body: jnp.ndarray,  # (3,) rotvec of body in world
    imu_accel: jnp.ndarray,          # (M,3)
    imu_valid: jnp.ndarray,          # (M,)
    weights: jnp.ndarray,            # (M,)
    accel_bias: jnp.ndarray,         # (3,)
    gravity_W: jnp.ndarray,          # (3,)
    eps_psd: float,
    eps_mass: float,
    chart_id: str,
    anchor_id: str,
) -> Tuple[ImuEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Closed-form Laplace/I-projection of the vMF-style accelerometer direction factor onto a 22D Gaussian info term.

    We differentiate only over the 3D rotation perturbation δθ applied on the right:
      R(δθ) = R0 @ Exp(δθ)
    """
    rotvec0 = jnp.asarray(rotvec_world_body, dtype=jnp.float64).reshape(-1)
    R0 = se3_jax.so3_exp(rotvec0)

    g = jnp.asarray(gravity_W, dtype=jnp.float64).reshape(-1)
    g_hat = g / (jnp.linalg.norm(g) + eps_mass)
    minus_g_hat = -g_hat

    xbar, Rbar, ess = _accel_resultant_direction_jax(
        imu_accel=jnp.asarray(imu_accel, dtype=jnp.float64),
        imu_valid=jnp.asarray(imu_valid).reshape(-1),
        weights=jnp.asarray(weights, dtype=jnp.float64).reshape(-1),
        accel_bias=jnp.asarray(accel_bias, dtype=jnp.float64).reshape(-1),
        eps_mass=eps_mass,
    )

    kappa = float(kappa_from_resultant_v2(float(Rbar), eps_r=constants.GC_EPS_R, eps_den=constants.GC_EPS_DEN).kappa)

    # Predicted mean direction in body frame at the linearization point.
    # mu0 = R0^T (-g_hat)
    mu0 = R0.T @ minus_g_hat

    kappa_f = jnp.array(kappa, dtype=jnp.float64)
    x_dot_mu = (xbar @ mu0)

    # Closed-form gradient and Hessian w.r.t. right-perturbation rotation δθ:
    #   g = ∂/∂δθ (-κ mu^T xbar) |0 = -κ (mu0 × xbar)
    #   H ≈ κ [ (x·mu) I - 0.5 (x mu^T + mu x^T) ]
    g_rot = -kappa_f * jnp.cross(mu0, xbar)
    I3 = jnp.eye(3, dtype=jnp.float64)
    H_rot = kappa_f * (x_dot_mu * I3 - 0.5 * (jnp.outer(xbar, mu0) + jnp.outer(mu0, xbar)))
    H_rot = 0.5 * (H_rot + H_rot.T)
    H_rot_psd, _ = domain_projection_psd_core(H_rot, eps_psd)

    L = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    L = L.at[0:3, 0:3].set(H_rot_psd)
    h = jnp.zeros((D_Z,), dtype=jnp.float64)
    h = h.at[0:3].set(-g_rot)

    nll_proxy = float(-kappa_f * (mu0 @ xbar))
    eigvals = jnp.linalg.eigvalsh(H_rot_psd)
    eig_min = float(jnp.min(eigvals))
    eig_max = float(jnp.max(eigvals))
    cond = eig_max / max(eig_min, 1e-18)

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ImuAccelDirectionClosedFormLaplace"],
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=int(jnp.sum(eigvals < eps_psd)),
        ),
        support=SupportCert(ess_total=float(ess), support_frac=1.0),
        mismatch=MismatchCert(nll_per_ess=nll_proxy / (float(ess) + eps_mass), directional_score=float(Rbar)),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    effect = ExpectedEffect(
        objective_name="imu_accel_direction_nll_proxy",
        predicted=nll_proxy,
        realized=None,
    )

    return ImuEvidenceResult(L_imu=L, h_imu=h, kappa=kappa, ess=float(ess)), cert, effect

