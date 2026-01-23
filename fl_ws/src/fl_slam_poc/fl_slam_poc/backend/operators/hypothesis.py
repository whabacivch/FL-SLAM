"""
HypothesisBarycenterProjection operator for Golden Child SLAM v2.

Combines K hypotheses into a single belief for publishing.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.15
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import BeliefGaussianInfo, D_Z, CHART_ID_GC_RIGHT_01
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    SupportCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_solve_lifted,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class HypothesisProjectionResult:
    """Result of HypothesisBarycenterProjection operator."""
    belief_out: BeliefGaussianInfo
    floor_adjustment: float  # Total weight floor adjustment


# =============================================================================
# Main Operator
# =============================================================================


def hypothesis_barycenter_projection(
    hypotheses: List[BeliefGaussianInfo],
    weights: jnp.ndarray,
    K_HYP: int = constants.GC_K_HYP,
    HYP_WEIGHT_FLOOR: float = constants.GC_HYP_WEIGHT_FLOOR,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> Tuple[HypothesisProjectionResult, CertBundle, ExpectedEffect]:
    """
    Combine K hypotheses into a single belief via barycenter projection.
    
    Always:
    1. Enforce weight floor continuously
    2. Renormalize weights
    3. Barycenter in information form
    4. Apply DomainProjectionPSD
    
    Args:
        hypotheses: List of K_HYP beliefs
        weights: Hypothesis weights (K_HYP,)
        K_HYP: Number of hypotheses (default from constants)
        HYP_WEIGHT_FLOOR: Minimum weight (default from constants)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        
    Returns:
        Tuple of (HypothesisProjectionResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.15
    """
    weights = jnp.asarray(weights, dtype=jnp.float64)
    
    if len(hypotheses) != K_HYP:
        raise ValueError(f"Expected {K_HYP} hypotheses, got {len(hypotheses)}")
    if weights.shape != (K_HYP,):
        raise ValueError(f"Expected weights shape ({K_HYP},), got {weights.shape}")
    
    # Step 1: Enforce weight floor (continuous, no branching)
    weights_floored = jnp.maximum(weights, HYP_WEIGHT_FLOOR)
    floor_adjustment = float(jnp.sum(jnp.abs(weights_floored - weights)))
    
    # Step 2: Renormalize
    weights_normalized = weights_floored / jnp.sum(weights_floored)
    
    # Step 3: Barycenter in information form
    # L_out = sum_j w_j L_j
    # h_out = sum_j w_j h_j
    L_out_raw = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    h_out = jnp.zeros(D_Z, dtype=jnp.float64)
    
    for j in range(K_HYP):
        w_j = weights_normalized[j]
        L_out_raw = L_out_raw + w_j * hypotheses[j].L
        h_out = h_out + w_j * hypotheses[j].h
    
    # Step 4: Apply DomainProjectionPSD (always)
    L_psd_result = domain_projection_psd(L_out_raw, eps_psd)
    L_out = L_psd_result.M_psd
    
    # Use the first hypothesis as template for anchor/chart info
    template = hypotheses[0]
    
    # Compute weighted mean anchor pose
    # For simplicity, use the first hypothesis anchor
    # (Full implementation would do proper SE3 barycenter)
    X_anchor_out = template.X_anchor
    
    # Compute weighted mean linearization point
    z_lin_out = jnp.zeros(D_Z, dtype=jnp.float64)
    for j in range(K_HYP):
        w_j = weights_normalized[j]
        z_lin_out = z_lin_out + w_j * hypotheses[j].z_lin
    
    # Build output belief
    cert_out = CertBundle.create_approx(
        chart_id=CHART_ID_GC_RIGHT_01,
        anchor_id=template.anchor_id,
        triggers=["HypothesisProjection"],
        conditioning=ConditioningCert(
            eig_min=L_psd_result.conditioning.eig_min,
            eig_max=L_psd_result.conditioning.eig_max,
            cond=L_psd_result.conditioning.cond,
            near_null_count=L_psd_result.conditioning.near_null_count,
        ),
        support=SupportCert(
            ess_total=float(1.0 / jnp.sum(weights_normalized ** 2)),  # ESS
            support_frac=float(jnp.sum(weights_normalized > HYP_WEIGHT_FLOOR) / K_HYP),
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=L_psd_result.projection_delta,
            mass_epsilon_ratio=floor_adjustment / K_HYP,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    
    belief_out = BeliefGaussianInfo(
        chart_id=CHART_ID_GC_RIGHT_01,
        anchor_id=template.anchor_id,
        X_anchor=X_anchor_out,
        stamp_sec=template.stamp_sec,
        z_lin=z_lin_out,
        L=L_out,
        h=h_out,
        cert=cert_out,
    )
    
    # Build result
    result = HypothesisProjectionResult(
        belief_out=belief_out,
        floor_adjustment=floor_adjustment,
    )
    
    # Compute spread proxy for expected effect
    # Weighted variance of mean increments
    means = []
    for j in range(K_HYP):
        mu_j = spd_cholesky_solve_lifted(hypotheses[j].L, hypotheses[j].h, eps_lift).x
        means.append(mu_j)
    
    means_stack = jnp.stack(means, axis=0)  # (K_HYP, D_Z)
    mean_of_means = jnp.sum(weights_normalized[:, None] * means_stack, axis=0)
    
    spread_proxy = 0.0
    for j in range(K_HYP):
        delta = means[j] - mean_of_means
        spread_proxy += float(weights_normalized[j]) * float(jnp.dot(delta, delta))
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_projection_spread_proxy",
        predicted=spread_proxy,
        realized=None,
    )
    
    return result, cert_out, expected_effect
