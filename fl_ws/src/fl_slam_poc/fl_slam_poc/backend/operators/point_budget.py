"""
PointBudgetResample operator for Golden Child SLAM v2.

Deterministically resample points to enforce N_POINTS_CAP.
All points contribute through mass redistribution.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    SupportCert,
    InfluenceCert,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PointBudgetResult:
    """Result of PointBudgetResample operator."""
    points: jnp.ndarray  # (M, 3) resampled points
    timestamps: jnp.ndarray  # (M,) timestamps
    weights: jnp.ndarray  # (M,) adjusted weights
    ring: jnp.ndarray  # (M,) uint8 (0 if unavailable)
    tag: jnp.ndarray   # (M,) uint8 (0 if unavailable)
    n_input: int
    n_output: int
    total_mass_in: float
    total_mass_out: float


# =============================================================================
# Main Operator
# =============================================================================


def point_budget_resample(
    points: jnp.ndarray,
    timestamps: jnp.ndarray,
    weights: jnp.ndarray,
    ring: jnp.ndarray | None = None,
    tag: jnp.ndarray | None = None,
    n_points_cap: int = constants.GC_N_POINTS_CAP,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[PointBudgetResult, CertBundle, ExpectedEffect]:
    """
    Deterministically resample points to enforce budget.
    
    Uses deterministic subsampling with mass preservation.
    All operations are continuous and total.
    
    Args:
        points: Input point positions (N, 3)
        timestamps: Per-point timestamps (N,)
        weights: Per-point weights (N,)
        n_points_cap: Maximum number of points (default from constants)
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (PointBudgetResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.1
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    timestamps = jnp.asarray(timestamps, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    if ring is None:
        ring = jnp.zeros((points.shape[0],), dtype=jnp.uint8)
    else:
        ring = jnp.asarray(ring, dtype=jnp.uint8).reshape(-1)
    if tag is None:
        tag = jnp.zeros((points.shape[0],), dtype=jnp.uint8)
    else:
        tag = jnp.asarray(tag, dtype=jnp.uint8).reshape(-1)
    
    n_input = points.shape[0]
    total_mass_in = float(jnp.sum(weights))
    
    # Compute support fraction (always computed, may be 1.0)
    support_frac = float(jnp.minimum(1.0, n_points_cap / (n_input + constants.GC_EPS_MASS)))
    
    # Deterministic subsampling using stride
    # Always compute stride, apply even if n_input <= n_points_cap
    stride = int(jnp.maximum(1, jnp.ceil(n_input / n_points_cap)))
    indices = jnp.arange(0, n_input, stride)
    n_selected = indices.shape[0]
    
    # Select points using indices
    points_out = points[indices]
    timestamps_out = timestamps[indices]
    weights_raw = weights[indices]
    ring_out = ring[indices]
    tag_out = tag[indices]
    
    # Preserve total mass by scaling weights
    total_mass_selected = float(jnp.sum(weights_raw))
    mass_scale = total_mass_in / (total_mass_selected + constants.GC_EPS_MASS)
    weights_out = weights_raw * mass_scale
    
    total_mass_out = float(jnp.sum(weights_out))
    
    # Build result
    result = PointBudgetResult(
        points=points_out,
        timestamps=timestamps_out,
        weights=weights_out,
        ring=ring_out,
        tag=tag_out,
        n_input=n_input,
        n_output=n_selected,
        total_mass_in=total_mass_in,
        total_mass_out=total_mass_out,
    )
    
    # Compute ESS (effective sample size)
    weights_normalized = weights_out / (total_mass_out + constants.GC_EPS_MASS)
    ess = float(1.0 / jnp.sum(weights_normalized ** 2 + constants.GC_EPS_MASS))
    
    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["PointBudgetResample"],
        support=SupportCert(
            ess_total=ess,
            support_frac=support_frac,
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=constants.GC_EPS_MASS / (total_mass_in + constants.GC_EPS_MASS),
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_ess",
        predicted=ess,
        realized=None,
    )
    
    return result, cert, expected_effect
