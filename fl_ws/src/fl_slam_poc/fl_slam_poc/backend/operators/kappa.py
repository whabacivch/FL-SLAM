"""
KappaFromResultant operator for Golden Child SLAM v2.

Single continuous formula for vMF concentration from resultant length.
No piecewise approximations.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.6
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
)
from fl_slam_poc.common.primitives import clamp


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class KappaResult:
    """Result of KappaFromResultant operator."""
    kappa: float
    R_clamped: float  # Clamped resultant length
    clamp_delta: float  # Amount clamped


# =============================================================================
# Main Operator
# =============================================================================


def _kappa_continuous_formula(R: float, d: int = 3) -> float:
    """
    Single continuous formula for kappa from resultant length R.
    
    Uses the approximation: kappa ≈ R * (d - R^2) / (1 - R^2)
    
    This is valid for all R in (0, 1) and is numerically stable.
    
    Args:
        R: Mean resultant length in (0, 1)
        d: Dimension (default 3 for S^2)
        
    Returns:
        Concentration parameter kappa
    """
    # Pure Python implementation (no JIT needed for scalar ops)
    R = float(R)
    d = float(d)
    
    # Numerator: R * (d - R^2)
    numerator = R * (d - R * R)
    
    # Denominator: 1 - R^2 + eps for numerical stability
    denominator = 1.0 - R * R + constants.GC_EPS_R
    
    kappa = numerator / denominator
    
    return kappa


@jax.jit
def kappa_from_resultant_batch(
    R_bar: jnp.ndarray,
    eps_r: float = constants.GC_EPS_R,
    d: int = 3,
) -> jnp.ndarray:
    """
    Batched kappa computation for arrays of resultant lengths.
    
    Pure JAX implementation - no host sync, fully vectorized.
    
    Args:
        R_bar: Mean resultant lengths (B,) in [0, 1)
        eps_r: Small epsilon to keep R away from 1
        d: Dimension (default 3 for S^2)
        
    Returns:
        kappa values (B,)
    """
    R_bar = jnp.asarray(R_bar, dtype=jnp.float64)
    
    # Clamp R to valid range (continuous, always applied)
    R_clamped = jnp.clip(R_bar, 0.0, 1.0 - eps_r)
    
    # Vectorized kappa formula: kappa = R * (d - R^2) / (1 - R^2 + eps)
    R2 = R_clamped * R_clamped
    numerator = R_clamped * (d - R2)
    denominator = 1.0 - R2 + eps_r
    kappa = numerator / denominator
    
    return kappa


def kappa_from_resultant_v2(
    R_bar: float,
    eps_r: float = constants.GC_EPS_R,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[KappaResult, CertBundle, ExpectedEffect]:
    """
    Compute vMF concentration from mean resultant length.
    
    Uses single continuous formula - no piecewise approximations.
    
    Args:
        R_bar: Mean resultant length (should be in [0, 1))
        eps_r: Small epsilon to keep R away from 1
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (KappaResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.6
    """
    # Clamp R to valid range (continuous, always applied)
    R_clamp_result = clamp(float(R_bar), 0.0, 1.0 - eps_r)
    R_clamped = R_clamp_result.value
    clamp_delta = R_clamp_result.clamp_delta
    
    # Apply continuous formula
    kappa = _kappa_continuous_formula(R_clamped)
    
    # Build result
    result = KappaResult(
        kappa=kappa,
        R_clamped=R_clamped,
        clamp_delta=clamp_delta,
    )
    
    # This is an exact operation (closed-form formula)
    cert = CertBundle.create_exact(
        chart_id=chart_id,
        anchor_id=anchor_id,
    )
    
    expected_effect = ExpectedEffect(
        objective_name="kappa",
        predicted=kappa,
        realized=None,
    )
    
    return result, cert, expected_effect
