"""
WahbaSVD operator for Golden Child SLAM v2.

Optimal rotation estimation from weighted direction correspondences.
Zero-weight bins contribute nothing by identity.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.7
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
    MismatchCert,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class WahbaResult:
    """Result of WahbaSVD operator."""
    R_hat: jnp.ndarray  # (3, 3) optimal rotation
    cost: float  # Wahba cost
    det_sign: float  # Sign of determinant (for reflection handling)


# =============================================================================
# Main Operator
# =============================================================================


def _wahba_svd_core(
    mu_map: jnp.ndarray,
    mu_scan: jnp.ndarray,
    weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Core Wahba SVD computation using JAX.
    
    Args:
        mu_map: Map directions (B, 3)
        mu_scan: Scan directions (B, 3)
        weights: Per-bin weights (B,)
        
    Returns:
        Tuple of (R_hat, cost, det_sign) - all as JAX arrays
    """
    # Build attitude profile matrix B = Σ w_b * mu_map_b @ mu_scan_b^T
    # Use einsum for vectorized computation
    B = jnp.einsum('b,bi,bj->ij', weights, mu_map, mu_scan)
    
    # SVD of B
    U, S, Vt = jnp.linalg.svd(B, full_matrices=True)
    
    # Optimal rotation: R = U @ diag(1, 1, det(U @ Vt)) @ Vt
    det_UVt = jnp.linalg.det(U @ Vt)
    det_sign = jnp.sign(det_UVt)
    
    # Handle reflection (det < 0) by flipping sign of smallest singular value direction
    diag_correction = jnp.array([1.0, 1.0, det_sign], dtype=jnp.float64)
    R_hat = U @ jnp.diag(diag_correction) @ Vt
    
    # Wahba cost = Σ w_b - trace(R_hat @ B)
    # Lower is better
    total_weight = jnp.sum(weights)
    cost = total_weight - jnp.trace(R_hat @ B.T)
    
    return R_hat, cost, det_sign


def wahba_svd(
    mu_map: jnp.ndarray,
    mu_scan: jnp.ndarray,
    weights: jnp.ndarray,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[WahbaResult, CertBundle, ExpectedEffect]:
    """
    Solve Wahba's problem using SVD.
    
    Finds optimal rotation R such that:
        minimize Σ w_b ||mu_map_b - R @ mu_scan_b||^2
    
    Zero-weight bins contribute nothing (by construction of weighted sum).
    
    Args:
        mu_map: Map mean directions (B, 3)
        mu_scan: Scan mean directions (B, 3)
        weights: Per-bin weights (B,) - typically N * kappa_map * kappa_scan
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (WahbaResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.7
    """
    mu_map = jnp.asarray(mu_map, dtype=jnp.float64)
    mu_scan = jnp.asarray(mu_scan, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    
    n_bins = weights.shape[0]
    
    # Core SVD computation
    R_hat, cost_jax, det_sign_jax = _wahba_svd_core(mu_map, mu_scan, weights)
    
    # Convert to Python floats outside the JIT context
    cost = float(cost_jax)
    det_sign = float(det_sign_jax)
    
    # Build result
    result = WahbaResult(
        R_hat=R_hat,
        cost=cost,
        det_sign=det_sign,
    )
    
    # Compute support metrics
    total_weight = float(jnp.sum(weights))
    nonzero_bins = float(jnp.sum(weights > constants.GC_EPS_MASS))
    
    # Normalized cost (lower is better)
    normalized_cost = cost / (total_weight + constants.GC_EPS_MASS)
    
    # This is an exact operation (closed-form SVD)
    cert = CertBundle.create_exact(
        chart_id=chart_id,
        anchor_id=anchor_id,
        support=SupportCert(
            ess_total=total_weight,
            support_frac=nonzero_bins / n_bins,
        ),
        mismatch=MismatchCert(
            nll_per_ess=normalized_cost,
            directional_score=1.0 - normalized_cost,  # Higher is better
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="wahba_cost",
        predicted=cost,
        realized=None,
    )
    
    return result, cert, expected_effect
