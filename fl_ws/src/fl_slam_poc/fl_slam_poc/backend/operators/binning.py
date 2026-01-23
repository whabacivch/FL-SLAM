"""
Binning operators for Golden Child SLAM v2.

Soft assignment and moment matching for directional bins.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Sections 5.4, 5.5
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
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    inv_mass,
    softmax,
    safe_normalize,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class BinSoftAssignResult:
    """Result of BinSoftAssign operator."""
    responsibilities: jnp.ndarray  # (N, B_BINS) soft assignment probabilities


@dataclass
class ScanBinStats:
    """Scan bin sufficient statistics."""
    N: jnp.ndarray  # (B_BINS,) total mass per bin
    s_dir: jnp.ndarray  # (B_BINS, 3) direction resultant vectors
    p_bar: jnp.ndarray  # (B_BINS, 3) centroids
    Sigma_p: jnp.ndarray  # (B_BINS, 3, 3) centroid covariances
    kappa_scan: jnp.ndarray  # (B_BINS,) concentration parameters


# =============================================================================
# Bin Atlas Creation
# =============================================================================


def create_bin_atlas(n_bins: int = constants.GC_B_BINS) -> jnp.ndarray:
    """
    Create Fibonacci lattice bin directions.
    
    Args:
        n_bins: Number of bins
        
    Returns:
        Bin directions (n_bins, 3)
    """
    indices = jnp.arange(n_bins, dtype=jnp.float64) + 0.5
    phi = jnp.arccos(1 - 2 * indices / n_bins)
    theta = jnp.pi * (1 + jnp.sqrt(5)) * indices
    
    x = jnp.sin(phi) * jnp.cos(theta)
    y = jnp.sin(phi) * jnp.sin(theta)
    z = jnp.cos(phi)
    
    dirs = jnp.stack([x, y, z], axis=1)
    
    # Normalize
    norms = jnp.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / (norms + constants.GC_EPS_MASS)
    
    return dirs


# =============================================================================
# Bin Soft Assign Operator
# =============================================================================


def bin_soft_assign(
    point_directions: jnp.ndarray,
    bin_directions: jnp.ndarray,
    tau: float = constants.GC_TAU_SOFT_ASSIGN,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[BinSoftAssignResult, CertBundle, ExpectedEffect]:
    """
    Soft assignment of points to bins using softmax.
    
    Never argmax - uses continuous soft responsibilities.
    
    Args:
        point_directions: Normalized point directions (N, 3)
        bin_directions: Normalized bin directions (B, 3)
        tau: Temperature parameter for softmax
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (BinSoftAssignResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.4
    """
    point_directions = jnp.asarray(point_directions, dtype=jnp.float64)
    bin_directions = jnp.asarray(bin_directions, dtype=jnp.float64)
    
    n_points = point_directions.shape[0]
    n_bins = bin_directions.shape[0]
    
    # Compute similarities (dot products)
    # Shape: (N, B)
    similarities = point_directions @ bin_directions.T
    
    # Apply softmax with temperature to each row
    # Higher similarity -> higher probability
    responsibilities = jnp.zeros((n_points, n_bins), dtype=jnp.float64)
    max_resp = 0.0
    total_entropy = 0.0
    
    for i in range(n_points):
        resp_i = softmax(similarities[i], tau)
        responsibilities = responsibilities.at[i].set(resp_i)
        max_resp = jnp.maximum(max_resp, jnp.max(resp_i))
        # Entropy per point (continuous measure of assignment quality)
        total_entropy = total_entropy - jnp.sum(resp_i * jnp.log(resp_i + constants.GC_EPS_MASS))
    
    avg_entropy = float(total_entropy / (n_points + constants.GC_EPS_MASS))
    
    # Build result
    result = BinSoftAssignResult(responsibilities=responsibilities)
    
    # Build certificate
    cert = CertBundle.create_exact(
        chart_id=chart_id,
        anchor_id=anchor_id,
        support=SupportCert(
            ess_total=float(jnp.exp(avg_entropy)),  # Effective number of bins per point
            support_frac=float(max_resp),
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_assignment_entropy",
        predicted=avg_entropy,
        realized=None,
    )
    
    return result, cert, expected_effect


# =============================================================================
# Scan Bin Moment Match Operator
# =============================================================================


def scan_bin_moment_match(
    points: jnp.ndarray,
    point_covariances: jnp.ndarray,
    weights: jnp.ndarray,
    responsibilities: jnp.ndarray,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_mass: float = constants.GC_EPS_MASS,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[ScanBinStats, CertBundle, ExpectedEffect]:
    """
    Compute scan bin statistics using soft responsibilities.
    
    Uses InvMass for all mass-based computations (no N==0 branches).
    
    Args:
        points: Point positions (N, 3)
        point_covariances: Per-point covariances (N, 3, 3)
        weights: Per-point weights (N,)
        responsibilities: Soft assignments (N, B_BINS)
        eps_psd: PSD projection epsilon
        eps_mass: Mass regularization
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (ScanBinStats, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.5
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    point_covariances = jnp.asarray(point_covariances, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    responsibilities = jnp.asarray(responsibilities, dtype=jnp.float64)
    
    n_points = points.shape[0]
    n_bins = responsibilities.shape[1]
    
    # Initialize accumulator arrays
    N = jnp.zeros(n_bins, dtype=jnp.float64)
    s_dir = jnp.zeros((n_bins, 3), dtype=jnp.float64)
    sum_p = jnp.zeros((n_bins, 3), dtype=jnp.float64)
    sum_ppT = jnp.zeros((n_bins, 3, 3), dtype=jnp.float64)
    sum_cov = jnp.zeros((n_bins, 3, 3), dtype=jnp.float64)
    
    # Accumulate sufficient statistics
    for i in range(n_points):
        w_i = weights[i]
        p_i = points[i]
        cov_i = point_covariances[i]
        
        # Normalize point direction
        d_i, _ = safe_normalize(p_i, eps_mass)
        
        for b in range(n_bins):
            r_ib = responsibilities[i, b]
            w_ib = w_i * r_ib  # Weighted responsibility
            
            # Mass
            N = N.at[b].add(w_ib)
            
            # Directional resultant
            s_dir = s_dir.at[b].add(w_ib * d_i)
            
            # Position sum
            sum_p = sum_p.at[b].add(w_ib * p_i)
            
            # Position scatter
            sum_ppT = sum_ppT.at[b].add(w_ib * jnp.outer(p_i, p_i))
            
            # Covariance sum (for uncertainty propagation)
            sum_cov = sum_cov.at[b].add(w_ib * cov_i)
    
    # Compute derived quantities using InvMass (no N==0 branches)
    p_bar = jnp.zeros((n_bins, 3), dtype=jnp.float64)
    Sigma_p = jnp.zeros((n_bins, 3, 3), dtype=jnp.float64)
    kappa_scan = jnp.zeros(n_bins, dtype=jnp.float64)
    
    # Import kappa operator
    from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_v2
    
    for b in range(n_bins):
        N_b = N[b]
        
        # InvMass - always applied, no conditional
        inv_N_b = inv_mass(float(N_b), eps_mass).inv_mass
        
        # Centroid
        c_b = sum_p[b] * inv_N_b
        p_bar = p_bar.at[b].set(c_b)
        
        # Scatter covariance
        scatter = sum_ppT[b] * inv_N_b - jnp.outer(c_b, c_b)
        
        # Add measurement covariance (weighted average)
        meas_cov = sum_cov[b] * inv_N_b
        
        # Total covariance
        Sigma_raw = scatter + meas_cov
        
        # Project to PSD (always)
        Sigma_psd = domain_projection_psd(Sigma_raw, eps_psd).M_psd
        Sigma_p = Sigma_p.at[b].set(Sigma_psd)
        
        # Kappa from resultant length
        S_b_norm = float(jnp.linalg.norm(s_dir[b]))
        Rbar_b = S_b_norm * inv_N_b  # R-bar in (0, 1)
        kappa_result, _, _ = kappa_from_resultant_v2(Rbar_b)
        kappa_scan = kappa_scan.at[b].set(kappa_result.kappa)
    
    # Build result
    result = ScanBinStats(
        N=N,
        s_dir=s_dir,
        p_bar=p_bar,
        Sigma_p=Sigma_p,
        kappa_scan=kappa_scan,
    )
    
    # Compute ESS
    total_mass = float(jnp.sum(N))
    ess = total_mass ** 2 / (jnp.sum(N ** 2) + eps_mass)
    
    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ScanBinMomentMatch"],
        support=SupportCert(
            ess_total=float(ess),
            support_frac=float(jnp.sum(N > eps_mass) / n_bins),
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,  # Would need to track
            mass_epsilon_ratio=eps_mass / (total_mass + eps_mass),
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_ess",
        predicted=float(ess),
        realized=None,
    )
    
    return result, cert, expected_effect
