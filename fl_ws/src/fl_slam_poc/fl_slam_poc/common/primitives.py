"""
Branch-free numeric primitives for Golden Child SLAM v2.

All functions in this module are TOTAL FUNCTIONS that always run.
They return (result, magnitude) where magnitude can be exactly 0.

Design invariants:
- No if/else branches that gate computation
- No early returns based on data values
- All numerical stabilization is ALWAYS applied
- Magnitude fields record the effect (can be 0 if no change)
- Uses JAX for all math operations

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp

# =============================================================================
# Result Types (pure Python dataclasses for results)
# =============================================================================


@dataclass
class SymmetrizeResult:
    """Result of Symmetrize operation."""
    M_sym: jnp.ndarray  # Symmetric matrix
    sym_delta: float  # ||M_sym - M||_F


@dataclass
class ConditioningInfo:
    """Eigenvalue conditioning information."""
    eig_min: float
    eig_max: float
    cond: float
    near_null_count: int  # Count of eigenvalues < 10 * eps_psd


@dataclass
class DomainProjectionPSDResult:
    """Result of DomainProjectionPSD operation."""
    M_psd: jnp.ndarray  # PSD matrix
    projection_delta: float  # ||M_psd - M_sym||_F
    sym_delta: float  # ||M_sym - M||_F from symmetrize step
    conditioning: ConditioningInfo


@dataclass
class SPDSolveResult:
    """Result of SPDCholeskySolveLifted operation."""
    x: jnp.ndarray  # Solution vector
    lift_strength: float  # eps_lift * dimension


@dataclass
class InvMassResult:
    """Result of InvMass operation."""
    inv_mass: float  # 1 / (m + eps_mass)
    mass_epsilon_ratio: float  # eps_mass / (m + eps_mass)


@dataclass
class ClampResult:
    """Result of Clamp operation."""
    value: float  # Clamped value
    clamp_delta: float  # |Clamp(x) - x|


# =============================================================================
# JAX Primitive Functions (Branch-Free, Always Execute)
# NOTE: No float() conversions inside JIT - do conversions in wrappers
# =============================================================================


def _symmetrize_jax(M: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX implementation of symmetrize (not JIT - called from wrapper)."""
    M_sym = 0.5 * (M + M.T)
    sym_delta = jnp.linalg.norm(M_sym - M, ord='fro')
    return M_sym, sym_delta


def symmetrize(M: jnp.ndarray) -> SymmetrizeResult:
    """
    Symmetrize a matrix (always computed).
    
    Args:
        M: Input matrix (d, d)
        
    Returns:
        SymmetrizeResult with symmetric matrix and delta magnitude
        
    Spec ref: Section 3.1
    """
    M = jnp.asarray(M, dtype=jnp.float64)
    M_sym, sym_delta = _symmetrize_jax(M)
    return SymmetrizeResult(M_sym=M_sym, sym_delta=float(sym_delta))


def _domain_projection_psd_jax(
    M: jnp.ndarray,
    eps_psd: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX implementation of PSD projection (not JIT - called from wrapper)."""
    # Always symmetrize first
    M_sym = 0.5 * (M + M.T)
    sym_delta = jnp.linalg.norm(M_sym - M, ord='fro')
    
    # Always compute eigendecomposition
    eigvals, eigvecs = jnp.linalg.eigh(M_sym)
    
    # Always clamp eigenvalues (no conditional)
    vals_clamped = jnp.maximum(eigvals, eps_psd)
    
    # Always reconstruct
    M_psd = eigvecs @ jnp.diag(vals_clamped) @ eigvecs.T
    
    # Always compute projection delta
    projection_delta = jnp.linalg.norm(M_psd - M_sym, ord='fro')
    
    return M_psd, vals_clamped, sym_delta, projection_delta


def domain_projection_psd(
    M: jnp.ndarray,
    eps_psd: float = 1e-12,
) -> DomainProjectionPSDResult:
    """
    Project matrix onto PSD cone (always computed).
    
    Always executes:
    1. Symmetrize
    2. Eigendecomposition
    3. Clamp eigenvalues to >= eps_psd
    4. Reconstruct
    
    No conditional "only if needed" - clamp always executed.
    
    Args:
        M: Input matrix (d, d)
        eps_psd: Minimum eigenvalue (default 1e-12)
        
    Returns:
        DomainProjectionPSDResult with PSD matrix and all metrics
        
    Spec ref: Section 3.2
    """
    M = jnp.asarray(M, dtype=jnp.float64)
    M_psd, vals_clamped, sym_delta, projection_delta = _domain_projection_psd_jax(M, eps_psd)
    
    # Compute conditioning info (outside JIT, can use float())
    near_null_threshold = 10.0 * eps_psd
    near_null_count = int(jnp.sum(vals_clamped < near_null_threshold))
    
    eig_min = float(jnp.min(vals_clamped))
    eig_max = float(jnp.max(vals_clamped))
    cond = eig_max / eig_min  # Always valid due to clamping
    
    conditioning = ConditioningInfo(
        eig_min=eig_min,
        eig_max=eig_max,
        cond=cond,
        near_null_count=near_null_count,
    )
    
    return DomainProjectionPSDResult(
        M_psd=M_psd,
        projection_delta=float(projection_delta),
        sym_delta=float(sym_delta),
        conditioning=conditioning,
    )


def _spd_cholesky_solve_lifted_jax(
    L: jnp.ndarray,
    b: jnp.ndarray,
    eps_lift: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX implementation of lifted Cholesky solve."""
    d = L.shape[0]
    
    # Always apply lift (never conditional)
    L_lifted = L + eps_lift * jnp.eye(d)
    lift_strength = eps_lift * d
    
    # Cholesky factorization (guaranteed to succeed due to lift)
    L_chol = jnp.linalg.cholesky(L_lifted)
    
    # Forward-backward solve using triangular solves
    y = jax.scipy.linalg.solve_triangular(L_chol, b, lower=True)
    x = jax.scipy.linalg.solve_triangular(L_chol.T, y, lower=False)
    
    return x, jnp.array(lift_strength)


def spd_cholesky_solve_lifted(
    L: jnp.ndarray,
    b: jnp.ndarray,
    eps_lift: float = 1e-9,
) -> SPDSolveResult:
    """
    Solve (L + eps_lift * I) x = b using Cholesky (always lifted).
    
    The lift is ALWAYS applied, never conditional. This guarantees
    the matrix is SPD and Cholesky succeeds.
    
    Args:
        L: Information matrix (d, d) - should be PSD
        b: Right-hand side (d,) or (d, k)
        eps_lift: Lift amount (default 1e-9)
        
    Returns:
        SPDSolveResult with solution and lift strength
        
    Spec ref: Section 3.3
    """
    L = jnp.asarray(L, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    
    x, lift_strength = _spd_cholesky_solve_lifted_jax(L, b, eps_lift)
    
    return SPDSolveResult(x=x, lift_strength=float(lift_strength))


def _spd_cholesky_inverse_lifted_jax(
    L: jnp.ndarray,
    eps_lift: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX implementation of lifted Cholesky inverse."""
    d = L.shape[0]
    
    # Always apply lift
    L_lifted = L + eps_lift * jnp.eye(d)
    lift_strength = eps_lift * d
    
    # Cholesky and invert
    L_chol = jnp.linalg.cholesky(L_lifted)
    L_chol_inv = jax.scipy.linalg.solve_triangular(L_chol, jnp.eye(d), lower=True)
    L_inv = L_chol_inv.T @ L_chol_inv
    
    return L_inv, jnp.array(lift_strength)


def spd_cholesky_inverse_lifted(
    L: jnp.ndarray,
    eps_lift: float = 1e-9,
) -> Tuple[jnp.ndarray, float]:
    """
    Compute inverse of (L + eps_lift * I) using Cholesky (always lifted).
    
    Args:
        L: Information matrix (d, d) - should be PSD
        eps_lift: Lift amount (default 1e-9)
        
    Returns:
        Tuple of (inverse matrix, lift_strength)
    """
    L = jnp.asarray(L, dtype=jnp.float64)
    L_inv, lift_strength = _spd_cholesky_inverse_lifted_jax(L, eps_lift)
    return L_inv, float(lift_strength)


def inv_mass(m: float, eps_mass: float = 1e-12) -> InvMassResult:
    """
    Compute inverse mass with epsilon regularization (always applied).
    
    inv_mass = 1 / max(m, eps_mass)
    
    This removes all division-by-zero gating.
    
    Args:
        m: Mass value (can be 0 or negative)
        eps_mass: Regularization epsilon (default 1e-12)
        
    Returns:
        InvMassResult with inverse mass and epsilon ratio
        
    Spec ref: Section 3.4
    """
    m = float(m)
    eps_mass = float(eps_mass)
    
    # Always compute (no conditional) - use max to ensure positive denominator
    denom = max(m, eps_mass)
    inv_m = 1.0 / denom
    mass_epsilon_ratio = eps_mass / denom if m >= eps_mass else 1.0
    
    return InvMassResult(inv_mass=inv_m, mass_epsilon_ratio=mass_epsilon_ratio)


def clamp(x: float, lo: float, hi: float) -> ClampResult:
    """
    Clamp value to [lo, hi] range (always computed).
    
    Args:
        x: Input value
        lo: Lower bound
        hi: Upper bound
        
    Returns:
        ClampResult with clamped value and delta
        
    Spec ref: Section 3.5
    """
    x = float(x)
    lo = float(lo)
    hi = float(hi)
    
    # Always compute using Python ops (branchless via min/max)
    clamped = max(lo, min(x, hi))
    clamp_delta = abs(clamped - x)
    
    return ClampResult(value=clamped, clamp_delta=clamp_delta)


def _clamp_array_jax(x: jnp.ndarray, lo: float, hi: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX implementation of array clamp."""
    clamped = jnp.clip(x, lo, hi)
    clamp_delta = jnp.linalg.norm(clamped - x)
    return clamped, clamp_delta


def clamp_array(x: jnp.ndarray, lo: float, hi: float) -> Tuple[jnp.ndarray, float]:
    """
    Clamp array values to [lo, hi] range (always computed).
    
    Args:
        x: Input array
        lo: Lower bound
        hi: Upper bound
        
    Returns:
        Tuple of (clamped array, total clamp delta norm)
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    clamped, clamp_delta = _clamp_array_jax(x, lo, hi)
    return clamped, float(clamp_delta)


# =============================================================================
# Derived Primitives (JAX)
# =============================================================================


def _safe_normalize_jax(v: jnp.ndarray, eps: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX implementation of safe normalize."""
    norm = jnp.linalg.norm(v)
    denom = norm + eps
    v_normalized = v / denom
    eps_ratio = eps / denom
    return v_normalized, eps_ratio


def safe_normalize(v: jnp.ndarray, eps: float = 1e-12) -> Tuple[jnp.ndarray, float]:
    """
    Normalize vector with epsilon regularization (always computed).
    
    Args:
        v: Input vector
        eps: Regularization epsilon
        
    Returns:
        Tuple of (normalized vector, epsilon ratio in denominator)
    """
    v = jnp.asarray(v, dtype=jnp.float64)
    v_normalized, eps_ratio = _safe_normalize_jax(v, eps)
    return v_normalized, float(eps_ratio)


def sigmoid(x: float) -> float:
    """
    Sigmoid function: 1 / (1 + exp(-x)).
    
    Numerically stable JAX implementation.
    """
    return float(jax.nn.sigmoid(jnp.array(x)))


def _softmax_jax(logits: jnp.ndarray, tau: float) -> jnp.ndarray:
    """JAX implementation of softmax with temperature."""
    scaled = logits / tau
    return jax.nn.softmax(scaled)


def softmax(logits: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:
    """
    Softmax function with temperature (always computed).
    
    Args:
        logits: Input logits
        tau: Temperature parameter
        
    Returns:
        Probability vector
    """
    logits = jnp.asarray(logits, dtype=jnp.float64)
    return _softmax_jax(logits, tau)


def log_sum_exp(x: jnp.ndarray) -> float:
    """
    Log-sum-exp with numerical stability (JAX).
    
    Args:
        x: Input array
        
    Returns:
        log(sum(exp(x)))
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    return float(jax.scipy.special.logsumexp(x))
