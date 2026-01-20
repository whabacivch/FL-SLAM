"""
Gaussian operations in information (natural parameter) form.

Following information geometry principles:
- Represent beliefs in canonical coordinates: θ = (Λ, η) where Λ = Σ⁻¹, η = Σ⁻¹μ
- Fusion is EXACT addition in natural parameters (no Jacobians)
- Order-invariant: (L₁ + L₂) + L₃ = L₁ + (L₂ + L₃)
- Commutative: L₁ + L₂ = L₂ + L₁

Why information form replaces EKF-style updates:
- EKF: Jacobian-based linearization + Kalman gain computation
- Here: θ_posterior = θ_prior + θ_likelihood (additive, exact)
- O(n) vs O(n² iter) for iterative methods

Reference: Barndorff-Nielsen (1978), Amari (2016), Combe (2022-2025)
"""

import math
import numpy as np
from typing import Tuple


def _as_vector(x: np.ndarray) -> np.ndarray:
    """
    Normalize any (n,), (n,1), (1,n) into a flat (n,) float vector.

    This prevents silent NumPy broadcasting bugs when mixing column vectors
    and 1D arrays in information-form operations.
    """
    x = np.asarray(x, dtype=float)
    return x.reshape(-1)


def make_evidence(mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (mean, covariance) to information form (Lambda, eta).
    
    Natural parameters for Gaussian:
        Λ = Σ⁻¹ (precision matrix)
        η = Σ⁻¹μ (information vector)
    
    The log-density is: log p(x) ∝ -½x'Λx + η'x - ψ(Λ,η)
    """
    cov = np.asarray(cov, dtype=float)
    mean = _as_vector(mean)
    L = np.linalg.inv(cov)
    h = (L @ mean.reshape(-1, 1)).reshape(-1)
    return L, h


def fuse_info(
    L: np.ndarray,
    h: np.ndarray,
    L_obs: np.ndarray,
    h_obs: np.ndarray,
    weight: float = 1.0,
    rho: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Order-invariant additive fusion in information space.
    
    For exponential families, Bayesian fusion in natural coordinates is:
        θ_posterior = ρ·θ_prior + w·θ_likelihood
    
    This is EXACT (no approximation) when both prior and likelihood
    are in the same exponential family.
    
    Properties:
    - Commutative: fuse(A, B) = fuse(B, A)
    - Associative: fuse(fuse(A, B), C) = fuse(A, fuse(B, C))
    - No Jacobians required
    """
    h = _as_vector(h)
    h_obs = _as_vector(h_obs)
    return (rho * L + weight * L_obs, rho * h + weight * h_obs)


def mean_cov(L: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert information form (Lambda, eta) to (mean, covariance).
    
    Recovery from natural parameters:
        Σ = Λ⁻¹
        μ = Ση = Λ⁻¹η
    """
    cov = np.linalg.inv(L)
    h = _as_vector(h)
    mean = (cov @ h.reshape(-1, 1)).reshape(-1)
    return mean, cov


def log_partition(L: np.ndarray, h: np.ndarray) -> float:
    """
    Log-partition function for Gaussian in information form.
    
    ψ(Λ, η) = ½η'Λ⁻¹η + ½(d log(2π) - log|Λ|)
    
    This is the convex potential that generates all cumulants.
    """
    d = L.shape[0]
    cov = np.linalg.inv(L)
    _, logdet_L = np.linalg.slogdet(L)
    h = _as_vector(h)
    quad = 0.5 * float(h @ (cov @ h))
    return quad + 0.5 * (d * math.log(2 * math.pi) - logdet_L)


def kl_divergence(L1: np.ndarray, h1: np.ndarray,
                  L2: np.ndarray, h2: np.ndarray) -> float:
    """
    KL divergence D_KL(N₁ || N₂) in information form.
    
    D_KL = ψ(θ₂) - ψ(θ₁) - ⟨∇ψ(θ₁), θ₂ - θ₁⟩
    
    This is the Bregman divergence induced by the log-partition.
    Closed-form, O(n³) for matrix operations.
    """
    mu1, cov1 = mean_cov(L1, h1)
    mu2, cov2 = mean_cov(L2, h2)
    
    d = cov1.shape[0]
    cov2_inv = L2  # L2 = Σ₂⁻¹
    
    term1 = np.trace(cov2_inv @ cov1)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    
    diff = mu2 - mu1
    kl = 0.5 * (term1 - d - logdet1 + logdet2 + float(diff @ cov2_inv @ diff))
    return float(kl)


def hellinger_distance(L1: np.ndarray, h1: np.ndarray,
                       L2: np.ndarray, h2: np.ndarray) -> float:
    """
    Hellinger distance between two Gaussians.
    
    H²(p₁, p₂) = 1 - BC where BC = ∫√(p₁p₂)dx
    
    For Gaussians, BC has closed form via the log-partition.
    """
    mu1, cov1 = mean_cov(L1, h1)
    mu2, cov2 = mean_cov(L2, h2)
    
    cov_avg = 0.5 * (cov1 + cov2)
    _, logdet_avg = np.linalg.slogdet(cov_avg)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    
    log_bc = 0.25 * logdet1 + 0.25 * logdet2 - 0.5 * logdet_avg
    
    diff = mu1 - mu2
    cov_avg_inv = np.linalg.inv(cov_avg)
    log_bc -= 0.125 * float(diff @ cov_avg_inv @ diff)
    
    bc = np.exp(log_bc)
    return float(math.sqrt(max(0.0, 1.0 - bc)))


def bhattacharyya_coefficient(L1: np.ndarray, h1: np.ndarray,
                               L2: np.ndarray, h2: np.ndarray) -> float:
    """
    Bhattacharyya coefficient BC = ∫√(p₁p₂)dx.
    
    BC ∈ [0, 1], with BC = 1 for identical distributions.
    Used for robust gating: BC < threshold indicates outlier.
    """
    mu1, cov1 = mean_cov(L1, h1)
    mu2, cov2 = mean_cov(L2, h2)
    
    cov_avg = 0.5 * (cov1 + cov2)
    _, logdet_avg = np.linalg.slogdet(cov_avg)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    
    db = 0.5 * logdet_avg - 0.25 * (logdet1 + logdet2)
    
    diff = mu1 - mu2
    cov_avg_inv = np.linalg.inv(cov_avg)
    db += 0.125 * float(diff @ cov_avg_inv @ diff)
    
    return float(np.exp(-db))


def fisher_information(L: np.ndarray) -> np.ndarray:
    """
    Fisher information matrix for Gaussian.
    
    For the mean parameter: I(μ) = Σ⁻¹ = Λ
    
    The Fisher metric is the Hessian of the log-partition:
        g = ∇²ψ = Cov(T) = Σ
    """
    return L.copy()


def natural_gradient(loss_grad: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Natural gradient: g⁻¹∇L = Σ∇L.
    
    The natural gradient accounts for the information geometry,
    giving steepest descent in the Fisher-Rao metric.
    
    O(n³) one-shot solve vs O(n² iter) for iterative methods.
    """
    cov = np.linalg.inv(L)
    return cov @ loss_grad


def marginalize(L: np.ndarray, h: np.ndarray, 
                keep_dims: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marginalize out dimensions not in keep_dims.
    
    In information form, marginalization requires converting to moment
    form, selecting dimensions, then converting back.
    """
    mu, cov = mean_cov(L, h)
    keep_dims = np.asarray(keep_dims, dtype=int)
    
    mu_marg = mu[keep_dims]
    cov_marg = cov[np.ix_(keep_dims, keep_dims)]
    
    return make_evidence(mu_marg, cov_marg)


def condition(L: np.ndarray, h: np.ndarray,
              obs_dims: np.ndarray, obs_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Condition on observed dimensions.
    
    Returns the posterior (L, h) for the unobserved dimensions
    given observations at obs_dims = obs_vals.
    """
    mu, cov = mean_cov(L, h)
    obs_dims = np.asarray(obs_dims, dtype=int)
    obs_vals = np.asarray(obs_vals, dtype=float)
    
    n = L.shape[0]
    all_dims = np.arange(n)
    unobs_dims = np.array([i for i in all_dims if i not in obs_dims])
    
    if len(unobs_dims) == 0:
        return np.array([[1.0]]), np.array([[0.0]])
    
    # Partition covariance
    Sigma_aa = cov[np.ix_(unobs_dims, unobs_dims)]
    Sigma_ab = cov[np.ix_(unobs_dims, obs_dims)]
    Sigma_bb = cov[np.ix_(obs_dims, obs_dims)]
    
    mu_a = mu[unobs_dims]
    mu_b = mu[obs_dims]
    
    # Conditional: μ_a|b = μ_a + Σ_ab Σ_bb⁻¹ (b - μ_b)
    Sigma_bb_inv = np.linalg.inv(Sigma_bb)
    mu_cond = mu_a + Sigma_ab @ Sigma_bb_inv @ (obs_vals - mu_b)
    cov_cond = Sigma_aa - Sigma_ab @ Sigma_bb_inv @ Sigma_ab.T
    
    return make_evidence(mu_cond, cov_cond)


def product_of_experts(experts: list[Tuple[np.ndarray, np.ndarray]],
                       weights: list[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse multiple Gaussian experts (product-of-experts).
    
    In information form, PoE is simply additive:
        θ_fused = Σᵢ wᵢ θᵢ
    
    This is EXACT and order-invariant.
    """
    if len(experts) == 0:
        raise ValueError("Need at least one expert")
    
    if weights is None:
        weights = [1.0] * len(experts)
    
    L_sum = np.zeros_like(experts[0][0])
    h_sum = np.zeros_like(_as_vector(experts[0][1]))
    
    for (L, h), w in zip(experts, weights):
        h = _as_vector(h)
        L_sum += w * L
        h_sum += w * h
    
    return L_sum, h_sum
