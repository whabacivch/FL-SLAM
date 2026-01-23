"""
Dirichlet-Categorical Routing Module.

Implements uncertainty-aware routing for IMU factor fusion with:
- Dirichlet conjugate prior over categorical routing probabilities
- Frobenius retention (cubic contraction) for belief persistence
- Hellinger shift monitoring for stability diagnostics

This is NOT "normal softmax" as the end belief. Softmax is used as the
categorical mean map to generate pseudo-count evidence, which is then
used to update the Dirichlet posterior.

Key principle (from CL framework):
    "Softmax IS the categorical exponential family mean map"
    The correct uncertainty-bearing object is Dirichlet-categorical conjugacy.

Reference: Compositional Legendre framework, Hellinger hierarchical construction
"""

from typing import Dict

from fl_slam_poc.common import constants
from fl_slam_poc.common.jax_init import jax, jnp
from jax import jit
from jax.scipy.linalg import cholesky, solve_triangular


# =============================================================================
# Numerical Constants
# =============================================================================

# Minimum probability/weight for numerical stability
# Prevents log(0) and division by zero
MIN_PROB = constants.MIN_MIXTURE_WEIGHT

# Default Frobenius retention base (cubic contraction applied)
# t = 0.95^3 ≈ 0.857, meaning ~14% belief decay per update
DEFAULT_RETENTION_BASE = 0.95

# Default Dirichlet prior concentration (symmetric)
# α = 1.0 corresponds to uniform prior (non-informative)
DEFAULT_ALPHA_PRIOR = 1.0

# Default evidence budget per update
# B = 1.0 means each observation contributes 1 total pseudo-count
DEFAULT_EVIDENCE_BUDGET = 1.0

# Prior strength for E[log θ] term in combined logits
DEFAULT_LAMBDA_PRIOR = 1.0


class DirichletRoutingModule:
    """
    Dirichlet-categorical routing with Frobenius retention.

    Maintains a Dirichlet posterior over anchor routing probabilities.
    Updates are Bayesian with a Frobenius cubic retention factor.

    The Dirichlet-categorical conjugacy means:
    - Prior: Dir(α)
    - Likelihood: Cat(θ) with θ ~ Dir(α)
    - Posterior: Dir(α + counts)

    Frobenius retention applies a cubic contraction α' = t³ * α before
    each update, implementing a principled "forgetting" that prevents
    belief collapse to a single anchor.
    """

    def __init__(
        self,
        n_anchors: int,
        alpha_prior: float = DEFAULT_ALPHA_PRIOR,
        retention_base: float = DEFAULT_RETENTION_BASE,
        evidence_budget: float = DEFAULT_EVIDENCE_BUDGET,
        lambda_prior: float = DEFAULT_LAMBDA_PRIOR,
    ):
        """
        Initialize Dirichlet routing module.

        Args:
            n_anchors: Number of anchors to route between
            alpha_prior: Initial Dirichlet concentration (symmetric)
            retention_base: Base for Frobenius cubic contraction (0 < t < 1)
            evidence_budget: Total pseudo-count per update (B)
            lambda_prior: Weight for E[log θ] term in combined logits

        Raises:
            ValueError: If parameters are out of valid range
        """
        if n_anchors <= 0:
            raise ValueError(f"n_anchors must be positive, got {n_anchors}")
        if alpha_prior <= 0:
            raise ValueError(f"alpha_prior must be positive, got {alpha_prior}")
        if not 0 < retention_base < 1:
            raise ValueError(f"retention_base must be in (0, 1), got {retention_base}")
        if evidence_budget <= 0:
            raise ValueError(f"evidence_budget must be positive, got {evidence_budget}")

        self.n_anchors = n_anchors
        self.alpha_prior = alpha_prior
        self.retention_base = retention_base
        self.evidence_budget = evidence_budget
        self.lambda_prior = lambda_prior

        # Initialize Dirichlet parameters (symmetric prior)
        self.alpha = jnp.full(n_anchors, alpha_prior, dtype=jnp.float64)

        # State for Hellinger shift monitoring
        self._prev_resp: jnp.ndarray | None = None
        self._last_hellinger_shift: float = 0.0
        self._update_count: int = 0

    def update(self, logits: jnp.ndarray) -> jnp.ndarray:
        """
        Update routing belief with new evidence.

        Algorithm:
        1. Frobenius retention: α' = t³ * α (cubic contraction)
        2. Combined logits: s_i = ω_i + λ * E[log θ_i]
           where E[log θ_i] = ψ(α_i) - ψ(Σα) ≈ log(α_i) - log(Σα)
        3. Softmax → pseudo-counts: c = B * softmax(s)
        4. Dirichlet update: α = α' + c
        5. Hellinger shift: H² = 1 - Σ√(π_t · π_{t-1})
        6. Return responsibilities: w = E[θ] = α / Σα

        Args:
            logits: Per-anchor log-weights from likelihood (M,)

        Returns:
            responsibilities: Dirichlet mean (normalized) (M,)

        Raises:
            ValueError: If logits dimension doesn't match n_anchors
        """
        logits = jnp.asarray(logits, dtype=jnp.float64).reshape(-1)

        if len(logits) != self.n_anchors:
            raise ValueError(
                f"logits dimension {len(logits)} != n_anchors {self.n_anchors}"
            )

        # Step 1: Frobenius retention (cubic contraction)
        retention = self.retention_base ** 3
        alpha_retained = retention * self.alpha

        # Step 2: Combined logits with Dirichlet prior term
        alpha_sum = jnp.sum(alpha_retained)
        expected_log_theta = jnp.log(alpha_retained + MIN_PROB) - jnp.log(alpha_sum + MIN_PROB)

        combined_logits = logits + self.lambda_prior * expected_log_theta

        # Step 3: Numerically stable softmax → pseudo-counts
        logits_shifted = combined_logits - jnp.max(combined_logits)
        exp_logits = jnp.exp(logits_shifted)
        softmax_probs = exp_logits / jnp.sum(exp_logits)

        pseudo_counts = self.evidence_budget * softmax_probs

        # Step 4: Dirichlet update
        self.alpha = alpha_retained + pseudo_counts

        # Step 5: Responsibilities (Dirichlet mean)
        responsibilities = self.alpha / jnp.sum(self.alpha)

        # Step 6: Hellinger shift diagnostic
        if self._prev_resp is not None:
            self._last_hellinger_shift = self._hellinger_squared(
                responsibilities, self._prev_resp
            )

        self._prev_resp = responsibilities.copy()
        self._update_count += 1

        return responsibilities

    def _hellinger_squared(self, p: jnp.ndarray, q: jnp.ndarray) -> float:
        """
        Squared Hellinger distance between discrete distributions.

        H²(p, q) = 1 - Σᵢ √(pᵢ · qᵢ)

        Properties:
        - H² ∈ [0, 1]
        - H² = 0 iff p = q
        - H² = 1 iff p and q have disjoint support
        """
        bc = jnp.sum(jnp.sqrt(jnp.maximum(p * q, 0.0)))
        return float(jnp.maximum(0.0, 1.0 - bc))

    def get_responsibilities(self) -> jnp.ndarray:
        """Get current responsibilities (Dirichlet mean)."""
        return self.alpha / jnp.sum(self.alpha)

    def get_retention_scalar(self) -> float:
        """Get Frobenius retention factor (t³)."""
        return self.retention_base ** 3

    def get_alpha(self) -> jnp.ndarray:
        """Get current Dirichlet alpha parameters."""
        return self.alpha.copy()

    def get_update_diagnostics(self) -> Dict[str, float | jnp.ndarray]:
        """Get routing diagnostics for logging."""
        return {
            "alpha": self.get_alpha(),
            "responsibilities": self.get_responsibilities(),
            "retention": self.get_retention_scalar(),
            "hellinger_shift": self.get_hellinger_shift(),
        }

    def get_hellinger_shift(self) -> float:
        """Get last computed Hellinger shift."""
        return self._last_hellinger_shift

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information for logging/monitoring."""
        responsibilities = self.get_responsibilities()
        return {
            "alpha": self.alpha.tolist(),
            "alpha_sum": float(jnp.sum(self.alpha)),
            "responsibilities": responsibilities.tolist(),
            "max_responsibility": float(jnp.max(responsibilities)),
            "retention_factor": self.get_retention_scalar(),
            "hellinger_shift": self._last_hellinger_shift,
            "update_count": self._update_count,
            "entropy": float(-jnp.sum(responsibilities * jnp.log(responsibilities + MIN_PROB))),
        }

    def resize(self, new_n_anchors: int) -> None:
        """
        Resize module for different number of anchors.

        New anchors receive the prior concentration.
        Removed anchors' mass is lost (not redistributed).

        Args:
            new_n_anchors: New number of anchors

        Raises:
            ValueError: If new_n_anchors <= 0
        """
        if new_n_anchors <= 0:
            raise ValueError(f"new_n_anchors must be positive, got {new_n_anchors}")

        if new_n_anchors > self.n_anchors:
            extra = jnp.full(new_n_anchors - self.n_anchors, self.alpha_prior, dtype=jnp.float64)
            self.alpha = jnp.concatenate([self.alpha, extra])
        elif new_n_anchors < self.n_anchors:
            self.alpha = self.alpha[:new_n_anchors]

        self.n_anchors = new_n_anchors
        self._prev_resp = None
