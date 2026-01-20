"""
Models module - Legacy compatibility.

NOTE: Models have moved to fl_slam_poc.backend.parameters.
This module provides backward compatibility re-exports.
"""

from fl_slam_poc.backend.parameters import (
    AdaptiveParameter,
    OnlineStats,
    TimeAlignmentModel,
    StochasticBirthModel,
    AdaptiveProcessNoise,
    NIGModel,
    NIG_PRIOR_KAPPA,
    NIG_PRIOR_ALPHA,
    NIG_PRIOR_BETA,
    combine_independent_weights,
)

__all__ = [
    "AdaptiveParameter",
    "OnlineStats",
    "TimeAlignmentModel",
    "StochasticBirthModel",
    "AdaptiveProcessNoise",
    "NIGModel",
    "NIG_PRIOR_KAPPA",
    "NIG_PRIOR_ALPHA",
    "NIG_PRIOR_BETA",
    "combine_independent_weights",
]
