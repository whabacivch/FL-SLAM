"""
Probabilistic models for FL-SLAM.

All models follow information geometry principles:
- Explicit generative model declarations
- Closed-form updates where possible
- Frobenius correction for approximations
"""

from fl_slam_poc.backend.parameters.adaptive import AdaptiveParameter, OnlineStats
from fl_slam_poc.backend.parameters.timestamp import TimeAlignmentModel
from fl_slam_poc.backend.parameters.birth import StochasticBirthModel
from fl_slam_poc.backend.parameters.process_noise import AdaptiveProcessNoise
from fl_slam_poc.backend.parameters.nig import NIGModel, NIG_PRIOR_KAPPA, NIG_PRIOR_ALPHA, NIG_PRIOR_BETA
from fl_slam_poc.backend.parameters.weights import combine_independent_weights

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

