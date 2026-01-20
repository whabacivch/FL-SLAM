"""
Probabilistic models for FL-SLAM.

All models follow information geometry principles:
- Explicit generative model declarations
- Closed-form updates where possible
- Frobenius correction for approximations
"""

from fl_slam_poc.models.adaptive import AdaptiveParameter, OnlineStats
from fl_slam_poc.models.timestamp import TimeAlignmentModel
from fl_slam_poc.models.birth import StochasticBirthModel
from fl_slam_poc.models.process_noise import AdaptiveProcessNoise
from fl_slam_poc.models.nig import NIGModel, NIG_PRIOR_KAPPA, NIG_PRIOR_ALPHA, NIG_PRIOR_BETA
from fl_slam_poc.models.weights import combine_independent_weights

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

