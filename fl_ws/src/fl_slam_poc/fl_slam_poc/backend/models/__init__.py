"""
Probabilistic models for backend estimation.

All models are declared generative models with explicit priors.
No hardcoded constants - all values are either priors or learned online.
"""

from fl_slam_poc.backend.models.adaptive import AdaptiveParameter, OnlineStats
from fl_slam_poc.backend.models.birth import StochasticBirthModel
from fl_slam_poc.backend.models.nig import NIGModel, NIG_PRIOR_KAPPA, NIG_PRIOR_ALPHA, NIG_PRIOR_BETA
from fl_slam_poc.backend.models.process_noise import (
    AdaptiveProcessNoise,
    WishartPrior,
    AdaptiveIMUNoiseModel,
)
from fl_slam_poc.backend.models.timestamp import TimeAlignmentModel

__all__ = [
    "AdaptiveParameter",
    "OnlineStats",
    "StochasticBirthModel",
    "NIGModel",
    "NIG_PRIOR_KAPPA",
    "NIG_PRIOR_ALPHA",
    "NIG_PRIOR_BETA",
    "AdaptiveProcessNoise",
    "WishartPrior",
    "AdaptiveIMUNoiseModel",
    "TimeAlignmentModel",
]
