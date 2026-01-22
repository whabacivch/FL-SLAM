"""
Backend package for FL-SLAM.

Information-geometric fusion and state estimation.

Subpackages (flattened):
- gaussian_info/gaussian_geom/information_distances
- adaptive/birth/nig/process_noise/timestamp/weights
- dirichlet_routing
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from fl_slam_poc.backend.adaptive import AdaptiveParameter, OnlineStats
from fl_slam_poc.backend.timestamp import TimeAlignmentModel
from fl_slam_poc.backend.birth import StochasticBirthModel
from fl_slam_poc.backend.process_noise import AdaptiveProcessNoise, AdaptiveIMUNoiseModel, WishartPrior
from fl_slam_poc.backend.nig import (
    NIGModel,
    NIG_PRIOR_KAPPA,
    NIG_PRIOR_ALPHA,
    NIG_PRIOR_BETA,
)
from fl_slam_poc.backend.weights import combine_independent_weights

__all__ = [
    # Parameters
    "AdaptiveParameter",
    "OnlineStats",
    "TimeAlignmentModel",
    "StochasticBirthModel",
    "AdaptiveProcessNoise",
    "AdaptiveIMUNoiseModel",
    "WishartPrior",
    "NIGModel",
    "NIG_PRIOR_KAPPA",
    "NIG_PRIOR_ALPHA",
    "NIG_PRIOR_BETA",
    "combine_independent_weights",
    # Routing
    "DirichletRoutingModule",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "DirichletRoutingModule": ("fl_slam_poc.backend.dirichlet_routing", "DirichletRoutingModule"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))
