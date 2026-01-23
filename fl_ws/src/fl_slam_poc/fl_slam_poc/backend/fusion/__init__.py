"""Closed-form fusion operators and barycenters."""

from fl_slam_poc.backend.fusion.gaussian_geom import gaussian_frobenius_correction
from fl_slam_poc.backend.fusion.gaussian_info import (
    make_evidence,
    fuse_info,
    mean_cov,
    trust_scaled_fusion,
    ALPHA_DIVERGENCE_DEFAULT,
    MAX_ALPHA_DIVERGENCE_PRIOR,
)
from fl_slam_poc.backend.fusion.information_distances import (
    fisher_rao_student_t,
    hellinger_gaussian,
)
from fl_slam_poc.backend.fusion.weights import combine_independent_weights

__all__ = [
    "gaussian_frobenius_correction",
    "make_evidence",
    "fuse_info",
    "mean_cov",
    "trust_scaled_fusion",
    "ALPHA_DIVERGENCE_DEFAULT",
    "MAX_ALPHA_DIVERGENCE_PRIOR",
    "fisher_rao_student_t",
    "hellinger_gaussian",
    "combine_independent_weights",
]
