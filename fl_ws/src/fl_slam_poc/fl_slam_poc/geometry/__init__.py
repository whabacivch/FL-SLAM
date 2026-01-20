"""
Geometry module - Legacy compatibility.

NOTE: Geometry has moved to fl_slam_poc.common.transforms.
This module provides backward compatibility re-exports.
"""

from fl_slam_poc.common.transforms import (
    quat_to_rotmat,
    rotmat_to_quat,
    quat_to_rotvec,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
    se3_apply,
    se3_compose,
    se3_inverse,
    se3_adjoint,
    se3_cov_compose,
    se3_log,
    se3_exp,
)

__all__ = [
    "quat_to_rotmat",
    "rotmat_to_quat",
    "quat_to_rotvec",
    "rotmat_to_rotvec",
    "rotvec_to_rotmat",
    "se3_apply",
    "se3_compose",
    "se3_inverse",
    "se3_adjoint",
    "se3_cov_compose",
    "se3_log",
    "se3_exp",
]
