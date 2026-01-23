"""
Geometry package for FL-SLAM.

Consolidated SE(3), SO(3), and vMF operations with both NumPy and JAX backends.

Modules:
- shared: Common constants and utilities
- se3_numpy: NumPy-based SE(3) operations (CPU)
- se3_jax: JAX-based SE(3) operations (GPU-accelerated)
- vmf: von Mises-Fisher geometry (surface normals, directional data)

Usage:
    # SE(3) operations
    from fl_slam_poc.common.geometry import (
        se3_compose,
        se3_inverse,
        se3_adjoint,
        rotvec_to_rotmat,
        rotmat_to_rotvec,
    )
    
    # vMF geometry
    from fl_slam_poc.common.geometry.vmf import (
        vmf_make_evidence,
        vmf_barycenter,
    )
"""

from __future__ import annotations

# Direct imports from se3_numpy
from fl_slam_poc.common.geometry.se3_numpy import (
    # Constants
    ROTATION_EPSILON,
    SINGULARITY_EPSILON,
    # SO(3) operations
    skew,
    unskew,
    rotvec_to_rotmat,
    rotmat_to_rotvec,
    # Quaternion operations
    quat_to_rotmat,
    rotmat_to_quat,
    quat_to_rotvec,
    # SE(3) operations
    se3_compose,
    se3_inverse,
    se3_relative,
    se3_apply,
    se3_adjoint,
    se3_cov_compose,
    se3_exp,
    se3_log,
)

# vMF geometry (directional data, surface normals)
from fl_slam_poc.common.geometry.vmf import vmf_make_evidence, vmf_barycenter

__all__ = [
    # Constants
    "ROTATION_EPSILON",
    "SINGULARITY_EPSILON",
    # SO(3) operations
    "skew",
    "unskew",
    "rotvec_to_rotmat",
    "rotmat_to_rotvec",
    # Quaternion operations
    "quat_to_rotmat",
    "rotmat_to_quat",
    "quat_to_rotvec",
    # SE(3) operations
    "se3_compose",
    "se3_inverse",
    "se3_relative",
    "se3_apply",
    "se3_adjoint",
    "se3_cov_compose",
    "se3_exp",
    "se3_log",
    # vMF geometry
    "vmf_make_evidence",
    "vmf_barycenter",
]
