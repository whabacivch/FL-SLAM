"""
Geometry module for SE(3) operations using Lie algebra representation.

SE(3) Representation Conventions
=================================

This module uses ROTATION VECTOR (axis-angle) representation throughout:

State Vector: (x, y, z, rx, ry, rz)
  - (x, y, z): Translation in R³ (meters)
  - (rx, ry, rz): Rotation vector in so(3) tangent space (radians)
  
The rotation vector encodes both axis and angle:
  - Direction: rotation axis (unit vector after normalization)
  - Magnitude: rotation angle in radians
  - Example: [0, 0, π/2] = 90° rotation about z-axis

Why Rotation Vectors?
---------------------
1. NO SINGULARITIES: Unlike Euler angles (gimbal lock at ±90°)
2. MINIMAL PARAMETERS: 3 DOF (vs 4 for quaternions, 9 for matrices)
3. TANGENT SPACE: Natural for covariance (6×6 in se(3))
4. EXACT OPERATIONS: Composition via Baker-Campbell-Hausdorff (BCH)

Representation Boundaries
--------------------------
Input/Output Layer (ROS Messages):
  - Quaternions (x, y, z, w) from geometry_msgs
  - Use quat_to_rotvec() for efficient conversion
  
Internal State (Backend/Frontend):
  - Rotation vectors (rx, ry, rz) everywhere
  - Covariance in se(3) tangent space
  
Computation (SE(3) Operations):
  - Rotation matrices for group operations
  - Convert via rotvec_to_rotmat() / rotmat_to_rotvec()
  
Key Insight: Rotation matrices are INTERMEDIATE, not stored.
We only materialize them for composition, then convert back.

Information Geometry Principles
--------------------------------
Following Barfoot (2017) and information geometry best practices:

1. COVARIANCE TRANSPORT via Adjoint:
   - Σ' = Ad_T @ Σ @ Ad_T.T (exact, not additive)
   - Preserves uncertainty geometry under transforms
   
2. NO LINEARIZATION in composition:
   - se3_compose() is exact group operation
   - Linearization only at sensor→evidence boundary (ICP)
   
3. CLOSED-FORM where possible:
   - Rodrigues formula for exp/log maps
   - SVD for pose alignment (ICP)
   - No iterative solvers in core geometry

Performance Notes
-----------------
- Rotation vector ↔ matrix: O(1) closed-form
- Quaternion → rotation vector: Direct (no matrix intermediate)
- SE(3) composition: O(1) matrix multiply + log/exp
- Adjoint covariance: O(n²) matrix operations

References
----------
- Barfoot (2017): "State Estimation for Robotics"
- Sola et al. (2018): "A micro Lie theory for state estimation"
- Combe (2022-2025): Information geometry framework
"""

from fl_slam_poc.geometry.se3 import (
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
    "quat_to_rotvec",  # NEW: Direct conversion (more efficient)
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

