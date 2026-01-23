"""
SE(3) geometry using Lie algebra (tangent space) representation.

State representation: (x, y, z, rx, ry, rz) where:
- (x, y, z): translation in R^3
- (rx, ry, rz): rotation vector (axis-angle) in so(3)

This avoids RPY singularities and enables proper covariance transport
via the adjoint representation. Following information geometry principles,
operations are closed-form in the tangent space.

Numerical Policy:
    Epsilon thresholds are chosen based on IEEE 754 double precision:
    - ROTATION_EPSILON = 1e-10: ~sqrt(machine_epsilon) for stable trig
    - SINGULARITY_EPSILON = 1e-6: threshold for π-singularity handling
    
    These are NUMERICAL STABILITY choices, not model parameters.
    They affect only the computational path, not the mathematical result.

References:
- Barfoot (2017): State Estimation for Robotics
- Sola et al. (2018): A micro Lie theory for state estimation
- Combe (2022-2025): Pre-Frobenius manifolds and information geometry
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# =============================================================================
# Numerical Constants (stability, not policy)
# =============================================================================
# These are chosen based on IEEE 754 double precision (~15 decimal digits)
# They affect computational path only, not mathematical result.

# For small-angle approximations: use when θ < ε to avoid division by ~0
# Choice: ~sqrt(machine_epsilon) ≈ 1e-8, use 1e-10 for safety margin
ROTATION_EPSILON: float = 1e-10

# For π-singularity handling: eigenvalue decomposition threshold
# Choice: 1e-6 provides stable numerics near θ = π
SINGULARITY_EPSILON: float = 1e-6


# =============================================================================
# Rotation vector <-> Rotation matrix conversions (so(3) <-> SO(3))
# =============================================================================


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector (hat operator)."""
    v = np.asarray(v, dtype=float).reshape(-1)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)


def unskew(S: np.ndarray) -> np.ndarray:
    """Extract 3-vector from skew-symmetric matrix (vee operator)."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector (axis-angle) to rotation matrix.
    Uses Rodrigues' formula: R = I + sin(θ)[ω]_× + (1-cos(θ))[ω]_×²
    
    This is the exponential map exp: so(3) -> SO(3).
    
    Numerical note: For θ < ROTATION_EPSILON, uses Taylor expansion
    to avoid division by small numbers. This is numerically equivalent
    to the exact formula.
    """
    rotvec = np.asarray(rotvec, dtype=float).reshape(-1)
    theta = np.linalg.norm(rotvec)
    
    if theta < ROTATION_EPSILON:
        # Small angle: R ≈ I + [rotvec]_× (first-order Taylor)
        # Error is O(θ²), which is < 1e-20 for θ < 1e-10
        return np.eye(3, dtype=float) + skew(rotvec)
    
    axis = rotvec / theta
    K = skew(axis)
    
    # Rodrigues' formula
    R = np.eye(3, dtype=float) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    return R


def rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to rotation vector (axis-angle).
    This is the logarithmic map log: SO(3) -> so(3).
    
    Handles three cases:
    1. θ ≈ 0: Extract from skew-symmetric part
    2. θ ≈ π: Use eigenvalue decomposition (singularity)
    3. Otherwise: Standard formula
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {R.shape}")
    
    # Check orthogonality (should be ~I, but allow small numerical error)
    RRT = R @ R.T
    if not np.allclose(RRT, np.eye(3), atol=1e-5):
        raise ValueError("Input matrix is not orthogonal (R @ R.T != I)")
    
    # Trace-based angle computation
    trace = np.trace(R)
    theta = math.acos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    
    if theta < ROTATION_EPSILON:
        # Case 1: Small angle - extract from skew-symmetric part
        # log(R) ≈ (R - R.T) / 2 for small angles
        S = (R - R.T) / 2.0
        return unskew(S)
    
    if abs(theta - math.pi) < SINGULARITY_EPSILON:
        # Case 2: Near π - use eigenvalue decomposition
        # R has eigenvalue 1 with eigenvector = rotation axis
        eigenvals, eigenvecs = np.linalg.eig(R)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvals - 1.0))
        axis = np.real(eigenvecs[:, idx])
        axis = axis / np.linalg.norm(axis)
        # Sign ambiguity: choose axis that gives positive trace
        if np.trace(R) < -1:
            axis = -axis
        return axis * math.pi
    
    # Case 3: Standard formula
    # log(R) = (θ / (2 sin(θ))) * (R - R.T)
    S = (R - R.T) / 2.0
    rotvec = unskew(S)
    rotvec = rotvec * (theta / math.sin(theta))
    return rotvec


# =============================================================================
# Quaternion conversions (for ROS compatibility)
# =============================================================================


def quat_to_rotmat(x_or_q, y=None, z=None, w=None) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to rotation matrix.
    
    ROS convention: q = [x, y, z, w] where w is scalar.
    
    Can be called as:
        quat_to_rotmat(np.array([x, y, z, w]))
        quat_to_rotmat(x, y, z, w)
    """
    # Support both array and separate args
    if y is not None and z is not None and w is not None:
        q = np.array([x_or_q, y, z, w], dtype=float)
    else:
        q = np.asarray(x_or_q, dtype=float).reshape(-1)
    
    if len(q) != 4:
        raise ValueError(f"Expected 4-element quaternion, got {len(q)}")
    
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    # Normalize
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-10:
        raise ValueError("Quaternion norm is too small (near zero)")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R


def rotmat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert rotation matrix to quaternion (x, y, z, w).
    
    Uses Shepperd's method for numerical stability.
    Returns ROS convention: [x, y, z, w] where w is scalar.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {R.shape}")
    
    trace = np.trace(R)
    
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return (x, y, z, w)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation vector."""
    R = quat_to_rotmat(q)
    return rotmat_to_rotvec(R)


# =============================================================================
# SE(3) group operations
# =============================================================================


def se3_compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Compose two SE(3) transforms: T_result = T1 ∘ T2.
    
    Args:
        T1, T2: SE(3) transforms as 6D vectors (x, y, z, rx, ry, rz)
    
    Returns:
        6D vector representing T1 ∘ T2
    """
    T1 = np.asarray(T1, dtype=float).reshape(-1)
    T2 = np.asarray(T2, dtype=float).reshape(-1)
    
    if len(T1) != 6 or len(T2) != 6:
        raise ValueError(f"Expected 6D vectors, got shapes {T1.shape}, {T2.shape}")
    
    t1 = T1[:3]
    t2 = T2[:3]
    rvec1 = T1[3:6]
    rvec2 = T2[3:6]
    
    R1 = rotvec_to_rotmat(rvec1)
    R2 = rotvec_to_rotmat(rvec2)
    
    # T_result = T1 * T2
    # R_result = R1 * R2
    # t_result = R1 * t2 + t1
    R_result = R1 @ R2
    t_result = R1 @ t2 + t1
    
    rvec_result = rotmat_to_rotvec(R_result)
    return np.concatenate([t_result, rvec_result])


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """
    Compute inverse of SE(3) transform: T_inv such that T ∘ T_inv = I.
    
    Args:
        T: SE(3) transform as 6D vector (x, y, z, rx, ry, rz)
    
    Returns:
        6D vector representing T^{-1}
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    if len(T) != 6:
        raise ValueError(f"Expected 6D vector, got shape {T.shape}")
    
    t = T[:3]
    rvec = T[3:6]
    
    R = rotvec_to_rotmat(rvec)
    R_inv = R.T  # For rotation matrices, inverse = transpose
    
    # T_inv = [R_inv, -R_inv * t]
    t_inv = -R_inv @ t
    rvec_inv = rotmat_to_rotvec(R_inv)
    
    return np.concatenate([t_inv, rvec_inv])


def se3_relative(T_from: np.ndarray, T_to: np.ndarray) -> np.ndarray:
    """
    Compute relative transform: T_rel = T_from^{-1} ∘ T_to.
    
    Args:
        T_from, T_to: SE(3) transforms as 6D vectors
    
    Returns:
        6D vector representing relative transform
    """
    T_from_inv = se3_inverse(T_from)
    return se3_compose(T_from_inv, T_to)


def se3_apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Apply SE(3) transform to point(s): p_transformed = T * p.
    
    Args:
        T: SE(3) transform as 6D vector
        p: 3D point (3,) or batch of points (N, 3)
    
    Returns:
        3D transformed point(s), same shape as input
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float)
    
    if len(T) != 6:
        raise ValueError(f"Expected 6D transform, got shape {T.shape}")
    
    t = T[:3]
    rvec = T[3:6]
    R = rotvec_to_rotmat(rvec)
    
    # Support both single point and batch
    if p.ndim == 1:
        if len(p) != 3:
            raise ValueError(f"Expected 3D point, got shape {p.shape}")
        return R @ p + t
    elif p.ndim == 2:
        if p.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) points, got shape {p.shape}")
        return (R @ p.T).T + t
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {p.shape}")


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """
    Compute adjoint representation of SE(3) transform.
    
    Adjoint is used for covariance transport:
        Cov(T * x) = Adjoint(T) * Cov(x) * Adjoint(T)^T
    
    Args:
        T: SE(3) transform as 6D vector
    
    Returns:
        6x6 adjoint matrix
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    if len(T) != 6:
        raise ValueError(f"Expected 6D vector, got shape {T.shape}")
    
    t = T[:3]
    rvec = T[3:6]
    R = rotvec_to_rotmat(rvec)
    
    # Adjoint = [R, [t]_× * R]
    #          [0, R]
    t_skew = skew(t)
    t_skew_R = t_skew @ R
    
    Ad = np.zeros((6, 6), dtype=float)
    Ad[:3, :3] = R
    Ad[:3, 3:6] = t_skew_R
    Ad[3:6, 3:6] = R
    
    return Ad


def se3_cov_compose(cov_a: np.ndarray, cov_b: np.ndarray, T: np.ndarray = None) -> np.ndarray:
    """
    Compose two covariances with optional SE(3) transport.
    
    If T is provided:
        Result = cov_a + Ad(T) @ cov_b @ Ad(T).T
        (Transport cov_b through T, then add cov_a)
    
    If T is None or identity:
        Result = cov_a + cov_b
    
    Args:
        cov_a: 6x6 covariance matrix (additive term)
        cov_b: 6x6 covariance matrix (transported term)
        T: SE(3) transform as 6D vector (optional, defaults to identity)
    
    Returns:
        6x6 composed covariance matrix
    """
    cov_a = np.asarray(cov_a, dtype=float)
    cov_b = np.asarray(cov_b, dtype=float)
    
    if cov_a.shape != (6, 6):
        raise ValueError(f"Expected 6x6 covariance for cov_a, got shape {cov_a.shape}")
    if cov_b.shape != (6, 6):
        raise ValueError(f"Expected 6x6 covariance for cov_b, got shape {cov_b.shape}")
    
    if T is None:
        # No transport, just add
        return cov_a + cov_b
    
    T = np.asarray(T, dtype=float).reshape(-1)
    if len(T) != 6:
        raise ValueError(f"Expected 6D vector for T, got shape {T.shape}")
    
    # Transport cov_b through T using adjoint
    Ad = se3_adjoint(T)
    cov_b_transported = Ad @ cov_b @ Ad.T
    
    return cov_a + cov_b_transported


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map: se(3) -> SE(3).
    
    Maps a 6D twist (v, ω) in se(3) to an SE(3) transform.
    
    Args:
        xi: 6D twist vector (vx, vy, vz, ωx, ωy, ωz)
    
    Returns:
        6D SE(3) transform (x, y, z, rx, ry, rz)
    """
    xi = np.asarray(xi, dtype=float).reshape(-1)
    if len(xi) != 6:
        raise ValueError(f"Expected 6D twist, got shape {xi.shape}")
    
    v = xi[:3]
    omega = xi[3:6]
    
    theta = np.linalg.norm(omega)
    
    if theta < ROTATION_EPSILON:
        # Small rotation: use first-order approximation
        R = np.eye(3, dtype=float) + skew(omega)
        t = v
    else:
        # Standard formula
        axis = omega / theta
        K = skew(axis)
        R = np.eye(3, dtype=float) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
        
        # Translation part
        V = np.eye(3, dtype=float) + ((1.0 - math.cos(theta)) / theta) * K + ((theta - math.sin(theta)) / theta) * (K @ K)
        t = V @ v
    
    rvec = rotmat_to_rotvec(R)
    return np.concatenate([t, rvec])


def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithmic map: SE(3) -> se(3).
    
    Maps an SE(3) transform to a 6D twist in se(3).
    
    Args:
        T: 6D SE(3) transform (x, y, z, rx, ry, rz)
    
    Returns:
        6D twist vector (vx, vy, vz, ωx, ωy, ωz)
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    if len(T) != 6:
        raise ValueError(f"Expected 6D transform, got shape {T.shape}")
    
    t = T[:3]
    rvec = T[3:6]
    R = rotvec_to_rotmat(rvec)
    
    theta = np.linalg.norm(rvec)
    
    if theta < ROTATION_EPSILON:
        # Small rotation: use first-order approximation
        omega = unskew(R - np.eye(3, dtype=float))
        V_inv = np.eye(3, dtype=float)
    else:
        # Standard formula
        omega = rvec
        K = skew(omega / theta)
        V_inv = np.eye(3, dtype=float) - 0.5 * K + ((1.0 - (theta / 2.0) * (1.0 / math.tan(theta / 2.0))) / (theta * theta)) * (K @ K)
    
    v = V_inv @ t
    return np.concatenate([v, omega])
