"""Contract validation utilities for FL-SLAM.

This module provides validation functions to enforce contracts at module boundaries,
detect hardcoded fake values, and catch common bugs early.
"""
import numpy as np
from typing import Any

from fl_slam_poc.common.constants import NUMERICAL_EPSILON


class ContractViolation(Exception):
    """Raised when a data contract is violated."""
    pass


def validate_pose(pose: np.ndarray, name: str = "pose") -> None:
    """Validate 6D pose vector [x, y, z, rx, ry, rz].

    Args:
        pose: 6D pose vector
        name: Name for error messages

    Raises:
        ContractViolation: If pose is invalid
    """
    if pose.shape != (6,):
        raise ContractViolation(f"{name}: Expected shape (6,), got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ContractViolation(f"{name}: Contains inf/nan: {pose}")


def validate_covariance(cov: np.ndarray, name: str = "covariance") -> None:
    """Validate covariance matrix is positive semi-definite.

    Args:
        cov: Covariance matrix
        name: Name for error messages

    Raises:
        ContractViolation: If covariance is invalid
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ContractViolation(f"{name}: Not a square matrix: {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ContractViolation(f"{name}: Contains inf/nan")

    # Check symmetry
    if not np.allclose(cov, cov.T, atol=NUMERICAL_EPSILON):
        raise ContractViolation(f"{name}: Not symmetric, max asymmetry: {np.max(np.abs(cov - cov.T))}")

    # Check positive semi-definite
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < -NUMERICAL_EPSILON):
        raise ContractViolation(f"{name}: Not PSD, min eigenvalue: {eigvals.min():.3e}")


def validate_information_form(L: np.ndarray, h: np.ndarray, name: str = "info") -> None:
    """Validate information form (L, h) representation.

    Args:
        L: Information matrix (precision matrix)
        h: Information vector
        name: Name for error messages

    Raises:
        ContractViolation: If information form is invalid
    """
    validate_covariance(L, name=f"{name}.L (information matrix)")
    if h.ndim != 1:
        raise ContractViolation(f"{name}.h: Expected 1D vector, got shape {h.shape}")
    if L.shape[0] != h.shape[0]:
        raise ContractViolation(f"{name}: L shape {L.shape} inconsistent with h shape {h.shape}")
    if not np.all(np.isfinite(h)):
        raise ContractViolation(f"{name}.h: Contains inf/nan: {h}")


def validate_timestamp(stamp_sec: float, name: str = "timestamp") -> None:
    """Validate timestamp is positive and reasonable.

    Args:
        stamp_sec: Timestamp in seconds since Unix epoch
        name: Name for error messages

    Raises:
        ContractViolation: If timestamp is invalid
    """
    if not np.isfinite(stamp_sec):
        raise ContractViolation(f"{name}: Not finite: {stamp_sec}")
    if stamp_sec <= 0:
        raise ContractViolation(f"{name}: Not positive: {stamp_sec}")
    # Sanity check: between 2020-01-01 and 2040-01-01
    if stamp_sec < 1577836800 or stamp_sec > 2208988800:
        raise ContractViolation(
            f"{name}: Unreasonable value: {stamp_sec} "
            f"(outside 2020-2040 range)"
        )


def detect_hardcoded_value(
    value: Any,
    expected_hardcoded: Any,
    name: str,
    tolerance: float = 1e-10
) -> None:
    """Detect if a value is suspiciously equal to a hardcoded constant.

    This helps catch bugs where debug/placeholder values are being used
    instead of actual computed values.

    Args:
        value: Value to check
        expected_hardcoded: Hardcoded value to check against
        name: Name for error messages
        tolerance: Tolerance for equality check (for arrays)

    Raises:
        ContractViolation: If value appears to be hardcoded
    """
    if isinstance(value, np.ndarray):
        if np.allclose(value, expected_hardcoded, atol=tolerance):
            raise ContractViolation(
                f"{name}: Suspiciously equal to hardcoded value. "
                "This may indicate fake debug values are being used. "
                f"Value shape: {value.shape}"
            )
    elif value == expected_hardcoded:
        raise ContractViolation(
            f"{name}: Exactly equal to hardcoded value {expected_hardcoded}. "
            "This may indicate fake debug values are being used."
        )


def warn_near_zero_delta(
    delta: np.ndarray,
    name: str,
    threshold: float = 1e-9
) -> bool:
    """Check if a delta value is suspiciously close to zero.

    Returns True if near zero (indicating potential dead integration).

    Args:
        delta: Delta vector to check
        name: Name for warning messages
        threshold: Threshold for zero check

    Returns:
        True if delta is near zero (warning condition)
    """
    norm = np.linalg.norm(delta)
    return norm < threshold


def validate_imu_factor(
    delta_p: np.ndarray,
    delta_v: np.ndarray,
    delta_theta: np.ndarray,
    covariance: np.ndarray,
    timestamp_start: float,
    timestamp_end: float
) -> None:
    """Validate IMU factor contract.

    Args:
        delta_p: Position delta [3]
        delta_v: Velocity delta [3]
        delta_theta: Rotation delta (axis-angle) [3]
        covariance: 9x9 covariance matrix
        timestamp_start: Start timestamp (seconds)
        timestamp_end: End timestamp (seconds)

    Raises:
        ContractViolation: If any field violates contract
    """
    # Validate timestamps
    validate_timestamp(timestamp_start, "imu_factor.timestamp_start")
    validate_timestamp(timestamp_end, "imu_factor.timestamp_end")

    if timestamp_end <= timestamp_start:
        raise ContractViolation(
            f"imu_factor: timestamp_end ({timestamp_end:.6f}) <= "
            f"timestamp_start ({timestamp_start:.6f})"
        )

    # Validate shapes
    if delta_p.shape != (3,):
        raise ContractViolation(f"imu_factor.delta_p: Expected shape (3,), got {delta_p.shape}")
    if delta_v.shape != (3,):
        raise ContractViolation(f"imu_factor.delta_v: Expected shape (3,), got {delta_v.shape}")
    if delta_theta.shape != (3,):
        raise ContractViolation(f"imu_factor.delta_theta: Expected shape (3,), got {delta_theta.shape}")

    # Validate finite values
    if not np.all(np.isfinite(delta_p)):
        raise ContractViolation(f"imu_factor.delta_p: Contains inf/nan: {delta_p}")
    if not np.all(np.isfinite(delta_v)):
        raise ContractViolation(f"imu_factor.delta_v: Contains inf/nan: {delta_v}")
    if not np.all(np.isfinite(delta_theta)):
        raise ContractViolation(f"imu_factor.delta_theta: Contains inf/nan: {delta_theta}")

    # Validate covariance
    validate_covariance(covariance, "imu_factor.covariance")

    # Detect hardcoded identity covariance (common fake value)
    detect_hardcoded_value(covariance, np.eye(9), "imu_factor.covariance")
