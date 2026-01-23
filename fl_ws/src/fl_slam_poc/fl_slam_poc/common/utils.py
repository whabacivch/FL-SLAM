"""
Common utility functions.

Consolidates duplicated functions from across the codebase.
"""
import numpy as np
from typing import Dict


def stamp_to_sec(stamp) -> float:
    """Convert ROS2 builtin_interfaces/Time to float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def vec_stats(arr: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for a vector/array.

    Args:
        arr: Input array

    Returns:
        dict with keys: mean, std, min, max, norm
    """
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "norm": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "norm": float(np.linalg.norm(arr)),
    }
