"""
Lightweight visual feature data types for GC SLAM (C++ preprocessing path).

This module intentionally contains only data structures used by the backend
and splat fusion code. It avoids OpenCV/Python feature extraction dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Feature3D:
    """A single visual feature lifted to 3D (camera frame) with uncertainty and soft weight."""
    # Pixel location
    u: float
    v: float

    # 3D point in camera frame
    xyz: np.ndarray  # (3,)

    # 3x3 covariance (camera frame)
    cov_xyz: np.ndarray  # (3,3)

    # 3x3 precision (information) = inv(cov_xyz), regularized
    info_xyz: np.ndarray  # (3,3)

    # log(det(cov_xyz)) for IG/MHT scoring (regularized)
    logdet_cov: float

    # Canonical natural parameters (θ = Λ μ)
    canonical_theta: np.ndarray  # (3,)
    canonical_log_partition: float

    # Descriptor (e.g., ORB binary)
    desc: np.ndarray  # (D,) dtype=uint8 or float32

    # Continuous measurement weight in [0,1] (for budgeting/prioritization)
    weight: float

    # Metadata for debugging/audit
    meta: Dict[str, Any]

    # Optional appearance (vMF): view-direction or normal; used for Hellinger / orthogonal score
    mu_app: Optional[np.ndarray] = None  # (3,) unit vector
    kappa_app: float = 0.0

    # Optional RGB from image at (u,v); [0,1]; used for map/splat coloring
    color: Optional[np.ndarray] = None  # (3,) RGB


@dataclass
class ExtractionResult:
    features: List[Feature3D]
    op_report: List[Any]
    timestamp_ns: int
