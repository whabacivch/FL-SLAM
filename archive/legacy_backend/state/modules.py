"""State modules for the backend atlas."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from fl_slam_poc.backend.fusion.gaussian_info import fuse_info, make_evidence, mean_cov
from fl_slam_poc.common.geometry.vmf import vmf_barycenter
from fl_slam_poc.common import constants


class BaseModule:
    """Base class for all modules in the atlas."""

    def __init__(self, module_id: int, module_type: str):
        self.module_id = module_id
        self.module_type = module_type  # "sparse_anchor" or "dense_3d"
        self.mass = float(constants.MODULE_MASS_PRIOR)
        self.last_updated = time.time()


class SparseAnchorModule(BaseModule):
    """
    Sparse anchor from laser SLAM.

    Stores:
    - SE(3) pose (mu) and covariance in information form (L, h)
    - Point cloud for visualization
    - Optional: NIG descriptor model

    Can be upgraded to 3D when fused with RGB-D evidence.
    """

    def __init__(self, anchor_id: int, mu: np.ndarray, cov: np.ndarray, points: Optional[np.ndarray] = None):
        super().__init__(anchor_id, "sparse_anchor")
        self.mu = mu.copy()
        self.cov = cov.copy()
        self.L, self.h = make_evidence(mu, cov)
        self.points = points.copy() if points is not None else np.empty((0, 3))
        self.desc_model = None  # NIG descriptor (set by frontend)
        self.rgbd_fused = False  # True if RGB-D evidence has been fused

    def fuse_rgbd_position(self, rgbd_L: np.ndarray, rgbd_h: np.ndarray, weight: float = 1.0):
        """
        Fuse RGB-D 3D position evidence at this anchor.

        Uses information form addition (exact, closed-form).
        """
        # Direct 3D fusion (anchor is already 6D SE(3), use position part)
        # Extract position-only information
        L_pos = self.L[:3, :3]
        h_pos = self.h[:3]

        # Fuse position evidence
        L_pos_fused = L_pos + weight * rgbd_L
        h_pos_fused = h_pos + weight * rgbd_h.reshape(-1)

        # Update anchor's position components
        self.L[:3, :3] = L_pos_fused
        self.h[:3] = h_pos_fused

        # Recover mean/cov
        self.mu, self.cov = mean_cov(self.L, self.h)
        self.mass += weight
        self.last_updated = time.time()
        self.rgbd_fused = True


class Dense3DModule(BaseModule):
    """
    Dense 3D Gaussian module from RGB-D.

    Stores:
    - 3D position + covariance in information form
    - vMF normal (surface normal as θ = κμ)
    - Color (RGB Gaussian)
    - Opacity (scalar)
    """

    def __init__(self, module_id: int, mu: np.ndarray, cov: np.ndarray):
        super().__init__(module_id, "dense_3d")
        self.mu = mu.copy()
        self.cov = cov.copy()
        self.L, self.h = make_evidence(mu, cov)

        # vMF normal (default: pointing up, κ=0 isotropic)
        self.normal_theta = np.array([0.0, 0.0, 1.0])

        # Color (RGB Gaussian)
        self.color_mean = np.array([0.5, 0.5, 0.5])
        self.color_cov = np.eye(3) * 0.01
        self.color_L, self.color_h = make_evidence(self.color_mean, self.color_cov)

        # Opacity
        self.alpha_mean = 1.0
        self.alpha_var = 0.1

    def update_from_evidence(self, evidence: dict, weight: float = 1.0):
        """
        Update module from RGB-D evidence dict.

        All operations use exact closed-form exponential family fusion.
        """
        # Position fusion (Gaussian info form)
        self.L, self.h = fuse_info(
            self.L,
            self.h,
            evidence["position_L"],
            evidence["position_h"],
            weight=weight,
        )
        self.mu, self.cov = mean_cov(self.L, self.h)

        # Normal fusion (vMF barycenter - exact via Bessel)
        thetas = [self.normal_theta, evidence["normal_theta"]]
        weights_vmf = [self.mass, weight]
        self.normal_theta, _ = vmf_barycenter(thetas, weights_vmf, d=3)

        # Color fusion (Gaussian info form)
        self.color_L, self.color_h = fuse_info(
            self.color_L,
            self.color_h,
            evidence["color_L"],
            evidence["color_h"],
            weight=weight,
        )
        self.color_mean, self.color_cov = mean_cov(self.color_L, self.color_h)

        # Opacity fusion (weighted average)
        obs_alpha = evidence.get("alpha_mean", 1.0)
        self.alpha_mean = (self.mass * self.alpha_mean + weight * obs_alpha) / (self.mass + weight)

        self.mass += weight
        self.last_updated = time.time()
