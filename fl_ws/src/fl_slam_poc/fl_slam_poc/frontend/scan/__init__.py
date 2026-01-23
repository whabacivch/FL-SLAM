"""
Scan processing subpackage.

Handles:
- Point cloud processing and filtering (pointcloud_gpu)
- ICP registration and alignment (icp)
- Descriptor extraction (descriptor_builder)
- Visual feature extraction (visual_feature_extractor)

Usage:
    from fl_slam_poc.frontend.scan import (
        PointCloudProcessor,
        icp_3d,
        DescriptorBuilder,
    )
"""

from __future__ import annotations

from fl_slam_poc.frontend.scan.pointcloud_gpu import GPUPointCloudProcessor, is_gpu_available
from fl_slam_poc.frontend.scan.icp import (
    icp_3d,
    icp_information_weight,
    icp_covariance_tangent,
    transport_covariance_to_frame,
    best_fit_se3,
    ICPResult,
)
from fl_slam_poc.common import constants
from fl_slam_poc.frontend.scan.descriptor_builder import DescriptorBuilder

__all__ = [
    "GPUPointCloudProcessor",
    "is_gpu_available",
    "icp_3d",
    "icp_information_weight",
    "icp_covariance_tangent",
    "transport_covariance_to_frame",
    "best_fit_se3",
    "ICPResult",
    "N_MIN_SE3_DOF",
    "K_SIGMOID",
    "DescriptorBuilder",
]

# Backward-compatible exports (now defined in constants).
N_MIN_SE3_DOF = constants.N_MIN_SE3_DOF
K_SIGMOID = constants.K_SIGMOID
