"""
Frontend loop detection modules for FL-SLAM.

Handles loop closure detection, ICP registration, and point cloud processing.
"""

from fl_slam_poc.frontend.loops.loop_processor import LoopProcessor
from fl_slam_poc.frontend.loops.icp import (
    ICPResult,
    icp_3d,
    best_fit_se3,
    icp_information_weight,
    icp_covariance_tangent,
    transport_covariance_to_frame,
    N_MIN_SE3_DOF,
    K_SIGMOID,
)
from fl_slam_poc.frontend.loops.pointcloud_gpu import (
    GPUPointCloudProcessor,
    is_gpu_available,
    voxel_filter_gpu,
    icp_gpu,
)
from fl_slam_poc.frontend.loops.vmf_geometry import (
    vmf_make_evidence,
    vmf_mean_param,
    vmf_barycenter,
    vmf_fisher_rao_distance,
    vmf_third_order_correction,
    vmf_hellinger_distance,
    A_d,
    A_d_inverse_series,
)

__all__ = [
    "LoopProcessor",
    # ICP
    "ICPResult",
    "icp_3d",
    "best_fit_se3",
    "icp_information_weight",
    "icp_covariance_tangent",
    "transport_covariance_to_frame",
    "N_MIN_SE3_DOF",
    "K_SIGMOID",
    # GPU
    "GPUPointCloudProcessor",
    "is_gpu_available",
    "voxel_filter_gpu",
    "icp_gpu",
    # vMF
    "vmf_make_evidence",
    "vmf_mean_param",
    "vmf_barycenter",
    "vmf_fisher_rao_distance",
    "vmf_third_order_correction",
    "vmf_hellinger_distance",
    "A_d",
    "A_d_inverse_series",
]
