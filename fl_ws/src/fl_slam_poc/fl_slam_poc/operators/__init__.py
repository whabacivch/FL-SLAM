"""
FL-SLAM Operators - Legacy compatibility module.

NOTE: This module provides backward compatibility re-exports.
Operators have been reorganized into:
- fl_slam_poc.backend.fusion/ (Gaussian, information distances, multimodal)
- fl_slam_poc.frontend.loops/ (ICP, vMF, point cloud GPU)
- fl_slam_poc.common/ (OpReport)
- fl_slam_poc.operators/ (Dirichlet - EXPERIMENTAL)

New code should import from the specific subpackages directly.
"""

# Re-export from backend/fusion/
from fl_slam_poc.backend.fusion.gaussian_info import (
    make_evidence,
    fuse_info,
    mean_cov,
    log_partition,
    kl_divergence,
    hellinger_distance,
    bhattacharyya_coefficient,
    fisher_information,
    natural_gradient,
    marginalize,
    condition,
    product_of_experts,
)

from fl_slam_poc.backend.fusion.information_distances import (
    hellinger_sq_expfam,
    hellinger_expfam,
    hellinger_gaussian,
    fisher_rao_gaussian_1d,
    fisher_rao_student_t,
    fisher_rao_student_t_vec,
    fisher_rao_spd,
    product_distance,
    product_distance_weighted,
    gaussian_kl,
    gaussian_kl_symmetric,
    wishart_bregman,
    bhattacharyya_coefficient_gaussian,
    bhattacharyya_distance_gaussian,
)

from fl_slam_poc.backend.fusion.gaussian_geom import gaussian_frobenius_correction

from fl_slam_poc.backend.fusion.multimodal_fusion import (
    laser_2d_to_3d_constraint,
    fuse_laser_rgbd,
    fuse_multimodal_3d,
    spatial_association_weight,
)

# Re-export from frontend/loops/
from fl_slam_poc.frontend.loops.icp import (
    ICPResult,
    best_fit_se3,
    icp_3d,
    icp_information_weight,
    icp_covariance_tangent,
    transport_covariance_to_frame,
    N_MIN_SE3_DOF,
    K_SIGMOID,
)

from fl_slam_poc.frontend.loops.vmf_geometry import (
    A_d,
    A_d_inverse_series,
    vmf_make_evidence,
    vmf_mean_param,
    vmf_barycenter,
    vmf_fisher_rao_distance,
    vmf_third_order_correction,
    vmf_hellinger_distance,
)

from fl_slam_poc.frontend.loops.pointcloud_gpu import (
    GPUPointCloudProcessor,
    is_gpu_available,
    voxel_filter_gpu,
    icp_gpu,
)

# Re-export from common/
from fl_slam_poc.common.op_report import OpReport

# Dirichlet geometry (EXPERIMENTAL - stays in operators/)
from fl_slam_poc.operators.dirichlet_geom import (
    dirichlet_log_partition,
    psi_potential,
    g_fisher,
    c_contract_uv,
    frob_product,
    third_order_correct,
    target_E_log_p_from_mixture,
    residual_f,
    iproject_dirichlet_from_mixture,
)

__all__ = [
    # Gaussian info
    "make_evidence",
    "fuse_info",
    "mean_cov",
    "log_partition",
    "kl_divergence",
    "hellinger_distance",
    "bhattacharyya_coefficient",
    "fisher_information",
    "natural_gradient",
    "marginalize",
    "condition",
    "product_of_experts",
    # Dirichlet
    "dirichlet_log_partition",
    "psi_potential",
    "g_fisher",
    "c_contract_uv",
    "frob_product",
    "third_order_correct",
    "target_E_log_p_from_mixture",
    "residual_f",
    "iproject_dirichlet_from_mixture",
    # Information distances
    "hellinger_sq_expfam",
    "hellinger_expfam",
    "hellinger_gaussian",
    "fisher_rao_gaussian_1d",
    "fisher_rao_student_t",
    "fisher_rao_student_t_vec",
    "fisher_rao_spd",
    "product_distance",
    "product_distance_weighted",
    "gaussian_kl",
    "gaussian_kl_symmetric",
    "wishart_bregman",
    "bhattacharyya_coefficient_gaussian",
    "bhattacharyya_distance_gaussian",
    # ICP
    "ICPResult",
    "best_fit_se3",
    "icp_3d",
    "icp_information_weight",
    "icp_covariance_tangent",
    "transport_covariance_to_frame",
    "N_MIN_SE3_DOF",
    "K_SIGMOID",
    # Gaussian Frobenius
    "gaussian_frobenius_correction",
    # vMF geometry
    "A_d",
    "A_d_inverse_series",
    "vmf_make_evidence",
    "vmf_mean_param",
    "vmf_barycenter",
    "vmf_fisher_rao_distance",
    "vmf_third_order_correction",
    "vmf_hellinger_distance",
    # Multi-modal fusion
    "laser_2d_to_3d_constraint",
    "fuse_laser_rgbd",
    "fuse_multimodal_3d",
    "spatial_association_weight",
    # GPU point cloud processing
    "GPUPointCloudProcessor",
    "is_gpu_available",
    "voxel_filter_gpu",
    "icp_gpu",
    # OpReport
    "OpReport",
]
