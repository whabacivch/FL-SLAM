"""
FL-SLAM Operators.

All operators follow information geometry principles:
- Closed-form solutions where possible
- Explicit approximation triggers
- Frobenius correction for non-closed-form operations
- OpReport for audit compliance
"""

# Gaussian operations in information form
from fl_slam_poc.operators.gaussian_info import (
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

# Dirichlet geometry
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

# Information distances (closed-form)
from fl_slam_poc.operators.information_distances import (
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

# ICP solver
from fl_slam_poc.operators.icp import (
    ICPResult,
    best_fit_se3,
    icp_3d,
    icp_information_weight,
    icp_covariance_tangent,
    transport_covariance_to_frame,
    N_MIN_SE3_DOF,
    K_SIGMOID,
)

# Gaussian Frobenius correction (no-op for C=0)
from fl_slam_poc.operators.gaussian_geom import gaussian_frobenius_correction

# vMF geometry for directional data
from fl_slam_poc.operators.vmf_geometry import (
    A_d,
    A_d_inverse_series,
    vmf_make_evidence,
    vmf_mean_param,
    vmf_barycenter,
    vmf_fisher_rao_distance,
    vmf_third_order_correction,
    vmf_hellinger_distance,
)

# Multi-modal sensor fusion
from fl_slam_poc.operators.multimodal_fusion import (
    laser_2d_to_3d_constraint,
    fuse_laser_rgbd,
    fuse_multimodal_3d,
    spatial_association_weight,
)

# OpReport for audit
from fl_slam_poc.operators.op_report import OpReport

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
    # OpReport
    "OpReport",
]
