"""
Backend fusion operators for FL-SLAM.

Information-geometric fusion in closed form.
"""

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
from fl_slam_poc.backend.fusion.gaussian_geom import gaussian_frobenius_correction
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
from fl_slam_poc.backend.fusion.multimodal_fusion import (
    laser_2d_to_3d_constraint,
    fuse_laser_rgbd,
    fuse_multimodal_3d,
    spatial_association_weight,
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
    # Gaussian geom
    "gaussian_frobenius_correction",
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
    # Multi-modal fusion
    "laser_2d_to_3d_constraint",
    "fuse_laser_rgbd",
    "fuse_multimodal_3d",
    "spatial_association_weight",
]
