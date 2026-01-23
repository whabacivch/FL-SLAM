"""
Golden Child SLAM v2 Pipeline.

Main per-scan execution following spec Section 7.
All steps run every time; influence may go to ~0 smoothly. No gates.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import BeliefGaussianInfo, D_Z, HypothesisSet
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    aggregate_certificates,
)
from fl_slam_poc.backend.structures.bin_atlas import (
    BinAtlas,
    MapBinStats,
    create_fibonacci_atlas,
    compute_map_derived_stats,
    apply_forgetting,
)

# Import all operators
from fl_slam_poc.backend.operators.point_budget import (
    point_budget_resample,
    PointBudgetResult,
)
from fl_slam_poc.backend.operators.predict import (
    predict_diffusion,
    build_default_process_noise,
)
from fl_slam_poc.backend.operators.deskew import (
    deskew_ut_moment_match,
    DeskewResult,
)
from fl_slam_poc.backend.operators.binning import (
    bin_soft_assign,
    scan_bin_moment_match,
    create_bin_atlas,
    ScanBinStats,
)
from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_v2
from fl_slam_poc.backend.operators.wahba import wahba_svd
from fl_slam_poc.backend.operators.translation import translation_wls
from fl_slam_poc.backend.operators.lidar_evidence import (
    lidar_quadratic_evidence,
    MapBinStats as LidarMapBinStats,
)
from fl_slam_poc.backend.operators.fusion import (
    fusion_scale_from_certificates,
    info_fusion_additive,
)
from fl_slam_poc.backend.operators.recompose import pose_update_frobenius_recompose
from fl_slam_poc.backend.operators.map_update import pos_cov_inflation_pushforward
from fl_slam_poc.backend.operators.anchor_drift import anchor_drift_update
from fl_slam_poc.backend.operators.hypothesis import hypothesis_barycenter_projection
from fl_slam_poc.common.primitives import inv_mass, safe_normalize


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the Golden Child pipeline."""
    # Budgets (hard constants)
    K_HYP: int = constants.GC_K_HYP
    B_BINS: int = constants.GC_B_BINS
    T_SLICES: int = constants.GC_T_SLICES
    N_POINTS_CAP: int = constants.GC_N_POINTS_CAP
    
    # Epsilon constants
    eps_psd: float = constants.GC_EPS_PSD
    eps_lift: float = constants.GC_EPS_LIFT
    eps_mass: float = constants.GC_EPS_MASS
    
    # Fusion parameters
    alpha_min: float = constants.GC_ALPHA_MIN
    alpha_max: float = constants.GC_ALPHA_MAX
    kappa_scale: float = constants.GC_KAPPA_SCALE
    c0_cond: float = constants.GC_C0_COND
    
    # Excitation coupling
    c_dt: float = constants.GC_C_DT
    c_ex: float = constants.GC_C_EX
    c_frob: float = constants.GC_C_FROB
    
    # Soft assign
    tau_soft_assign: float = constants.GC_TAU_SOFT_ASSIGN
    
    # Forgetting
    forgetting_factor: float = 0.99
    
    # Measurement noise
    Sigma_meas: jnp.ndarray = None
    
    def __post_init__(self):
        if self.Sigma_meas is None:
            # Default measurement noise (3x3 isotropic)
            self.Sigma_meas = 0.01 * jnp.eye(3, dtype=jnp.float64)


# =============================================================================
# Per-Scan Pipeline Result
# =============================================================================


@dataclass
class ScanPipelineResult:
    """Result of processing a single scan for one hypothesis."""
    belief_updated: BeliefGaussianInfo
    map_increments: jnp.ndarray  # Increments to map statistics
    all_certs: List[CertBundle]
    aggregated_cert: CertBundle


# =============================================================================
# Main Pipeline Functions
# =============================================================================


def process_scan_single_hypothesis(
    belief_prev: BeliefGaussianInfo,
    raw_points: jnp.ndarray,
    raw_timestamps: jnp.ndarray,
    raw_weights: jnp.ndarray,
    scan_start_time: float,
    scan_end_time: float,
    dt_sec: float,
    Q: jnp.ndarray,
    bin_atlas: BinAtlas,
    map_stats: MapBinStats,
    config: PipelineConfig,
) -> ScanPipelineResult:
    """
    Process a single scan for one hypothesis.
    
    Follows the exact 15-step order from spec Section 7:
    1. PointBudgetResample
    2. PredictDiffusion
    3. DeskewUTMomentMatch (produces ut_cache)
    4. BinSoftAssign
    5. ScanBinMomentMatch
    6. KappaFromResultant (map and scan)
    7. WahbaSVD
    8. TranslationWLS
    9. LidarQuadraticEvidence (reuses ut_cache)
    10. FusionScaleFromCertificates
    11. InfoFusionAdditive
    12. PoseUpdateFrobeniusRecompose
    13. PoseCovInflationPushforward
    14. AnchorDriftUpdate
    
    All steps run every time. No gates.
    
    Args:
        belief_prev: Previous belief
        raw_points: Raw LiDAR points (N, 3)
        raw_timestamps: Per-point timestamps (N,)
        raw_weights: Per-point weights (N,)
        scan_start_time: Scan start timestamp
        scan_end_time: Scan end timestamp
        dt_sec: Time delta since last update
        Q: Process noise matrix (D_Z, D_Z)
        bin_atlas: Bin atlas for binning
        map_stats: Current map statistics
        config: Pipeline configuration
        
    Returns:
        ScanPipelineResult with updated belief and certificates
    """
    all_certs = []
    
    # =========================================================================
    # Step 1: PointBudgetResample
    # =========================================================================
    budget_result, budget_cert, budget_effect = point_budget_resample(
        points=raw_points,
        timestamps=raw_timestamps,
        weights=raw_weights,
        n_points_cap=config.N_POINTS_CAP,
        chart_id=belief_prev.chart_id,
        anchor_id=belief_prev.anchor_id,
    )
    all_certs.append(budget_cert)
    
    points = budget_result.points
    timestamps = budget_result.timestamps
    weights = budget_result.weights
    
    # =========================================================================
    # Step 2: PredictDiffusion
    # =========================================================================
    belief_pred, pred_cert, pred_effect = predict_diffusion(
        belief_prev=belief_prev,
        Q=Q,
        dt_sec=dt_sec,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
    )
    all_certs.append(pred_cert)
    
    # =========================================================================
    # Step 3: DeskewUTMomentMatch
    # =========================================================================
    deskew_result, deskew_cert, deskew_effect = deskew_ut_moment_match(
        belief_pred=belief_pred,
        points=points,
        timestamps=timestamps,
        weights=weights,
        scan_start_time=scan_start_time,
        scan_end_time=scan_end_time,
        T_SLICES=config.T_SLICES,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
    )
    all_certs.append(deskew_cert)
    ut_cache = deskew_result.ut_cache
    
    # Extract deskewed point data
    deskewed_points = jnp.stack([p.p_mean for p in deskew_result.deskewed_points], axis=0)
    deskewed_covs = jnp.stack([p.p_cov for p in deskew_result.deskewed_points], axis=0)
    deskewed_weights = jnp.array([p.weight for p in deskew_result.deskewed_points])
    
    # Compute point directions for binning
    point_directions = jnp.zeros_like(deskewed_points)
    for i in range(deskewed_points.shape[0]):
        point_directions = point_directions.at[i].set(
            safe_normalize(deskewed_points[i], config.eps_mass)[0]
        )
    
    # =========================================================================
    # Step 4: BinSoftAssign
    # =========================================================================
    assign_result, assign_cert, assign_effect = bin_soft_assign(
        point_directions=point_directions,
        bin_directions=bin_atlas.dirs,
        tau=config.tau_soft_assign,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(assign_cert)
    responsibilities = assign_result.responsibilities
    
    # =========================================================================
    # Step 5: ScanBinMomentMatch
    # =========================================================================
    scan_bins, scan_cert, scan_effect = scan_bin_moment_match(
        points=deskewed_points,
        point_covariances=deskewed_covs,
        weights=deskewed_weights,
        responsibilities=responsibilities,
        eps_psd=config.eps_psd,
        eps_mass=config.eps_mass,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(scan_cert)
    
    # =========================================================================
    # Step 6: KappaFromResultant (map and scan) - already done in scan_bin_moment_match
    # =========================================================================
    # Compute map derived stats including kappa
    mu_map, kappa_map, c_map, Sigma_c_map = compute_map_derived_stats(
        map_stats=map_stats,
        eps_mass=config.eps_mass,
        eps_psd=config.eps_psd,
    )
    
    # Compute scan mean directions
    mu_scan = jnp.zeros((config.B_BINS, 3), dtype=jnp.float64)
    for b in range(config.B_BINS):
        mu_scan = mu_scan.at[b].set(
            safe_normalize(scan_bins.s_dir[b], config.eps_mass)[0]
        )
    
    # =========================================================================
    # Step 7: WahbaSVD
    # =========================================================================
    # Compute weights: w_b = N[b] * kappa_map[b] * kappa_scan[b]
    wahba_weights = scan_bins.N * kappa_map * scan_bins.kappa_scan
    
    wahba_result, wahba_cert, wahba_effect = wahba_svd(
        mu_map=mu_map,
        mu_scan=mu_scan,
        weights=wahba_weights,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(wahba_cert)
    R_hat = wahba_result.R_hat
    
    # =========================================================================
    # Step 8: TranslationWLS
    # =========================================================================
    trans_result, trans_cert, trans_effect = translation_wls(
        c_map=c_map,
        Sigma_c_map=Sigma_c_map,
        p_bar_scan=scan_bins.p_bar,
        Sigma_p_scan=scan_bins.Sigma_p,
        R_hat=R_hat,
        weights=wahba_weights,
        Sigma_meas=config.Sigma_meas,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(trans_cert)
    t_hat = trans_result.t_hat
    
    # =========================================================================
    # Step 9: LidarQuadraticEvidence (reuses ut_cache)
    # =========================================================================
    # Build map bins structure for lidar evidence
    lidar_map_bins = LidarMapBinStats(
        S_dir=map_stats.S_dir,
        N_dir=map_stats.N_dir,
        N_pos=map_stats.N_pos,
        sum_p=map_stats.sum_p,
        sum_ppT=map_stats.sum_ppT,
        mu_dir=mu_map,
        kappa_map=kappa_map,
        centroid=c_map,
        Sigma_c=Sigma_c_map,
    )
    
    evidence_result, evidence_cert, evidence_effect = lidar_quadratic_evidence(
        belief_pred=belief_pred,
        scan_bins=scan_bins,
        map_bins=lidar_map_bins,
        R_hat=R_hat,
        t_hat=t_hat,
        ut_cache=ut_cache,
        c_dt=config.c_dt,
        c_ex=config.c_ex,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
    )
    all_certs.append(evidence_cert)
    
    # =========================================================================
    # Step 10: FusionScaleFromCertificates
    # =========================================================================
    fusion_scale_result, fusion_scale_cert, fusion_scale_effect = fusion_scale_from_certificates(
        cert_evidence=evidence_cert,
        cert_belief=pred_cert,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        kappa_scale=config.kappa_scale,
        c0_cond=config.c0_cond,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(fusion_scale_cert)
    alpha = fusion_scale_result.alpha
    
    # =========================================================================
    # Step 11: InfoFusionAdditive
    # =========================================================================
    belief_post, fusion_cert, fusion_effect = info_fusion_additive(
        belief_pred=belief_pred,
        L_evidence=evidence_result.L_lidar,
        h_evidence=evidence_result.h_lidar,
        alpha=alpha,
        eps_psd=config.eps_psd,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(fusion_cert)
    
    # =========================================================================
    # Step 12: PoseUpdateFrobeniusRecompose
    # =========================================================================
    total_trigger_magnitude = sum(c.total_trigger_magnitude() for c in all_certs)
    
    recompose_result, belief_recomposed, recompose_cert, recompose_effect = pose_update_frobenius_recompose(
        belief_post=belief_post,
        total_trigger_magnitude=total_trigger_magnitude,
        c_frob=config.c_frob,
        eps_lift=config.eps_lift,
    )
    all_certs.append(recompose_cert)
    
    # =========================================================================
    # Step 13: PoseCovInflationPushforward
    # =========================================================================
    map_update_result, map_update_cert, map_update_effect = pos_cov_inflation_pushforward(
        belief_post=belief_recomposed,
        scan_N=scan_bins.N,
        scan_s_dir=scan_bins.s_dir,
        scan_p_bar=scan_bins.p_bar,
        scan_Sigma_p=scan_bins.Sigma_p,
        R_hat=R_hat,
        t_hat=t_hat,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
    )
    all_certs.append(map_update_cert)
    
    # =========================================================================
    # Step 14: AnchorDriftUpdate
    # =========================================================================
    drift_result, belief_final, drift_cert, drift_effect = anchor_drift_update(
        belief=belief_recomposed,
        eps_lift=config.eps_lift,
        eps_psd=config.eps_psd,
    )
    all_certs.append(drift_cert)
    
    # =========================================================================
    # Aggregate certificates
    # =========================================================================
    aggregated_cert = aggregate_certificates(all_certs)
    
    return ScanPipelineResult(
        belief_updated=belief_final,
        map_increments=jnp.zeros(1),  # Placeholder - would need proper increment structure
        all_certs=all_certs,
        aggregated_cert=aggregated_cert,
    )


def process_hypotheses(
    hypotheses: List[BeliefGaussianInfo],
    weights: jnp.ndarray,
    config: PipelineConfig,
) -> Tuple[BeliefGaussianInfo, CertBundle, ExpectedEffect]:
    """
    Combine hypotheses via barycenter projection.
    
    Step 15 in the pipeline.
    
    Args:
        hypotheses: List of K_HYP beliefs (one per hypothesis)
        weights: Hypothesis weights (K_HYP,)
        config: Pipeline configuration
        
    Returns:
        Tuple of (combined_belief, CertBundle, ExpectedEffect)
    """
    result, cert, effect = hypothesis_barycenter_projection(
        hypotheses=hypotheses,
        weights=weights,
        K_HYP=config.K_HYP,
        HYP_WEIGHT_FLOOR=constants.GC_HYP_WEIGHT_FLOOR,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
    )
    
    return result.belief_out, cert, effect


# =============================================================================
# Runtime Manifest
# =============================================================================


@dataclass
class RuntimeManifest:
    """
    Runtime manifest per spec Section 6.
    
    Nodes must publish/log this at startup.
    """
    chart_id: str = constants.GC_CHART_ID
    
    D_Z: int = constants.GC_D_Z
    D_DESKEW: int = constants.GC_D_DESKEW
    K_HYP: int = constants.GC_K_HYP
    HYP_WEIGHT_FLOOR: float = constants.GC_HYP_WEIGHT_FLOOR
    B_BINS: int = constants.GC_B_BINS
    T_SLICES: int = constants.GC_T_SLICES
    SIGMA_POINTS: int = constants.GC_SIGMA_POINTS
    N_POINTS_CAP: int = constants.GC_N_POINTS_CAP
    
    tau_soft_assign: float = constants.GC_TAU_SOFT_ASSIGN
    
    eps_psd: float = constants.GC_EPS_PSD
    eps_lift: float = constants.GC_EPS_LIFT
    eps_mass: float = constants.GC_EPS_MASS
    eps_r: float = constants.GC_EPS_R
    eps_den: float = constants.GC_EPS_DEN
    
    alpha_min: float = constants.GC_ALPHA_MIN
    alpha_max: float = constants.GC_ALPHA_MAX
    kappa_scale: float = constants.GC_KAPPA_SCALE
    c0_cond: float = constants.GC_C0_COND
    
    c_dt: float = constants.GC_C_DT
    c_ex: float = constants.GC_C_EX
    c_frob: float = constants.GC_C_FROB
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/publishing."""
        return {
            "chart_id": self.chart_id,
            "D_Z": self.D_Z,
            "D_DESKEW": self.D_DESKEW,
            "K_HYP": self.K_HYP,
            "HYP_WEIGHT_FLOOR": self.HYP_WEIGHT_FLOOR,
            "B_BINS": self.B_BINS,
            "T_SLICES": self.T_SLICES,
            "SIGMA_POINTS": self.SIGMA_POINTS,
            "N_POINTS_CAP": self.N_POINTS_CAP,
            "tau_soft_assign": self.tau_soft_assign,
            "eps_psd": self.eps_psd,
            "eps_lift": self.eps_lift,
            "eps_mass": self.eps_mass,
            "eps_r": self.eps_r,
            "eps_den": self.eps_den,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "kappa_scale": self.kappa_scale,
            "c0_cond": self.c0_cond,
            "c_dt": self.c_dt,
            "c_ex": self.c_ex,
            "c_frob": self.c_frob,
        }
