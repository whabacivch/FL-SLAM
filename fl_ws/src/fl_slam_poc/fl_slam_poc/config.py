"""
Configuration classes for FL-SLAM parameters.

Organizes parameters into logical groups for better maintainability.
"""

from dataclasses import dataclass
from typing import Optional
from fl_slam_poc import constants


@dataclass
class TopicConfig:
    """ROS topic configuration."""
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    camera_topic: str = "/camera/image_raw"
    depth_topic: str = "/camera/depth/image_raw"
    camera_info_topic: str = "/camera/depth/camera_info"
    
    # Output topics (computed from namespace)
    state_topic: str = "/cdwm/state"
    markers_topic: str = "/cdwm/markers"
    trajectory_topic: str = "/cdwm/trajectory"
    debug_topic: str = "/cdwm/debug"
    report_topic: str = "/cdwm/op_report"
    status_topic: str = "/cdwm/backend_status"


@dataclass
class FrameConfig:
    """TF frame configuration."""
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    camera_frame: str = "camera_link"
    scan_frame: str = "base_link"
    tf_timeout_sec: float = 0.05


@dataclass
class SensorConfig:
    """Sensor processing configuration."""
    enable_image: bool = True
    enable_depth: bool = True
    enable_camera_info: bool = True
    odom_is_delta: bool = False
    
    depth_stride: int = constants.DEPTH_STRIDE_DEFAULT
    feature_buffer_len: int = constants.FEATURE_BUFFER_MAX_LENGTH
    
    sensor_timeout: float = constants.SENSOR_TIMEOUT_DEFAULT
    startup_grace_period: float = constants.SENSOR_STARTUP_GRACE_PERIOD


@dataclass
class ICPConfig:
    """ICP solver configuration."""
    max_iter_prior: int = constants.ICP_MAX_ITER_DEFAULT
    tol_prior: float = constants.ICP_TOLERANCE_DEFAULT
    prior_strength: float = 10.0
    n_ref: float = constants.ICP_N_REF_DEFAULT
    sigma_mse: float = constants.ICP_SIGMA_MSE_DEFAULT


@dataclass
class AlignmentConfig:
    """Timestamp alignment configuration."""
    sigma_prior: float = constants.ALIGNMENT_SIGMA_PRIOR
    prior_strength: float = constants.ALIGNMENT_PRIOR_STRENGTH
    sigma_floor: float = constants.ALIGNMENT_SIGMA_FLOOR


@dataclass
class ProcessNoiseConfig:
    """Process noise model configuration."""
    trans_prior: float = constants.PROCESS_NOISE_TRANS_PRIOR
    rot_prior: float = constants.PROCESS_NOISE_ROT_PRIOR
    prior_strength: float = constants.PROCESS_NOISE_PRIOR_STRENGTH


@dataclass
class DescriptorConfig:
    """Descriptor model configuration."""
    bins: int = constants.DESCRIPTOR_BINS_DEFAULT
    
    # NIG prior parameters
    nig_kappa: float = constants.NIG_PRIOR_KAPPA
    nig_alpha: float = constants.NIG_PRIOR_ALPHA
    nig_beta: float = constants.NIG_PRIOR_BETA
    
    # Fisher-Rao distance
    fr_distance_scale_prior: float = constants.FR_DISTANCE_SCALE_PRIOR
    fr_scale_prior_strength: float = constants.FR_SCALE_PRIOR_STRENGTH


@dataclass
class BirthModelConfig:
    """Stochastic birth model configuration."""
    intensity: float = constants.BIRTH_INTENSITY_DEFAULT
    scan_period: float = constants.SCAN_PERIOD_DEFAULT
    base_component_weight: float = constants.BASE_COMPONENT_WEIGHT_DEFAULT


@dataclass
class BudgetConfig:
    """Memory/computation budget configuration."""
    anchor_budget: int = 0  # 0 = unlimited
    loop_budget: int = 0    # 0 = unlimited
    anchor_id_offset: int = 0


@dataclass
class FrontendConfig:
    """Complete frontend configuration."""
    topics: TopicConfig
    frames: FrameConfig
    sensors: SensorConfig
    icp: ICPConfig
    alignment: AlignmentConfig
    descriptor: DescriptorConfig
    birth: BirthModelConfig
    budget: BudgetConfig
    
    @classmethod
    def from_ros_node(cls, node):
        """Create configuration from ROS node parameters."""
        topics = TopicConfig(
            scan_topic=str(node.get_parameter("scan_topic").value),
            odom_topic=str(node.get_parameter("odom_topic").value),
            camera_topic=str(node.get_parameter("camera_topic").value),
            depth_topic=str(node.get_parameter("depth_topic").value),
            camera_info_topic=str(node.get_parameter("camera_info_topic").value),
        )
        
        frames = FrameConfig(
            odom_frame=str(node.get_parameter("odom_frame").value),
            base_frame=str(node.get_parameter("base_frame").value),
            camera_frame=str(node.get_parameter("camera_frame").value),
            scan_frame=str(node.get_parameter("scan_frame").value),
            tf_timeout_sec=float(node.get_parameter("tf_timeout_sec").value),
        )
        
        sensors = SensorConfig(
            enable_image=bool(node.get_parameter("enable_image").value),
            enable_depth=bool(node.get_parameter("enable_depth").value),
            enable_camera_info=bool(node.get_parameter("enable_camera_info").value),
            odom_is_delta=bool(node.get_parameter("odom_is_delta").value),
            depth_stride=int(node.get_parameter("depth_stride").value),
            feature_buffer_len=int(node.get_parameter("feature_buffer_len").value),
        )
        
        icp = ICPConfig(
            max_iter_prior=int(node.get_parameter("icp_max_iter_prior").value),
            tol_prior=float(node.get_parameter("icp_tol_prior").value),
            prior_strength=float(node.get_parameter("icp_prior_strength").value),
            n_ref=float(node.get_parameter("icp_n_ref").value),
            sigma_mse=float(node.get_parameter("icp_sigma_mse").value),
        )
        
        alignment = AlignmentConfig(
            sigma_prior=float(node.get_parameter("alignment_sigma_prior").value),
            prior_strength=float(node.get_parameter("alignment_prior_strength").value),
            sigma_floor=float(node.get_parameter("alignment_sigma_floor").value),
        )
        
        descriptor = DescriptorConfig(
            bins=int(node.get_parameter("descriptor_bins").value),
            fr_distance_scale_prior=float(node.get_parameter("fr_distance_scale_prior").value),
            fr_scale_prior_strength=float(node.get_parameter("fr_scale_prior_strength").value),
        )
        
        birth = BirthModelConfig(
            intensity=float(node.get_parameter("birth_intensity").value),
            scan_period=float(node.get_parameter("scan_period").value),
            base_component_weight=float(node.get_parameter("base_component_weight").value),
        )
        
        budget = BudgetConfig(
            anchor_budget=int(node.get_parameter("anchor_budget").value),
            loop_budget=int(node.get_parameter("loop_budget").value),
            anchor_id_offset=int(node.get_parameter("anchor_id_offset").value),
        )
        
        return cls(
            topics=topics,
            frames=frames,
            sensors=sensors,
            icp=icp,
            alignment=alignment,
            descriptor=descriptor,
            birth=birth,
            budget=budget,
        )


@dataclass
class BackendConfig:
    """Complete backend configuration."""
    topics: TopicConfig
    frames: FrameConfig
    alignment: AlignmentConfig
    process_noise: ProcessNoiseConfig
    
    @classmethod
    def from_ros_node(cls, node):
        """Create configuration from ROS node parameters."""
        topics = TopicConfig()  # Use defaults, backend doesn't customize topics
        
        frames = FrameConfig(
            odom_frame=str(node.get_parameter("odom_frame").value),
        )
        
        alignment = AlignmentConfig(
            sigma_prior=float(node.get_parameter("alignment_sigma_prior").value),
            prior_strength=float(node.get_parameter("alignment_prior_strength").value),
            sigma_floor=float(node.get_parameter("alignment_sigma_floor").value),
        )
        
        process_noise = ProcessNoiseConfig(
            trans_prior=float(node.get_parameter("process_noise_trans_prior").value),
            rot_prior=float(node.get_parameter("process_noise_rot_prior").value),
            prior_strength=float(node.get_parameter("process_noise_prior_strength").value),
        )
        
        return cls(
            topics=topics,
            frames=frames,
            alignment=alignment,
            process_noise=process_noise,
        )
