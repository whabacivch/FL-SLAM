"""
Frobenius-Legendre SLAM Frontend Node.

Data association using information-geometric distances:
- Fisher-Rao distance on Student-t predictive (NIG model)
- Product manifold distance for multi-channel descriptors
- Probabilistic domain constraints (no hard gates)

Loop Factor Convention (EXPLICIT):
    Z = T_anchor^{-1} ∘ T_current
    Backend reconstruction: T_current = T_anchor ∘ Z

Covariance Convention:
    se(3) tangent space at identity, [δx, δy, δz, δωx, δωy, δωz]
    Transported via adjoint representation.

Observability:
    Publishes /cdwm/frontend_status (JSON) with sensor connection status.
    Logs WARN when expected sensors are missing or stop arriving.
    You will KNOW if the system is running without real sensor data.

Reference: Miyamoto et al. (2024), Combe (2022-2025), Barfoot (2017)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from tf2_ros import TransformException


@dataclass
class SensorStatus:
    """Tracks sensor connection status with timestamps."""
    name: str
    topic: str
    last_received: Optional[float] = None
    message_count: int = 0
    warned_missing: bool = False
    warned_stale: bool = False
    
    def mark_received(self):
        self.last_received = time.time()
        self.message_count += 1
        self.warned_stale = False  # Reset stale warning on new data
    
    def is_connected(self, timeout_sec: float = 5.0) -> bool:
        if self.last_received is None:
            return False
        return (time.time() - self.last_received) < timeout_sec
    
    def age_sec(self) -> Optional[float]:
        if self.last_received is None:
            return None
        return time.time() - self.last_received
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "topic": self.topic,
            "connected": self.is_connected(),
            "message_count": self.message_count,
            "age_sec": self.age_sec(),
            "ever_received": self.last_received is not None,
        }

from fl_slam_poc.geometry.se3 import (
    quat_to_rotmat,
    rotmat_to_quat,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
    se3_compose,
    se3_inverse,
    se3_adjoint,
)
from fl_slam_poc.models import (
    AdaptiveParameter,
    TimeAlignmentModel,
    StochasticBirthModel,
    NIGModel,
    NIG_PRIOR_KAPPA,
    NIG_PRIOR_ALPHA,
    NIG_PRIOR_BETA,
)
from fl_slam_poc.operators import (
    third_order_correct,
    OpReport,
    icp_3d,
    icp_information_weight,
    icp_covariance_tangent,
    transport_covariance_to_frame,
    gaussian_frobenius_correction,
)
from fl_slam_poc.msg import AnchorCreate, LoopFactor


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def compute_relative_transform(T_anchor: np.ndarray, T_current: np.ndarray) -> np.ndarray:
    """Z = T_anchor^{-1} ∘ T_current"""
    return se3_compose(se3_inverse(T_anchor), T_current)


def _vec_stats(vec: np.ndarray) -> dict:
    """Statistics summary for OpReport."""
    v = np.asarray(vec, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "norm": float(np.linalg.norm(v)),
    }


@dataclass
class Anchor:
    """Anchor landmark with NIG descriptor model."""
    anchor_id: int
    stamp_sec: float
    pose: np.ndarray
    desc_model: NIGModel
    weight: float
    depth_points: np.ndarray
    frame_id: str


class Frontend(Node):
    def __init__(self):
        super().__init__("fl_frontend")
        self._declare_parameters()
        self._init_from_params()

    def _declare_parameters(self):
        """Declare all ROS parameters with defaults."""
        # Topics
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_is_delta", False)
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("enable_image", True)
        self.declare_parameter("enable_depth", True)
        self.declare_parameter("enable_camera_info", True)
        
        # Budgets
        self.declare_parameter("descriptor_bins", 60)
        self.declare_parameter("anchor_budget", 0)
        self.declare_parameter("loop_budget", 0)
        self.declare_parameter("anchor_id_offset", 0)
        
        # ICP
        self.declare_parameter("icp_max_iter_prior", 15)
        self.declare_parameter("icp_tol_prior", 1e-4)
        self.declare_parameter("icp_prior_strength", 10.0)
        self.declare_parameter("icp_n_ref", 100.0)
        self.declare_parameter("icp_sigma_mse", 0.01)
        
        # Sensor
        self.declare_parameter("depth_stride", 4)
        self.declare_parameter("feature_buffer_len", 10)
        
        # Timestamp alignment
        self.declare_parameter("alignment_sigma_prior", 0.1)
        self.declare_parameter("alignment_prior_strength", 5.0)
        self.declare_parameter("alignment_sigma_floor", 0.001)
        
        # Birth model
        self.declare_parameter("birth_intensity", 10.0)
        self.declare_parameter("scan_period", 0.1)
        self.declare_parameter("base_component_weight", 1.0)
        
        # Frames
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("scan_frame", "base_link")
        self.declare_parameter("tf_timeout_sec", 0.05)
        
        # Fisher-Rao
        self.declare_parameter("fr_distance_scale_prior", 1.0)
        self.declare_parameter("fr_scale_prior_strength", 5.0)

    def _init_from_params(self):
        """Initialize state from parameters."""
        # Topics
        scan_topic = str(self.get_parameter("scan_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        camera_topic = str(self.get_parameter("camera_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.enable_image = bool(self.get_parameter("enable_image").value)
        self.enable_depth = bool(self.get_parameter("enable_depth").value)
        self.enable_camera_info = bool(self.get_parameter("enable_camera_info").value)

        self.odom_is_delta = bool(self.get_parameter("odom_is_delta").value)
        self.anchor_id_offset = int(self.get_parameter("anchor_id_offset").value)
        self.base_weight = float(self.get_parameter("base_component_weight").value)
        
        # Frames
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.scan_frame = str(self.get_parameter("scan_frame").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        
        # ICP
        self.icp_n_ref = float(self.get_parameter("icp_n_ref").value)
        self.icp_sigma_mse = float(self.get_parameter("icp_sigma_mse").value)

        # Adaptive models
        align_prior = float(self.get_parameter("alignment_sigma_prior").value)
        align_strength = float(self.get_parameter("alignment_prior_strength").value)
        align_floor = float(self.get_parameter("alignment_sigma_floor").value)
        
        self.align_pose = TimeAlignmentModel(align_prior, align_strength, align_floor)
        self.align_image = TimeAlignmentModel(align_prior, align_strength, align_floor)
        self.align_depth = TimeAlignmentModel(align_prior, align_strength, align_floor)
        
        fr_prior = float(self.get_parameter("fr_distance_scale_prior").value)
        fr_strength = float(self.get_parameter("fr_scale_prior_strength").value)
        self.fr_distance_scale = AdaptiveParameter(fr_prior, fr_strength, floor=0.01)
        
        icp_iter_prior = int(self.get_parameter("icp_max_iter_prior").value)
        icp_tol_prior = float(self.get_parameter("icp_tol_prior").value)
        icp_strength = float(self.get_parameter("icp_prior_strength").value)
        self.icp_max_iter = AdaptiveParameter(float(icp_iter_prior), icp_strength, floor=3.0)
        self.icp_tol = AdaptiveParameter(icp_tol_prior, icp_strength, floor=1e-6)
        
        # Birth model
        birth_intensity = float(self.get_parameter("birth_intensity").value)
        scan_period = float(self.get_parameter("scan_period").value)
        self.birth_model = StochasticBirthModel(birth_intensity, scan_period)

        # Descriptor sizing
        self.descriptor_bins = int(self.get_parameter("descriptor_bins").value)
        self.image_feat_dim = 1
        self.depth_feat_dim = 2
        self.desc_dim = self.descriptor_bins + self.image_feat_dim + self.depth_feat_dim
        self.last_image_feat: Optional[np.ndarray] = None
        self.last_depth_feat: Optional[np.ndarray] = None

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 50)
        self.sub_img = None
        self.sub_depth = None
        self.sub_info = None
        if self.enable_image:
            self.sub_img = self.create_subscription(Image, camera_topic, self.on_image, 5)
        if self.enable_depth:
            self.sub_depth = self.create_subscription(Image, depth_topic, self.on_depth, 5)
        if self.enable_camera_info:
            self.sub_info = self.create_subscription(CameraInfo, camera_info_topic, self.on_camera_info, 5)

        # Publishers
        self.pub_loop = self.create_publisher(LoopFactor, "/sim/loop_factor", 10)
        self.pub_anchor = self.create_publisher(AnchorCreate, "/sim/anchor_create", 10)
        self.pub_report = self.create_publisher(String, "/cdwm/op_report", 10)
        self.pub_status = self.create_publisher(String, "/cdwm/frontend_status", 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        tf_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        tf_static_qos = QoSProfile(
            depth=100,  # Increased depth to catch late-joiner static TFs
            reliability=ReliabilityPolicy.RELIABLE,
            # Standard for /tf_static is TRANSIENT_LOCAL (late-joining subscribers receive history)
            # This matches what rosbag2 playback publishes
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, qos=tf_qos, static_qos=tf_static_qos
        )

        # State
        self.last_pose: Optional[np.ndarray] = None
        self.last_odom_stamp: Optional[float] = None
        self.depth_intrinsics: Optional[tuple] = None
        self.odom_buffer: list[tuple[float, np.ndarray]] = []
        self.image_buffer: list[tuple[float, np.ndarray]] = []
        self.depth_buffer: list[tuple[float, np.ndarray, np.ndarray]] = []
        self.anchors: list[Anchor] = []
        self.anchor_counter = 0
        self.global_desc_model: Optional[NIGModel] = None
        
        # Sensor status tracking - YOU WILL KNOW WHAT'S CONNECTED
        self.sensor_status = {
            "odom": SensorStatus("Odometry", odom_topic),
            "scan": SensorStatus("LaserScan", scan_topic),
        }
        if self.enable_image:
            self.sensor_status["image"] = SensorStatus("Camera Image", camera_topic)
        if self.enable_depth:
            self.sensor_status["depth"] = SensorStatus("Depth Image", depth_topic)
        if self.enable_camera_info:
            self.sensor_status["camera_info"] = SensorStatus("Camera Info", camera_info_topic)
        self.status_check_period = 2.0  # seconds
        self.sensor_timeout = 5.0  # seconds before warning
        self.startup_grace_period = 10.0  # seconds before first warnings
        self.node_start_time = time.time()
        
        # Status check timer
        self.status_timer = self.create_timer(self.status_check_period, self._check_sensor_status)
        
        # Log startup message
        self.get_logger().info("=" * 60)
        self.get_logger().info("FL-SLAM Frontend starting")
        self.get_logger().info(f"  Odom topic: {odom_topic} (delta={self.odom_is_delta})")
        self.get_logger().info(f"  Scan topic: {scan_topic}")
        self.get_logger().info(
            f"  Image topic: {camera_topic} ({'ENABLED' if self.enable_image else 'DISABLED'})"
        )
        self.get_logger().info(
            f"  Depth topic: {depth_topic} ({'ENABLED' if self.enable_depth else 'DISABLED'})"
        )
        self.get_logger().info(
            f"  CameraInfo topic: {camera_info_topic} ({'ENABLED' if self.enable_camera_info else 'DISABLED'})"
        )
        self.get_logger().info("Waiting for sensor data...")
        self.get_logger().info("=" * 60)

    # =========================================================================
    # Sensor Callbacks
    # =========================================================================
    
    def on_odom(self, msg: Odometry):
        self.sensor_status["odom"].mark_received()
        
        pose = msg.pose.pose
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        rotvec = rotmat_to_rotvec(quat_to_rotmat(qx, qy, qz, qw))
        stamp_sec = stamp_to_sec(msg.header.stamp)
        
        if self.odom_is_delta:
            delta = np.array([pose.position.x, pose.position.y, pose.position.z,
                             rotvec[0], rotvec[1], rotvec[2]], dtype=float)
            if self.last_pose is None:
                self.last_pose = np.zeros(6, dtype=float)
            self.last_pose = se3_compose(self.last_pose, delta)
        else:
            self.last_pose = np.array([pose.position.x, pose.position.y, pose.position.z,
                                       rotvec[0], rotvec[1], rotvec[2]], dtype=float)
        
        self.last_odom_stamp = stamp_sec
        self.odom_buffer.append((stamp_sec, self.last_pose.copy()))
        self._trim_buffer(self.odom_buffer)

    def on_image(self, msg: Image):
        self.sensor_status["image"].mark_received()
        
        if msg.encoding not in ("mono8", "rgb8", "bgr8"):
            return
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == "mono8":
            mean_val = float(np.mean(data)) / 255.0
        else:
            if data.size != msg.height * msg.width * 3:
                return
            mean_val = float(np.mean(data.reshape(msg.height, msg.width, 3))) / 255.0
        self.image_buffer.append((stamp_to_sec(msg.header.stamp), np.array([mean_val])))
        self._trim_buffer(self.image_buffer)

    def on_depth(self, msg: Image):
        self.sensor_status["depth"].mark_received()
        
        if msg.encoding not in ("32FC1", "16UC1"):
            return
        if msg.encoding == "16UC1":
            data = np.frombuffer(msg.data, dtype=np.uint16).astype(np.float32) * 0.001
        else:
            data = np.frombuffer(msg.data, dtype=np.float32)
        if data.size != msg.height * msg.width:
            return
        depth = data.reshape(msg.height, msg.width)
        valid = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid):
            return
        vals = depth[valid]
        depth_feat = np.array([float(np.mean(vals)), float(np.std(vals))], dtype=float)
        depth_points = self._depth_to_points(depth, msg.header)
        self.depth_buffer.append((stamp_to_sec(msg.header.stamp), depth_feat, depth_points))
        self._trim_buffer(self.depth_buffer)

    def on_camera_info(self, msg: CameraInfo):
        self.sensor_status["camera_info"].mark_received()
        
        if msg.k[0] <= 0.0 or msg.k[4] <= 0.0:
            return
        self.depth_intrinsics = (float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5]))

    def _trim_buffer(self, buffer: list):
        max_len = max(1, int(self.get_parameter("feature_buffer_len").value))
        if len(buffer) > max_len:
            del buffer[:len(buffer) - max_len]

    # =========================================================================
    # Sensor Status Monitoring
    # =========================================================================

    def _check_sensor_status(self):
        """Periodic check for sensor connectivity - warns loudly if sensors are missing."""
        elapsed = time.time() - self.node_start_time
        in_grace_period = elapsed < self.startup_grace_period
        
        status_dict = {
            "timestamp": time.time(),
            "elapsed_sec": elapsed,
            "in_grace_period": in_grace_period,
            "sensors": {},
            "slam_operational": False,
            "warnings": [],
        }
        
        # Check each sensor
        for key, sensor in self.sensor_status.items():
            status_dict["sensors"][key] = sensor.to_dict()
            
            if not in_grace_period:
                # Warn if sensor was never received
                if not sensor.last_received and not sensor.warned_missing:
                    sensor.warned_missing = True
                    msg = f"SENSOR MISSING: {sensor.name} on {sensor.topic} - never received!"
                    self.get_logger().warn(msg)
                    status_dict["warnings"].append(msg)
                
                # Warn if sensor went stale
                elif sensor.last_received and not sensor.is_connected(self.sensor_timeout):
                    if not sensor.warned_stale:
                        sensor.warned_stale = True
                        age = sensor.age_sec()
                        msg = f"SENSOR STALE: {sensor.name} on {sensor.topic} - last received {age:.1f}s ago"
                        self.get_logger().warn(msg)
                        status_dict["warnings"].append(msg)
        
        # Determine if SLAM is operational
        odom_ok = self.sensor_status["odom"].is_connected(self.sensor_timeout)
        scan_ok = self.sensor_status["scan"].is_connected(self.sensor_timeout)
        depth_sensor = self.sensor_status.get("depth")
        depth_ok = depth_sensor.is_connected(self.sensor_timeout) if depth_sensor else False
        
        # We can run scan-only loop factors (2D points in base frame).
        slam_operational = odom_ok and scan_ok
        status_dict["slam_operational"] = slam_operational
        status_dict["loop_source"] = "depth" if (self.enable_depth and depth_ok) else "scan"
        
        # Summary status
        status_dict["summary"] = {
            "odom": "OK" if odom_ok else "MISSING",
            "scan": "OK" if scan_ok else "MISSING", 
            "depth": ("OK" if depth_ok else "MISSING") if self.enable_depth else "DISABLED",
            "image": ("OK" if self.sensor_status.get("image") and self.sensor_status["image"].is_connected(self.sensor_timeout) else "MISSING") if self.enable_image else "DISABLED",
            "camera_info": ("OK" if self.sensor_status.get("camera_info") and self.sensor_status["camera_info"].is_connected(self.sensor_timeout) else "MISSING") if self.enable_camera_info else "DISABLED",
            "anchors_created": self.anchor_counter,
            "anchors_active": len(self.anchors),
        }
        
        # Log periodic status (every check when not operational, less frequently when operational)
        if not slam_operational and not in_grace_period:
            missing = [k for k, v in [("odom", odom_ok), ("scan", scan_ok)] if not v]
            self.get_logger().warn(
                f"SLAM NOT OPERATIONAL - missing: {missing}. "
                f"Anchors: {self.anchor_counter} created, {len(self.anchors)} active. "
                "Connect real sensors or a simulator."
            )
        elif slam_operational and self.anchor_counter == 0 and elapsed > 30.0:
            self.get_logger().info(
                "Sensors connected but no anchors created yet. "
                "Check if depth data is being processed correctly."
            )
        
        # Publish status for external monitoring
        msg = String()
        msg.data = json.dumps(status_dict)
        self.pub_status.publish(msg)

    # =========================================================================
    # Main Processing
    # =========================================================================

    def on_scan(self, msg: LaserScan):
        self.sensor_status["scan"].mark_received()

        # Debug: count scans
        if not hasattr(self, '_scan_count'):
            self._scan_count = 0
        self._scan_count += 1
        if self._scan_count <= 5 or self._scan_count % 20 == 0:
            self.get_logger().info(f"on_scan called: scan #{self._scan_count}")

        if self.last_pose is None:
            if not hasattr(self, '_warn_no_pose_count'):
                self._warn_no_pose_count = 0
            if self._warn_no_pose_count < 3:
                self._warn_no_pose_count += 1
                self.get_logger().warn("on_scan: last_pose is None, waiting for odom")
            return

        scan_stamp = stamp_to_sec(msg.header.stamp)

        # Get aligned data with probabilistic weights
        pose, pose_dt = self._get_nearest(self.odom_buffer, scan_stamp)
        if pose is None:
            if not hasattr(self, '_warn_no_odom_count'):
                self._warn_no_odom_count = 0
            if self._warn_no_odom_count < 3:
                self._warn_no_odom_count += 1
                self.get_logger().warn(f"on_scan: no odom in buffer for scan at t={scan_stamp:.3f}, buffer size={len(self.odom_buffer)}")
            return
        self.last_pose = pose.copy()
        self.align_pose.update(pose_dt)
        pose_weight = self.align_pose.weight(pose_dt)

        if self.enable_image:
            image_feat, image_dt = self._get_nearest_feature(self.image_buffer, scan_stamp)
            if image_dt is not None:
                self.align_image.update(image_dt)
            image_weight = self.align_image.weight(image_dt)
        else:
            image_feat, image_dt = None, None
            image_weight = 1.0

        if self.enable_depth:
            depth_item = self._get_nearest_depth(scan_stamp)
            depth_feat = depth_item[1] if depth_item else None
            depth_points = depth_item[2] if depth_item else None
            depth_dt = depth_item[0] - scan_stamp if depth_item else None
            if depth_dt is not None:
                self.align_depth.update(depth_dt)
            depth_weight = self.align_depth.weight(depth_dt)
        else:
            depth_feat, depth_points, depth_dt = None, None, None
            depth_weight = 1.0

        scan_points = self._scan_to_points(msg)
        if self._scan_count <= 10:
            self.get_logger().info(f"Scan #{self._scan_count}: scan_points={'OK' if scan_points is not None else 'None'}, {len(scan_points) if scan_points is not None else 0} points")
        points = depth_points if depth_points is not None else scan_points
        point_source = "depth" if depth_points is not None else ("scan" if scan_points is not None else "none")

        # Build descriptor
        scan_desc = self._scan_descriptor(msg)
        desc = self._compose_descriptor(scan_desc, image_feat, depth_feat)

        obs_weight = pose_weight * image_weight * depth_weight

        if self.global_desc_model is None:
            self.global_desc_model = NIGModel.from_prior(
                mu=desc, kappa=NIG_PRIOR_KAPPA, alpha=NIG_PRIOR_ALPHA, beta=NIG_PRIOR_BETA)

        responsibilities, r_new = self._compute_responsibilities(desc)
        self._update_anchors(desc, responsibilities, r_new, msg.header.stamp, points, obs_weight)
        self.global_desc_model.update(desc, weight=obs_weight)
        self._publish_loop_factors(responsibilities, msg, points, obs_weight, point_source=point_source)

    def _get_nearest(self, buffer: list, stamp_sec: float):
        if not buffer:
            return None, None
        closest = min(buffer, key=lambda x: abs(x[0] - stamp_sec))
        return closest[1], float(stamp_sec - closest[0])

    def _get_nearest_feature(self, buffer: list, stamp_sec: float):
        if not buffer:
            return None, None
        closest = min(buffer, key=lambda x: abs(x[0] - stamp_sec))
        return closest[1], float(stamp_sec - closest[0])

    def _get_nearest_depth(self, stamp_sec: float):
        if not self.depth_buffer:
            return None
        return min(self.depth_buffer, key=lambda x: abs(x[0] - stamp_sec))

    def _depth_to_points(self, depth: np.ndarray, header) -> Optional[np.ndarray]:
        if self.depth_intrinsics is None:
            return None
        fx, fy, cx, cy = self.depth_intrinsics
        stride = max(1, int(self.get_parameter("depth_stride").value))
        h, w = depth.shape
        ys, xs = np.arange(0, h, stride, dtype=float), np.arange(0, w, stride, dtype=float)
        grid_x, grid_y = np.meshgrid(xs, ys)
        zs = depth[::stride, ::stride]
        valid = np.isfinite(zs) & (zs > 0.0)
        if not np.any(valid):
            return None
        z = zs[valid]
        x = (grid_x[valid] - cx) * z / fx
        y = (grid_y[valid] - cy) * z / fy
        points_cam = np.stack([x, y, z], axis=1)
        
        frame_id = header.frame_id or self.camera_frame
        transform = self._lookup_transform(self.base_frame, frame_id, header.stamp)
        if transform is None:
            return None
        return self._transform_points(points_cam, transform)

    def _scan_to_points(self, msg: LaserScan) -> Optional[np.ndarray]:
        """Convert LaserScan to a local point cloud in `base_frame` (z=0)."""
        ranges = np.asarray(msg.ranges, dtype=float).reshape(-1)
        if ranges.size == 0:
            if not hasattr(self, '_scan_warn_empty'):
                self._scan_warn_empty = True
                self.get_logger().warn("Scan has no ranges (empty array)")
            return None

        angles = msg.angle_min + np.arange(ranges.size, dtype=float) * msg.angle_increment
        valid = np.isfinite(ranges)
        valid &= (ranges >= float(msg.range_min))
        valid &= (ranges <= float(msg.range_max))
        if not np.any(valid):
            if not hasattr(self, '_scan_warn_invalid'):
                self._scan_warn_invalid = True
                self.get_logger().warn(
                    f"Scan has no valid points (range {msg.range_min}-{msg.range_max})")
            return None

        r = ranges[valid]
        a = angles[valid]
        x = r * np.cos(a)
        y = r * np.sin(a)
        z = np.zeros_like(x)
        points_scan = np.stack([x, y, z], axis=1)

        frame_id = msg.header.frame_id or str(self.get_parameter("scan_frame").value) or self.base_frame
        transform = self._lookup_transform(self.base_frame, frame_id, msg.header.stamp)
        if transform is None:
            # TF error already logged in _lookup_transform
            return None

        # Log success on first valid conversion
        if not hasattr(self, '_scan_success_logged'):
            self._scan_success_logged = True
            self.get_logger().info(
                f"Successfully converted scan to {len(points_scan)} points "
                f"(frame: {frame_id} -> {self.base_frame})")

        return self._transform_points(points_scan, transform)

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp):
        try:
            stamp_time = rclpy.time.Time.from_msg(stamp)
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp_time,
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec))
        except TransformException as e:
            # Fallback: try latest available transform (helps with bag playback)
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec))
            except TransformException as e2:
                # Log the actual error (throttled to avoid spam)
                if not hasattr(self, '_tf_error_count'):
                    self._tf_error_count = {}
                key = f"{target_frame}->{source_frame}"
                if self._tf_error_count.get(key, 0) < 5:
                    self._tf_error_count[key] = self._tf_error_count.get(key, 0) + 1
                    stamp_sec = stamp_time.nanoseconds / 1e9 if 'stamp_time' in locals() else 0.0
                    self.get_logger().warn(
                        f"TF lookup failed (exact + latest): {target_frame} -> {source_frame} "
                        f"at t={stamp_sec:.3f}s. Error: {str(e2)[:100]}")
                return None

    @staticmethod
    def _transform_points(points: np.ndarray, transform) -> np.ndarray:
        q = transform.transform.rotation
        t = transform.transform.translation
        R = quat_to_rotmat(q.x, q.y, q.z, q.w)
        return (R @ np.asarray(points, dtype=float).T).T + np.array([t.x, t.y, t.z])

    def _scan_descriptor(self, msg: LaserScan) -> np.ndarray:
        ranges = np.array(msg.ranges, dtype=float)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        bins = self.descriptor_bins
        idx = np.linspace(0, ranges.size, bins + 1, dtype=int)
        out = np.array([float(np.mean(ranges[idx[i]:idx[i+1]])) if idx[i] < idx[i+1] else msg.range_max
                        for i in range(bins)], dtype=float)
        if msg.range_max > 0.0:
            out = out / float(msg.range_max)
        return out

    def _compose_descriptor(
        self,
        scan_desc: np.ndarray,
        image_feat: Optional[np.ndarray],
        depth_feat: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compose a fixed-size descriptor vector.

        scan_desc: length `descriptor_bins`
        image_feat: length 1 (mean intensity) or None
        depth_feat: length 2 ([mean, std]) or None
        """
        scan_desc = np.asarray(scan_desc, dtype=float).reshape(-1)
        if scan_desc.size != self.descriptor_bins:
            # Pad/truncate defensively (should not happen in normal operation)
            scan_desc = np.resize(scan_desc, self.descriptor_bins)

        if image_feat is None:
            img = self.last_image_feat
        else:
            img = np.asarray(image_feat, dtype=float).reshape(-1)
            self.last_image_feat = img
        if img is None:
            img = np.zeros(self.image_feat_dim, dtype=float)
        else:
            img = np.resize(np.asarray(img, dtype=float).reshape(-1), self.image_feat_dim)

        if depth_feat is None:
            dep = self.last_depth_feat
        else:
            dep = np.asarray(depth_feat, dtype=float).reshape(-1)
            self.last_depth_feat = dep
        if dep is None:
            dep = np.zeros(self.depth_feat_dim, dtype=float)
        else:
            dep = np.resize(np.asarray(dep, dtype=float).reshape(-1), self.depth_feat_dim)

        return np.concatenate([scan_desc, img, dep])

    # =========================================================================
    # Data Association (Fisher-Rao)
    # =========================================================================

    def _compute_responsibilities(self, desc: np.ndarray) -> tuple[dict[int, float], float]:
        """Fisher-Rao metric for data association (proper Riemannian distance)."""
        import math
        
        if not self.anchors:
            return {}, 1.0

        obs_model = NIGModel.from_prior(
            mu=desc, kappa=NIG_PRIOR_KAPPA, alpha=NIG_PRIOR_ALPHA, beta=NIG_PRIOR_BETA)

        distances = np.array([a.desc_model.fisher_rao_distance(obs_model) for a in self.anchors])
        d_new = self.global_desc_model.fisher_rao_distance(obs_model) if self.global_desc_model else 0.0
        
        if len(distances) > 0:
            self.fr_distance_scale.update(float(np.median(distances)))

        weights = np.array([a.weight for a in self.anchors], dtype=float)
        log_weights = np.log(np.maximum(weights, 1e-12))
        scale = self.fr_distance_scale.value()
        
        log_numer = log_weights - distances / scale
        log_new = math.log(max(self.base_weight, 1e-12)) - d_new / scale

        max_log = max(np.max(log_numer) if len(log_numer) > 0 else -np.inf, log_new)
        numer = np.exp(log_numer - max_log)
        new_term = math.exp(log_new - max_log)
        denom = float(np.sum(numer) + new_term)

        if denom <= 0.0:
            return {}, 1.0

        resp = {a.anchor_id: float(n / denom) for a, n in zip(self.anchors, numer)}
        r_new = float(new_term / denom)
        
        self._publish_report(OpReport(
            name="FisherRaoAssociation",
            exact=True,
            family_in="StudentT",
            family_out="Categorical",
            closed_form=True,
            metrics={
                "n_anchors": len(self.anchors),
                "min_distance": float(np.min(distances)) if len(distances) > 0 else None,
                "max_distance": float(np.max(distances)) if len(distances) > 0 else None,
                "r_new": r_new,
            },
            notes="Fisher-Rao metric on Student-t predictive; closed-form.",
        ))
        
        return resp, r_new

    # =========================================================================
    # Anchor Management
    # =========================================================================

    def _update_anchors(self, desc: np.ndarray, responsibilities: dict[int, float],
                        r_new: float, stamp, points: Optional[np.ndarray], obs_weight: float):
        """Update anchor beliefs and stochastically create new anchors."""
        for anchor in self.anchors:
            r = responsibilities.get(anchor.anchor_id, 0.0)
            w = obs_weight * float(r)
            anchor.desc_model.update(desc, w)
            anchor.weight += w

        r_new_eff = obs_weight * float(r_new)
        self.base_weight += r_new_eff

        # Stochastic birth decision
        birth_prob = self.birth_model.birth_probability(r_new_eff)
        should_birth = self.birth_model.sample_birth(r_new_eff)

        # Debug: log birth attempts
        if not hasattr(self, '_birth_attempt_count'):
            self._birth_attempt_count = 0
        self._birth_attempt_count += 1
        if self._birth_attempt_count <= 10:
            self.get_logger().info(
                f"Birth attempt #{self._birth_attempt_count}: r_new_eff={r_new_eff:.4f}, "
                f"birth_prob={birth_prob:.4f}, should_birth={should_birth}, points={'OK' if points is not None else 'None'}")

        if should_birth and points is not None:
            try:
                if self.global_desc_model is None:
                    self.global_desc_model = NIGModel.from_prior(
                        mu=desc, kappa=NIG_PRIOR_KAPPA, alpha=NIG_PRIOR_ALPHA, beta=NIG_PRIOR_BETA)

                anchor_id = self.anchor_id_offset + self.anchor_counter
                self.anchors.append(Anchor(
                    anchor_id=anchor_id, stamp_sec=stamp_to_sec(stamp), pose=self.last_pose.copy(),
                    desc_model=self.global_desc_model.copy(), weight=r_new_eff,
                    depth_points=points.copy(), frame_id=self.odom_frame))
                self.anchors[-1].desc_model.update(desc, r_new_eff)
                self.anchor_counter += 1
                self._publish_anchor_create(anchor_id, stamp)

                self.get_logger().info(f"✓ Anchor #{anchor_id} created successfully! (r_new_eff={r_new_eff:.4f})")

                self._publish_report(OpReport(
                    name="StochasticAnchorBirth",
                    exact=True,
                    family_in="Poisson",
                    family_out="Anchor",
                    closed_form=True,
                    metrics={"anchor_id": anchor_id, "r_new_eff": r_new_eff, "birth_probability": birth_prob},
                    notes="Poisson birth with intensity λ = λ₀ * r_new.",
                ))
            except Exception as e:
                self.get_logger().error(f"Failed to create anchor: {e}", throttle_duration_sec=1.0)
        elif points is None and r_new_eff > 0.1:
            self._publish_report(OpReport(
                name="AnchorBirthSkipped",
                exact=True,
                family_in="PointCloud",
                family_out="None",
                closed_form=True,
                domain_projection=True,
                metrics={"r_new_eff": r_new_eff, "reason": "point_cloud unavailable"},
            ))

        # Budget enforcement
        budget = int(self.get_parameter("anchor_budget").value)
        if budget > 0 and len(self.anchors) > budget:
            self._apply_anchor_budget(budget)

    def _apply_anchor_budget(self, budget: int):
        """KL-minimizing projection with Frobenius correction."""
        weights = {a.anchor_id: float(a.weight) for a in self.anchors}
        if sum(weights.values()) <= 0.0:
            return
        
        sorted_ids = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        selected_ids = [aid for aid, _ in sorted_ids[:budget]]
        dropped_ids = [aid for aid, _ in sorted_ids[budget:]]
        dropped_mass = sum(weights[i] for i in dropped_ids)

        # Frobenius-corrected projection
        ids = list(weights.keys())
        p = np.array([weights[i] for i in ids], dtype=float)
        total = float(np.sum(p))
        p = p / total
        
        mask = np.array([1.0 if i in selected_ids else 0.0 for i in ids], dtype=float)
        p_sel = p * mask
        sel_sum = float(np.sum(p_sel))
        if sel_sum <= 0.0:
            return
        
        q = p_sel / sel_sum
        tau = total
        alpha_before = tau * p
        alpha_after = tau * q
        delta = alpha_after - alpha_before
        delta_corr = third_order_correct(alpha_before, delta)
        alpha_corr = np.maximum(alpha_before + delta_corr, 1e-12)
        q_corr = alpha_corr / float(np.sum(alpha_corr))

        self.base_weight += dropped_mass
        id_to_anchor = {a.anchor_id: a for a in self.anchors}
        self.anchors = [id_to_anchor[i] for i in selected_ids]
        for i, anchor in enumerate(self.anchors):
            idx = ids.index(anchor.anchor_id)
            anchor.weight = total * float(q_corr[idx])

        self._publish_report(OpReport(
            name="AnchorBudgetProjection",
            exact=False,
            approximation_triggers=["BudgetTruncation"],
            family_in="DescriptorMixture",
            family_out="DescriptorMixture",
            closed_form=True,
            frobenius_applied=True,
            frobenius_operator="dirichlet_third_order",
            frobenius_delta_norm=float(np.linalg.norm(delta_corr - delta)),
            frobenius_input_stats={"alpha": _vec_stats(alpha_before), "delta": _vec_stats(delta)},
            frobenius_output_stats={"delta_corr": _vec_stats(delta_corr)},
            metrics={"dropped": len(dropped_ids), "budget": budget},
        ))

    # =========================================================================
    # Loop Factor Publishing
    # =========================================================================

    def _publish_loop_factors(self, responsibilities: dict[int, float], msg: LaserScan,
                              points: Optional[np.ndarray], obs_weight: float, point_source: str):
        if points is None:
            if self.anchors:
                self._publish_report(OpReport(
                    name="LoopFactorSkipped",
                    exact=True,
                    family_in="PointCloud",
                    family_out="None",
                    closed_form=True,
                    domain_projection=True,
                    metrics={"n_anchors": len(self.anchors), "reason": "point_cloud unavailable"},
                ))
            return
        
        if not self.anchors:
            return

        weights = dict(responsibilities)
        budget = int(self.get_parameter("loop_budget").value)

        truncation_applied = False
        if budget > 0 and len(weights) > budget:
            truncation_applied = True
            sorted_ids = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            selected_ids = [aid for aid, _ in sorted_ids[:budget]]
            dropped_ids = [aid for aid, _ in sorted_ids[budget:]]

            total_mass = float(sum(weights.values()))
            sel_mass = float(sum(weights[aid] for aid in selected_ids))
            dropped_mass = float(sum(weights[aid] for aid in dropped_ids))

            # Preserve total probability mass (no silent drops): renormalize selected weights.
            renorm_factor = (total_mass / sel_mass) if sel_mass > 0.0 else 0.0
            weights = {aid: float(weights[aid]) * renorm_factor for aid in selected_ids}

            self._publish_report(OpReport(
                name="LoopBudgetTruncation",
                exact=False,
                approximation_triggers=["BudgetTruncation"],
                family_in="Categorical",
                family_out="Categorical",
                closed_form=True,
                frobenius_applied=False,
                frobenius_required=False,
                metrics={
                    "budget": budget,
                    "n_total": len(sorted_ids),
                    "n_selected": len(selected_ids),
                    "dropped_mass": dropped_mass,
                    "selected_mass_before": sel_mass,
                    "total_mass_before": total_mass,
                    "renorm_factor": renorm_factor,
                },
                notes="Compute budget enforced by truncating low-responsibility anchors; "
                      "selected weights renormalized to preserve total mass.",
            ))

        for anchor_id, weight in weights.items():
            anchor = next((a for a in self.anchors if a.anchor_id == anchor_id), None)
            if anchor is None:
                continue
            
            # Compute relative transform
            init = compute_relative_transform(anchor.pose, self.last_pose)
            
            # Run ICP
            icp_result = icp_3d(
                source=points,
                target=anchor.depth_points,
                init=init,
                max_iter=int(self.icp_max_iter.value()),
                tol=self.icp_tol.value(),
            )
            
            # Update adaptive parameters
            self.icp_max_iter.update(float(icp_result.iterations))
            self.icp_tol.update(icp_result.mse)
            
            # Compute weight
            info_weight = icp_information_weight(
                icp_result.n_source, icp_result.n_target, icp_result.mse,
                n_ref=self.icp_n_ref, sigma_mse=self.icp_sigma_mse)
            
            final_weight = obs_weight * float(weight) * info_weight
            if final_weight < 1e-12:
                continue

            # Compute covariance
            cov = icp_covariance_tangent(icp_result.src_transformed, icp_result.mse)
            cov_transported = transport_covariance_to_frame(cov, anchor.pose)
            
            self._publish_loop(icp_result.transform, cov_transported, msg.header.stamp,
                               anchor_id, final_weight, icp_result, truncation_applied=truncation_applied,
                               point_source=point_source)

            _, frob_stats = gaussian_frobenius_correction(icp_result.transform)
            
            self._publish_report(OpReport(
                name="LoopFactorPublished",
                exact=False,
                approximation_triggers=["Linearization"],
                family_in="PointCloud",
                family_out="Gaussian",
                closed_form=False,
                solver_used="ICP",
                frobenius_applied=True,
                frobenius_operator="gaussian_identity_third_order",
                frobenius_delta_norm=float(frob_stats["delta_norm"]),
                frobenius_input_stats=dict(frob_stats["input_stats"]),
                frobenius_output_stats=dict(frob_stats["output_stats"]),
                metrics={
                    "anchor_id": anchor_id,
                    "weight": final_weight,
                    "mse": icp_result.mse,
                    "iterations": icp_result.iterations,
                    "converged": icp_result.converged,
                    "point_source": point_source,
                },
                notes="ICP linearization at sensor layer per Jacobian policy.",
            ))

    def _publish_loop(self, rel_pose: np.ndarray, cov: np.ndarray, stamp,
                      anchor_id: int, weight: float, icp_result, truncation_applied: bool = False,
                      point_source: str = "unknown"):
        loop = LoopFactor()
        loop.header.stamp = stamp
        loop.header.frame_id = self.odom_frame
        loop.anchor_id = int(anchor_id)
        loop.weight = float(weight)
        loop.rel_pose.position.x = float(rel_pose[0])
        loop.rel_pose.position.y = float(rel_pose[1])
        loop.rel_pose.position.z = float(rel_pose[2])
        R = rotvec_to_rotmat(rel_pose[3:6])
        qx, qy, qz, qw = rotmat_to_quat(R)
        loop.rel_pose.orientation.x = qx
        loop.rel_pose.orientation.y = qy
        loop.rel_pose.orientation.z = qz
        loop.rel_pose.orientation.w = qw
        loop.covariance = cov.reshape(-1).tolist()
        loop.approximation_triggers = ["Linearization"] + (["BudgetTruncation"] if truncation_applied else [])
        loop.solver_name = "ICP_SCAN" if point_source == "scan" else "ICP"
        loop.solver_objective = icp_result.final_objective
        loop.solver_tolerance = icp_result.tolerance
        loop.solver_iterations = icp_result.iterations
        loop.solver_max_iterations = icp_result.max_iterations
        loop.information_weight = icp_information_weight(
            icp_result.n_source, icp_result.n_target, icp_result.mse,
            n_ref=self.icp_n_ref, sigma_mse=self.icp_sigma_mse)
        self.pub_loop.publish(loop)

    def _publish_anchor_create(self, anchor_id: int, stamp):
        msg = AnchorCreate()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.anchor_id = int(anchor_id)
        self.pub_anchor.publish(msg)

    def _publish_report(self, report: OpReport):
        report.validate()
        msg = String()
        msg.data = report.to_json()
        self.pub_report.publish(msg)


def main():
    rclpy.init()
    node = Frontend()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
