"""
Golden Child SLAM v2 Backend Node.

Actually uses the GC operators to process LiDAR scans.
This is NOT passthrough - it runs the full 14-step pipeline.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md
"""

import json
import struct
import time
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, Imu, PointField
from std_msgs.msg import String

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import (
    BeliefGaussianInfo,
    D_Z,
    se3_identity,
    se3_from_rotvec_trans,
    se3_to_rotvec_trans,
)
from fl_slam_poc.common.certificates import CertBundle
from fl_slam_poc.backend.pipeline import (
    PipelineConfig,
    RuntimeManifest,
    process_scan_single_hypothesis,
    process_hypotheses,
    ScanPipelineResult,
)
from fl_slam_poc.backend.operators.predict import (
    predict_diffusion,
    build_default_process_noise,
)
from fl_slam_poc.backend.structures.bin_atlas import (
    BinAtlas,
    MapBinStats,
    create_fibonacci_atlas,
    create_empty_map_stats,
    apply_forgetting,
    update_map_stats,
)

from scipy.spatial.transform import Rotation


def _smooth_window_weight(dist: float, min_r: float, max_r: float, sigma: float) -> float:
    """Continuous range weighting without hard gates."""
    # Smooth window: sigmoid(dist-min_r) * sigmoid(max_r-dist)
    a = (dist - min_r) / sigma
    b = (max_r - dist) / sigma
    w_min = 1.0 / (1.0 + np.exp(-a))
    w_max = 1.0 / (1.0 + np.exp(-b))
    return float(w_min * w_max)


def parse_pointcloud2(msg: PointCloud2) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Parse PointCloud2 message to extract xyz points.
    
    Returns:
        points: (N, 3) array of xyz coordinates
        timestamps: (N,) array of per-point timestamps (or zeros if unavailable)
        weights: (N,) array of continuous point weights
    """
    # Find field offsets
    field_map = {f.name: (f.offset, f.datatype) for f in msg.fields}
    
    if 'x' not in field_map or 'y' not in field_map or 'z' not in field_map:
        return jnp.zeros((0, 3)), jnp.zeros(0), jnp.zeros(0)
    
    x_off, x_type = field_map['x']
    y_off, y_type = field_map['y']
    z_off, z_type = field_map['z']
    
    point_step = msg.point_step
    n_points = msg.width * msg.height
    data = msg.data

    # Optional time fields (from livox_converter schema)
    timebase_low_off = field_map.get("timebase_low", (None, None))[0]
    timebase_high_off = field_map.get("timebase_high", (None, None))[0]
    time_offset_off = field_map.get("time_offset", (None, None))[0]

    # Compute message timebase once if available
    timebase_sec = None
    if timebase_low_off is not None and timebase_high_off is not None and n_points > 0:
        low = struct.unpack_from('<I', data, timebase_low_off)[0]
        high = struct.unpack_from('<I', data, timebase_high_off)[0]
        timebase = (np.uint64(high) << np.uint64(32)) | np.uint64(low)
        timebase_sec = float(timebase) * 1e-9

    # Extract points
    points = []
    timestamps = []
    weights = []
    header_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    for i in range(n_points):
        base = i * point_step
        x = struct.unpack_from('<f', data, base + x_off)[0]
        y = struct.unpack_from('<f', data, base + y_off)[0]
        z = struct.unpack_from('<f', data, base + z_off)[0]

        # Keep finite points only
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue

        dist = np.sqrt(x * x + y * y + z * z)
        w = _smooth_window_weight(dist, min_r=0.5, max_r=50.0, sigma=0.25)

        # Per-point timestamps if available, otherwise header time
        if timebase_sec is not None and time_offset_off is not None:
            offset = struct.unpack_from('<I', data, base + time_offset_off)[0]
            t_sec = timebase_sec + float(np.uint64(offset)) * 1e-9
        else:
            t_sec = header_time

        points.append([x, y, z])
        timestamps.append(t_sec)
        weights.append(w)

    if len(points) == 0:
        return jnp.zeros((0, 3)), jnp.zeros(0), jnp.zeros(0)

    points_arr = jnp.array(points, dtype=jnp.float64)
    timestamps_arr = jnp.array(timestamps, dtype=jnp.float64)
    weights_arr = jnp.array(weights, dtype=jnp.float64)

    return points_arr, timestamps_arr, weights_arr


class GoldenChildBackend(Node):
    """
    Golden Child SLAM v2 Backend.
    
    Actually runs the 14-step pipeline on each LiDAR scan.
    """

    def __init__(self):
        super().__init__("gc_backend")
        
        self._declare_parameters()
        self._init_state()
        self._init_ros()
        self._publish_runtime_manifest()

        self.get_logger().info("Golden Child SLAM v2 Backend initialized - PIPELINE ENABLED")

    def _declare_parameters(self):
        """Declare ROS parameters."""
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        # Backend subscribes ONLY to /gc/sensors/* (canonical topics from sensor hub)
        self.declare_parameter("lidar_topic", "/gc/sensors/lidar_points")
        self.declare_parameter("odom_topic", "/gc/sensors/odom")
        self.declare_parameter("imu_topic", "/gc/sensors/imu")
        self.declare_parameter("trajectory_export_path", "/tmp/gc_slam_trajectory.tum")
        self.declare_parameter("status_check_period_sec", 5.0)
        self.declare_parameter("forgetting_factor", 0.99)

    def _init_state(self):
        """Initialize Golden Child state."""
        # Pipeline configuration
        self.config = PipelineConfig()
        
        # Process noise matrix (22x22 for full state)
        self.Q = build_default_process_noise()
        
        # Bin atlas for directional binning
        self.bin_atlas = create_fibonacci_atlas(self.config.B_BINS)
        
        # Map statistics (accumulates over time with forgetting)
        self.map_stats = create_empty_map_stats(self.config.B_BINS)
        self.forgetting_factor = float(self.get_parameter("forgetting_factor").value)
        
        # Initialize hypotheses with identity prior
        self.hypotheses: List[BeliefGaussianInfo] = []
        self.hyp_weights = jnp.ones(self.config.K_HYP) / self.config.K_HYP
        
        for i in range(self.config.K_HYP):
            belief = BeliefGaussianInfo.create_identity_prior(
                anchor_id=f"hyp_{i}_anchor_0",
                stamp_sec=0.0,
                prior_precision=1e-6,
            )
            self.hypotheses.append(belief)
        
        # Current best estimate from hypotheses
        self.current_belief: Optional[BeliefGaussianInfo] = self.hypotheses[0]
        
        # Odometry for initial guess / prediction
        self.last_odom_pose = None
        self.last_odom_stamp = 0.0
        
        # IMU buffer for high-rate prediction
        self.imu_buffer: List[Tuple[float, jnp.ndarray, jnp.ndarray]] = []
        self.max_imu_buffer = 200
        
        # Tracking
        self.imu_count = 0
        self.odom_count = 0
        self.scan_count = 0
        self.pipeline_runs = 0
        self.last_scan_stamp = 0.0
        self.node_start_time = time.time()
        
        # Certificate history
        self.cert_history: List[CertBundle] = []

    def _init_ros(self):
        """Initialize ROS interfaces."""
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        lidar_topic = str(self.get_parameter("lidar_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        imu_topic = str(self.get_parameter("imu_topic").value)
        
        self.sub_lidar = self.create_subscription(
            PointCloud2, lidar_topic, self.on_lidar, qos_sensor
        )
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.on_odom, qos_reliable
        )
        self.sub_imu = self.create_subscription(
            Imu, imu_topic, self.on_imu, qos_sensor
        )
        
        self.get_logger().info(f"LiDAR: {lidar_topic} (PIPELINE ACTIVE)")
        self.get_logger().info(f"Odom: {odom_topic}")
        self.get_logger().info(f"IMU: {imu_topic}")
        
        # Publishers
        self.pub_state = self.create_publisher(Odometry, "/gc/state", 10)
        self.pub_path = self.create_publisher(Path, "/gc/trajectory", 10)
        self.pub_manifest = self.create_publisher(String, "/gc/runtime_manifest", 10)
        self.pub_cert = self.create_publisher(String, "/gc/certificate", 10)
        self.pub_status = self.create_publisher(String, "/gc/status", 10)
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Trajectory export
        self.trajectory_export_path = str(self.get_parameter("trajectory_export_path").value)
        self.trajectory_file = None
        if self.trajectory_export_path:
            self.trajectory_file = open(self.trajectory_export_path, "w")
            self.trajectory_file.write("# timestamp x y z qx qy qz qw\n")
        
        self.trajectory_poses: List[PoseStamped] = []
        self.max_path_length = 1000
        
        # Status timer
        status_period = float(self.get_parameter("status_check_period_sec").value)
        self._status_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.status_timer = self.create_timer(
            status_period, self._publish_status, clock=self._status_clock
        )

    def _publish_runtime_manifest(self):
        """Publish RuntimeManifest at startup."""
        manifest = RuntimeManifest()
        manifest_dict = manifest.to_dict()
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("GOLDEN CHILD RUNTIME MANIFEST")
        self.get_logger().info("=" * 60)
        for key, value in manifest_dict.items():
            self.get_logger().info(f"  {key}: {value}")
        self.get_logger().info("=" * 60)
        
        msg = String()
        msg.data = json.dumps(manifest_dict)
        self.pub_manifest.publish(msg)

    def on_imu(self, msg: Imu):
        """Buffer IMU measurements for prediction."""
        self.imu_count += 1
        
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        gyro = jnp.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=jnp.float64)
        
        accel = jnp.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=jnp.float64)
        
        self.imu_buffer.append((stamp_sec, gyro, accel))
        
        # Keep buffer bounded
        if len(self.imu_buffer) > self.max_imu_buffer:
            self.imu_buffer.pop(0)

    def on_odom(self, msg: Odometry):
        """Store latest odometry for delta computation."""
        self.odom_count += 1
        
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        quat = [ori.x, ori.y, ori.z, ori.w]
        R = Rotation.from_quat(quat)
        rotvec = R.as_rotvec()
        
        self.last_odom_pose = se3_from_rotvec_trans(
            jnp.array(rotvec, dtype=jnp.float64),
            jnp.array([pos.x, pos.y, pos.z], dtype=jnp.float64)
        )
        self.last_odom_stamp = stamp_sec

    def on_lidar(self, msg: PointCloud2):
        """
        Process LiDAR scan through the full GC pipeline.
        
        This is where the actual SLAM happens!
        """
        self.scan_count += 1
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Parse point cloud
        points, timestamps, weights = parse_pointcloud2(msg)
        n_points = points.shape[0]

        if n_points == 0:
            # Keep pipeline total by supplying a zero-weight dummy point.
            points = jnp.zeros((1, 3), dtype=jnp.float64)
            timestamps = jnp.zeros(1, dtype=jnp.float64)
            weights = jnp.zeros(1, dtype=jnp.float64)
            n_points = 1
        
        # Compute dt since last scan
        # Derive scan bounds from per-point timestamps when available; always include header stamp.
        stamp_j = jnp.array(stamp_sec, dtype=jnp.float64)
        scan_start_time = float(jnp.minimum(stamp_j, jnp.min(timestamps)))
        scan_end_time = float(jnp.maximum(stamp_j, jnp.max(timestamps)))

        dt_raw = (
            scan_end_time - self.last_scan_stamp
            if self.last_scan_stamp > 0
            else (scan_end_time - scan_start_time)
        )
        eps_dt = np.finfo(np.float64).eps
        dt_sec = float(jnp.sqrt(jnp.array(dt_raw, dtype=jnp.float64) ** 2 + eps_dt))
        self.last_scan_stamp = scan_end_time
        
        # Apply forgetting to map stats
        self.map_stats = apply_forgetting(self.map_stats, self.forgetting_factor)
        
        # Run pipeline for each hypothesis
        results: List[ScanPipelineResult] = []
        
        try:
            for i, belief in enumerate(self.hypotheses):
                result = process_scan_single_hypothesis(
                    belief_prev=belief,
                    raw_points=points,
                    raw_timestamps=timestamps,
                    raw_weights=weights,
                    scan_start_time=scan_start_time,
                    scan_end_time=scan_end_time,
                    dt_sec=dt_sec,
                    Q=self.Q,
                    bin_atlas=self.bin_atlas,
                    map_stats=self.map_stats,
                    config=self.config,
                )
                results.append(result)
                self.hypotheses[i] = result.belief_updated
            
            # Combine hypotheses
            combined_belief, combo_cert, combo_effect = process_hypotheses(
                hypotheses=self.hypotheses,
                weights=self.hyp_weights,
                config=self.config,
            )
            
            self.current_belief = combined_belief
            self.pipeline_runs += 1
            
            # Update map statistics with weighted increments
            delta_S_dir = jnp.zeros_like(self.map_stats.S_dir)
            delta_N_dir = jnp.zeros_like(self.map_stats.N_dir)
            delta_N_pos = jnp.zeros_like(self.map_stats.N_pos)
            delta_sum_p = jnp.zeros_like(self.map_stats.sum_p)
            delta_sum_ppT = jnp.zeros_like(self.map_stats.sum_ppT)
            for i, result in enumerate(results):
                w_h = float(self.hyp_weights[i])
                delta_S_dir = delta_S_dir + w_h * result.map_increments.delta_S_dir
                delta_N_dir = delta_N_dir + w_h * result.map_increments.delta_N_dir
                delta_N_pos = delta_N_pos + w_h * result.map_increments.delta_N_pos
                delta_sum_p = delta_sum_p + w_h * result.map_increments.delta_sum_p
                delta_sum_ppT = delta_sum_ppT + w_h * result.map_increments.delta_sum_ppT
            self.map_stats = update_map_stats(
                self.map_stats,
                delta_S_dir,
                delta_N_dir,
                delta_N_pos,
                delta_sum_p,
                delta_sum_ppT,
            )

            # Store certificate
            if results:
                self.cert_history.append(results[0].aggregated_cert)
                if len(self.cert_history) > 100:
                    self.cert_history.pop(0)
            
            # Extract pose from belief and publish
            pose_6d = combined_belief.mean_world_pose()
            self._publish_state_from_pose(pose_6d, stamp_sec)
            
            if self.scan_count <= 10 or self.scan_count % 50 == 0:
                self.get_logger().info(
                    f"Scan {self.scan_count}: {n_points} pts, pipeline #{self.pipeline_runs}, "
                    f"dt={dt_sec:.3f}s"
                )
        
        except Exception as e:
            # Log error and fail fast (no silent fallbacks)
            self.get_logger().error(f"Pipeline error on scan {self.scan_count}: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            raise

    def _publish_state_from_pose(self, pose_6d: jnp.ndarray, stamp_sec: float):
        """Publish state from a 6D pose [trans, rotvec]."""
        rotvec, trans = se3_to_rotvec_trans(pose_6d)
        R = Rotation.from_rotvec(np.array(rotvec))
        quat = R.as_quat()
        
        # Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        
        odom_msg.pose.pose.position.x = float(trans[0])
        odom_msg.pose.pose.position.y = float(trans[1])
        odom_msg.pose.pose.position.z = float(trans[2])
        odom_msg.pose.pose.orientation.x = float(quat[0])
        odom_msg.pose.pose.orientation.y = float(quat[1])
        odom_msg.pose.pose.orientation.z = float(quat[2])
        odom_msg.pose.pose.orientation.w = float(quat[3])
        
        self.pub_state.publish(odom_msg)
        
        # TF
        tf_msg = TransformStamped()
        tf_msg.header = odom_msg.header
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = float(trans[0])
        tf_msg.transform.translation.y = float(trans[1])
        tf_msg.transform.translation.z = float(trans[2])
        tf_msg.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)
        
        # Trajectory
        pose_stamped = PoseStamped()
        pose_stamped.header = odom_msg.header
        pose_stamped.pose = odom_msg.pose.pose
        self.trajectory_poses.append(pose_stamped)
        
        if len(self.trajectory_poses) > self.max_path_length:
            self.trajectory_poses.pop(0)
        
        path_msg = Path()
        path_msg.header = odom_msg.header
        path_msg.poses = self.trajectory_poses
        self.pub_path.publish(path_msg)
        
        # TUM export
        if self.trajectory_file:
            self.trajectory_file.write(
                f"{stamp_sec:.9f} {trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )
            self.trajectory_file.flush()

    def _publish_status(self):
        """Publish periodic status."""
        elapsed = time.time() - self.node_start_time
        
        status = {
            "elapsed_sec": elapsed,
            "odom_count": self.odom_count,
            "scan_count": self.scan_count,
            "imu_count": self.imu_count,
            "pipeline_runs": self.pipeline_runs,
            "hypotheses": self.config.K_HYP,
            "map_bins_active": int(jnp.sum(self.map_stats.N_dir > 0)),
        }
        
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)
        
        self.get_logger().info(
            f"GC Status: odom={self.odom_count}, scans={self.scan_count}, "
            f"imu={self.imu_count}, pipeline={self.pipeline_runs}, "
            f"map_bins={status['map_bins_active']}/{self.config.B_BINS}"
        )

    def destroy_node(self):
        """Clean up."""
        if self.trajectory_file:
            self.trajectory_file.flush()
            self.trajectory_file.close()
            self.get_logger().info(f"Trajectory saved: {self.trajectory_export_path}")
        super().destroy_node()


def main():
    rclpy.init()
    node = GoldenChildBackend()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
