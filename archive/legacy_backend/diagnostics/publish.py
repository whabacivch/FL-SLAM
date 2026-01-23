"""
Backend publishing utilities.

Handles all ROS 2 message publishing for state, map, markers, reports, and trajectory.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from fl_slam_poc.backend.fusion.gaussian_info import mean_cov
from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.common.geometry.se3_numpy import rotmat_to_quat, rotvec_to_rotmat

if TYPE_CHECKING:
    from fl_slam_poc.backend.backend_node import FLBackend


def publish_state(
    backend: "FLBackend",
    tag: str,
    L: np.ndarray,
    h: np.ndarray,
    odom_frame: str,
    pub_state,
    pub_path,
    tf_broadcaster,
    trajectory_poses: list,
    max_path_length: int,
    trajectory_file,
    last_odom_stamp: float | None,
) -> None:
    """
    Publish current state as Odometry message, TF transform, and trajectory path.
    
    Args:
        backend: Backend node instance
        tag: Update tag (e.g., "odom", "loop")
        L, h: State in information form
        odom_frame: Frame ID for odom frame
        pub_state: Publisher for /cdwm/state
        pub_path: Publisher for /cdwm/trajectory
        tf_broadcaster: TF broadcaster instance
        trajectory_poses: List of PoseStamped for path visualization
        max_path_length: Maximum path history length
        trajectory_file: Optional file handle for trajectory export
        last_odom_stamp: Last odometry timestamp for trajectory export
    """
    mu, cov = mean_cov(L, h)
    
    # Extract pose (6D) from potentially 15D state
    mu_pose = mu[:6]
    cov_pose = cov[:6, :6]  # Pose covariance only
    
    out = Odometry()
    out.header.stamp = backend.get_clock().now().to_msg()
    out.header.frame_id = odom_frame
    out.child_frame_id = "base_link"
    
    out.pose.pose.position.x = float(mu_pose[0])
    out.pose.pose.position.y = float(mu_pose[1])
    out.pose.pose.position.z = float(mu_pose[2])
    
    R = rotvec_to_rotmat(mu_pose[3:6])
    qx, qy, qz, qw = rotmat_to_quat(R)
    out.pose.pose.orientation.x = qx
    out.pose.pose.orientation.y = qy
    out.pose.pose.orientation.z = qz
    out.pose.pose.orientation.w = qw
    out.pose.covariance = cov_pose.reshape(-1).tolist()  # Always 6x6 = 36 elements
    pub_state.publish(out)
    
    # Publish TF: odom -> base_link using the same pose as /cdwm/state
    tf_msg = TransformStamped()
    tf_msg.header = out.header
    tf_msg.child_frame_id = out.child_frame_id
    tf_msg.transform.translation.x = float(mu_pose[0])
    tf_msg.transform.translation.y = float(mu_pose[1])
    tf_msg.transform.translation.z = float(mu_pose[2])
    tf_msg.transform.rotation.x = float(qx)
    tf_msg.transform.rotation.y = float(qy)
    tf_msg.transform.rotation.z = float(qz)
    tf_msg.transform.rotation.w = float(qw)
    tf_broadcaster.sendTransform(tf_msg)
    
    # Trajectory path for visualization
    pose_stamped = PoseStamped()
    pose_stamped.header = out.header
    pose_stamped.pose = out.pose.pose
    trajectory_poses.append(pose_stamped)
    
    # Trim trajectory if too long
    if len(trajectory_poses) > max_path_length:
        trajectory_poses[:] = trajectory_poses[-max_path_length:]
    
    # Publish path
    path = Path()
    path.header = out.header
    path.poses = trajectory_poses
    pub_path.publish(path)
    
    # Export trajectory to file with odometry timestamps for evaluation
    if trajectory_file and last_odom_stamp is not None and tag == "odom":
        timestamp = last_odom_stamp
        trajectory_file.write(
            f"{timestamp:.6f} {mu_pose[0]:.6f} {mu_pose[1]:.6f} {mu_pose[2]:.6f} "
            f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
        )
        trajectory_file.flush()


def publish_report(backend: "FLBackend", report: OpReport, pub_report) -> None:
    """
    Publish OpReport as JSON string.
    
    Args:
        backend: Backend node instance
        report: OpReport to publish
        pub_report: Publisher for /cdwm/op_report
    """
    try:
        report.validate()
    except ValueError as exc:
        backend.get_logger().error(f"OpReport validation failed: {exc}")
        raise
    msg = String()
    msg.data = report.to_json()
    pub_report.publish(msg)


def publish_anchor_marker(
    backend: "FLBackend",
    anchor_id: int,
    mu: np.ndarray,
    odom_frame: str,
    pub_loop_markers,
) -> None:
    """
    Publish anchor position as visualization marker.
    
    Args:
        backend: Backend node instance
        anchor_id: Anchor ID
        mu: Anchor mean (at least 3D position)
        odom_frame: Frame ID for odom frame
        pub_loop_markers: Publisher for /cdwm/loop_markers
    """
    ma = MarkerArray()
    m = Marker()
    m.header.stamp = backend.get_clock().now().to_msg()
    m.header.frame_id = odom_frame
    m.ns = "anchors"
    m.id = anchor_id
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(mu[0])
    m.pose.position.y = float(mu[1])
    m.pose.position.z = float(mu[2])
    m.scale.x = 0.08
    m.scale.y = 0.08
    m.scale.z = 0.08
    m.color.a = 0.9
    m.color.r = 0.9
    m.color.g = 0.7
    m.color.b = 0.1
    ma.markers.append(m)
    pub_loop_markers.publish(ma)


def publish_loop_marker(
    backend: "FLBackend",
    anchor_id: int,
    mu_anchor: np.ndarray,
    mu_current: np.ndarray,
    odom_frame: str,
    pub_loop_markers,
) -> None:
    """
    Publish loop closure as line marker connecting anchor and current pose.
    
    Args:
        backend: Backend node instance
        anchor_id: Anchor ID
        mu_anchor: Anchor mean position (at least 3D)
        mu_current: Current pose mean (at least 3D)
        odom_frame: Frame ID for odom frame
        pub_loop_markers: Publisher for /cdwm/loop_markers
    """
    ma = MarkerArray()
    line = Marker()
    line.header.stamp = backend.get_clock().now().to_msg()
    line.header.frame_id = odom_frame
    line.ns = "loops"
    line.id = anchor_id
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.scale.x = 0.03
    line.color.a = 0.8
    line.color.r = 0.2
    line.color.g = 0.6
    line.color.b = 0.9
    line.points = []
    
    start = Point()
    start.x = float(mu_anchor[0])
    start.y = float(mu_anchor[1])
    start.z = float(mu_anchor[2])
    end = Point()
    end.x = float(mu_current[0])
    end.y = float(mu_current[1])
    end.z = float(mu_current[2])
    line.points.append(start)
    line.points.append(end)
    ma.markers.append(line)
    pub_loop_markers.publish(ma)


def publish_map(
    backend: "FLBackend",
    anchors: dict,
    dense_modules: dict,
    odom_frame: str,
    pub_map,
    PointCloud2,
    PointField,
) -> None:
    """
    Publish dual-layer point cloud map.
    
    Two layers:
    - Sparse anchors (yellow) - laser keyframes
    - Dense modules (true color) - RGB-D modules
    
    Args:
        backend: Backend node instance
        anchors: Dict of anchor_id -> (mu, cov, L, h, points)
        dense_modules: Dict of module_id -> Dense3DModule
        odom_frame: Frame ID for odom frame
        pub_map: Publisher for /cdwm/map
        PointCloud2: PointCloud2 message class
        PointField: PointField message class
    """
    from fl_slam_poc.common.geometry.se3_numpy import rotvec_to_rotmat
    
    # Collect all points with colors
    points_with_color = []  # List of (x, y, z, r, g, b)
    anchor_point_counts = {}  # Track points per anchor for logging
    
    # Layer 1: Sparse anchor point clouds (yellow)
    for anchor_id, (mu_anchor, _, _, _, points) in anchors.items():
        if len(points) == 0:
            anchor_point_counts[anchor_id] = 0
            continue
        
        # Transform points from anchor frame to global frame
        R = rotvec_to_rotmat(mu_anchor[3:6])
        t = mu_anchor[:3]
        points_transformed = (R @ points.T).T + t
        
        # Track points for this anchor
        anchor_point_counts[anchor_id] = len(points_transformed)
        
        # Yellow color for sparse anchors
        for pt in points_transformed:
            points_with_color.append((
                float(pt[0]), float(pt[1]), float(pt[2]),
                255, 255, 0  # Yellow
            ))
    
    # Layer 2: Dense modules (true RGB color)
    for mod in dense_modules.values():
        # Get module color (clamped to [0, 255])
        rgb = np.clip(mod.color_mean * 255, 0, 255).astype(np.uint8)
        points_with_color.append((
            float(mod.mu[0]), float(mod.mu[1]), float(mod.mu[2]),
            int(rgb[0]), int(rgb[1]), int(rgb[2])
        ))
    
    if len(points_with_color) == 0:
        backend.get_logger().debug("No points to publish in map")
        return
    
    # Log with anchor breakdown
    sparse_total = sum(anchor_point_counts.values())
    backend.get_logger().info(
        f"Publishing map: {len(points_with_color)} pts total "
        f"(sparse: {sparse_total} from {len(anchor_point_counts)} anchors, "
        f"dense: {len(dense_modules)} modules)"
    )
    
    # Create PointCloud2 message with XYZRGB
    msg = PointCloud2()
    msg.header.stamp = backend.get_clock().now().to_msg()
    msg.header.frame_id = odom_frame
    msg.height = 1
    msg.width = len(points_with_color)
    msg.is_dense = True
    msg.is_bigendian = False
    
    # Define fields (XYZRGB) - Foxglove compatible format
    # Pack RGB as a single float32 for maximum compatibility
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 16  # 4 floats = 16 bytes
    msg.row_step = msg.point_step * msg.width
    
    # Pack points to bytes
    cloud_data = bytearray()
    for pt in points_with_color:
        # Pack RGB into a single float32 - Foxglove/RViz use BGR order (little-endian)
        r, g, b = int(pt[3]), int(pt[4]), int(pt[5])
        # BGR order: blue in LSB, then green, then red in MSB
        rgb_packed = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
        cloud_data.extend(struct.pack('<ffff', pt[0], pt[1], pt[2], rgb_packed))
    
    msg.data = bytes(cloud_data)
    pub_map.publish(msg)
