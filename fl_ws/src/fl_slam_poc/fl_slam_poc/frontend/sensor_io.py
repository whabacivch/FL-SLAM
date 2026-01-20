"""
Sensor I/O Layer - NO MATH.

Handles:
- Sensor subscriptions and buffering
- TF lookups
- Point cloud conversions (depth→3D, scan→3D)
- Timestamp-based data retrieval

All geometric transforms use geometry.se3 (exact operations).
"""

from typing import Optional, Tuple
import numpy as np
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from nav_msgs.msg import Odometry
from tf2_ros import TransformException
try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

from fl_slam_poc.geometry.se3 import quat_to_rotmat


def stamp_to_sec(stamp) -> float:
    """Convert ROS timestamp to seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SensorIO:
    """
    Sensor I/O manager - buffering, TF, and point cloud conversion.
    
    Pure I/O layer - NO mathematical inference.
    """
    
    def __init__(self, node: Node, config: dict):
        """
        Args:
            node: ROS node for subscriptions and logging
            config: Dict with keys: scan_topic, odom_topic, camera_topic, depth_topic,
                    camera_info_topic, enable_image, enable_depth, enable_camera_info,
                    odom_is_delta, odom_frame, base_frame, camera_frame, scan_frame,
                    tf_timeout_sec, feature_buffer_len, depth_stride
        """
        self.node = node
        self.config = config
        
        # Buffers (timestamp, data)
        self.odom_buffer = []
        self.image_buffer = []  # (timestamp, rgb_array, frame_id)
        self.depth_buffer = []  # (timestamp, depth_array, points, frame_id)
        
        # State
        self.last_pose = None
        self.depth_intrinsics = None
        self._last_msg_keys = {}
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        # Rosbag playback frequently publishes static transforms on /tf_static.
        # Use TRANSIENT_LOCAL for static TF so late-joining subscribers receive it.
        tf_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        tf_static_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            node,
            qos=tf_qos,
            static_qos=tf_static_qos,
        )
        
        # CV Bridge
        self.cv_bridge = CvBridge() if CvBridge is not None else None
        
        # Subscribe to sensors
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        """Create ROS subscriptions for all enabled sensors."""
        qos_profiles, qos_names = self._resolve_qos_profiles()
        for qos in qos_profiles:
            self.node.create_subscription(
                LaserScan, self.config["scan_topic"], self._on_scan_internal, qos)
            self.node.create_subscription(
                Odometry, self.config["odom_topic"], self._on_odom, qos)
            
            if self.config.get("enable_image", False):
                self.node.create_subscription(
                    Image, self.config["camera_topic"], self._on_image, qos)
            
            if self.config.get("enable_depth", False):
                self.node.create_subscription(
                    Image, self.config["depth_topic"], self._on_depth, qos)
            
            if self.config.get("enable_camera_info", False):
                self.node.create_subscription(
                    CameraInfo, self.config["camera_info_topic"], self._on_camera_info, qos)
        
        self.node.get_logger().info(
            f"SensorIO subscribed to {self.config['scan_topic']} and {self.config['odom_topic']} "
            f"with QoS reliability: {', '.join(qos_names)}"
        )

    def _resolve_qos_profiles(self):
        """
        Resolve sensor QoS profiles from config.

        Supported values:
          - reliable
          - best_effort
          - system_default
          - both (subscribe twice: RELIABLE + BEST_EFFORT)
        """
        reliability = str(self.config.get("sensor_qos_reliability", "reliable")).lower()
        rel_map = {
            "reliable": ReliabilityPolicy.RELIABLE,
            "best_effort": ReliabilityPolicy.BEST_EFFORT,
            "system_default": ReliabilityPolicy.SYSTEM_DEFAULT,
        }
        if reliability == "both":
            rels = [ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT]
            names = ["reliable", "best_effort"]
        elif reliability in rel_map:
            rels = [rel_map[reliability]]
            names = [reliability]
        else:
            rels = [ReliabilityPolicy.RELIABLE]
            names = ["reliable"]
        
        profiles = [
            QoSProfile(
                reliability=rel,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            for rel in rels
        ]
        
        return profiles, names

    def _is_duplicate(self, key: str, stamp, frame_id: str) -> bool:
        """Prevent double-processing when subscribing with multiple QoS profiles."""
        if stamp is None:
            return False
        stamp_key = (stamp.sec, stamp.nanosec, frame_id or "")
        if self._last_msg_keys.get(key) == stamp_key:
            return True
        self._last_msg_keys[key] = stamp_key
        return False
    
    def set_scan_callback(self, callback):
        """Set external callback for scan processing."""
        self._scan_callback = callback

    def set_odom_callback(self, callback):
        """Set external callback for odom processing."""
        self._odom_callback = callback

    def set_image_callback(self, callback):
        """Set external callback for image processing."""
        self._image_callback = callback

    def set_depth_callback(self, callback):
        """Set external callback for depth processing."""
        self._depth_callback = callback
    
    def _on_scan_internal(self, msg: LaserScan):
        """Internal scan handler - calls external callback if set."""
        if self._is_duplicate("scan", msg.header.stamp, msg.header.frame_id):
            return
        # Debug: Log first scan received
        if not hasattr(self, '_first_scan_logged'):
            self._first_scan_logged = True
            self.node.get_logger().info(
                f"SensorIO: First scan received, frame_id={msg.header.frame_id}, "
                f"ranges={len(msg.ranges)}, last_pose={'SET' if self.last_pose is not None else 'NONE'}"
            )
        
        if hasattr(self, '_scan_callback'):
            self._scan_callback(msg)
    
    def _on_odom(self, msg: Odometry):
        """Buffer odometry (pose only)."""
        if self._is_duplicate("odom", msg.header.stamp, msg.header.frame_id):
            return
        stamp = stamp_to_sec(msg.header.stamp)
        
        # Debug logging for first odom message
        if self.last_pose is None:
            self.node.get_logger().info(
                f"SensorIO: First odom received at stamp {stamp:.3f}, frame_id={msg.header.frame_id}"
            )
        
        if self.config.get("odom_is_delta", False):
            # Accumulate deltas (for sim_world_node delta odom)
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            delta = np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z], dtype=float)
            
            if self.last_pose is None:
                self.last_pose = np.zeros(6, dtype=float)
            self.last_pose[:3] += delta[:3]
            # Simplified rotation accumulation (proper SE(3) composition in operators)
            self.last_pose[3:6] += delta[3:6]
            pose = self.last_pose.copy()
        else:
            # Absolute pose - convert quaternion to rotation vector
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            
            # Use geometry.se3 for exact conversion
            R = quat_to_rotmat(ori.x, ori.y, ori.z, ori.w)
            from fl_slam_poc.geometry.se3 import rotmat_to_rotvec
            rotvec = rotmat_to_rotvec(R)
            
            pose = np.array([pos.x, pos.y, pos.z, rotvec[0], rotvec[1], rotvec[2]], dtype=float)
            self.last_pose = pose.copy()
        
        # Buffer management
        buffer_len = self.config.get("feature_buffer_len", 10)
        self.odom_buffer.append((stamp, pose))
        if len(self.odom_buffer) > buffer_len:
            self.odom_buffer.pop(0)

        if hasattr(self, "_odom_callback"):
            self._odom_callback(msg)
    
    def _on_image(self, msg: Image):
        """Buffer RGB image array for RGB-D evidence extraction."""
        if self._is_duplicate("image", msg.header.stamp, msg.header.frame_id):
            return
        if self.cv_bridge is None:
            return
        
        try:
            # Convert to numpy array (RGB8 format)
            rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            rgb = np.asarray(rgb, dtype=np.uint8)
        except Exception as e:
            self.node.get_logger().warn(f"RGB conversion failed: {e}", throttle_duration_sec=5.0)
            return
        
        stamp = stamp_to_sec(msg.header.stamp)
        frame_id = msg.header.frame_id or self.config.get("camera_frame", "camera_link")
        self.image_buffer.append((stamp, rgb, frame_id))
        
        buffer_len = self.config.get("feature_buffer_len", 10)
        if len(self.image_buffer) > buffer_len:
            self.image_buffer.pop(0)

        if hasattr(self, "_image_callback"):
            self._image_callback(msg)
    
    def _on_depth(self, msg: Image):
        """Buffer depth array and 3D points for RGB-D evidence extraction."""
        if self._is_duplicate("depth", msg.header.stamp, msg.header.frame_id):
            return
        if self.cv_bridge is None or self.depth_intrinsics is None:
            return
        
        try:
            depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth = np.asarray(depth, dtype=np.float32)
        except Exception as e:
            self.node.get_logger().warn(f"Depth conversion failed: {e}", throttle_duration_sec=5.0)
            return
        
        # Still compute points for legacy descriptor/ICP use
        points = self._depth_to_points(depth, msg.header)
        if points is None:
            return
        
        stamp = stamp_to_sec(msg.header.stamp)
        frame_id = msg.header.frame_id or self.config.get("camera_frame", "camera_link")
        # Store depth array AND points (depth needed for normals, points for legacy)
        self.depth_buffer.append((stamp, depth, points, frame_id))
        
        buffer_len = self.config.get("feature_buffer_len", 10)
        if len(self.depth_buffer) > buffer_len:
            self.depth_buffer.pop(0)

        if hasattr(self, "_depth_callback"):
            self._depth_callback(msg)
    
    def _on_camera_info(self, msg: CameraInfo):
        """Extract camera intrinsics."""
        if self._is_duplicate("camera_info", msg.header.stamp, msg.header.frame_id):
            return
        self.depth_intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy
    
    def _depth_to_points(self, depth: np.ndarray, header) -> Optional[np.ndarray]:
        """Convert depth image to 3D points in base_frame."""
        if self.depth_intrinsics is None:
            return None
        
        fx, fy, cx, cy = self.depth_intrinsics
        stride = self.config.get("depth_stride", 4)
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
        
        # Transform to base_frame
        frame_id = header.frame_id or self.config.get("camera_frame", "camera_link")
        transform = self._lookup_transform(self.config["base_frame"], frame_id, header.stamp)
        if transform is None:
            return None
        
        return self._transform_points(points_cam, transform)
    
    def scan_to_points(self, msg: LaserScan) -> Optional[np.ndarray]:
        """Convert LaserScan to 3D points in base_frame."""
        ranges = np.asarray(msg.ranges, dtype=float).reshape(-1)
        if ranges.size == 0:
            return None
        
        angles = msg.angle_min + np.arange(ranges.size, dtype=float) * msg.angle_increment
        valid = np.isfinite(ranges)
        valid &= (ranges >= float(msg.range_min))
        valid &= (ranges <= float(msg.range_max))
        
        if not np.any(valid):
            return None
        
        r = ranges[valid]
        a = angles[valid]
        x = r * np.cos(a)
        y = r * np.sin(a)
        z = np.zeros_like(x)
        points_scan = np.stack([x, y, z], axis=1)
        
        # Transform to base_frame
        frame_id = msg.header.frame_id or self.config.get("scan_frame", "base_link")
        transform = self._lookup_transform(self.config["base_frame"], frame_id, msg.header.stamp)
        if transform is None:
            return None
        
        return self._transform_points(points_scan, transform)

    def _coerce_time(self, stamp):
        """
        Coerce various stamp types into `rclpy.time.Time`.

        `tf2_ros.Buffer.lookup_transform()` expects an `rclpy.time.Time` object.
        Passing `builtin_interfaces.msg.Time` can raise `TypeError` (and crash the node)
        during rosbag playback.
        """
        from rclpy.time import Time

        if stamp is None:
            return Time()

        # Already an rclpy Time
        if isinstance(stamp, Time):
            return stamp

        # Likely builtin_interfaces.msg.Time
        try:
            return Time.from_msg(stamp)
        except Exception:
            # Fallback: "latest available"
            return Time()
    
    def _lookup_transform(self, target_frame: str, source_frame: str, stamp) -> Optional[np.ndarray]:
        """Lookup TF transform as SE(3) pose."""
        try:
            from rclpy.duration import Duration
            timeout = Duration(seconds=self.config.get("tf_timeout_sec", 0.05))
            
            query_time = self._coerce_time(stamp)
            t = self.tf_buffer.lookup_transform(
                target_frame, source_frame, query_time,
                timeout=timeout)
            
            trans = t.transform.translation
            rot = t.transform.rotation
            
            # Convert to rotation vector using geometry.se3
            R = quat_to_rotmat(rot.x, rot.y, rot.z, rot.w)
            from fl_slam_poc.geometry.se3 import rotmat_to_rotvec
            rotvec = rotmat_to_rotvec(R)
            
            return np.array([trans.x, trans.y, trans.z, rotvec[0], rotvec[1], rotvec[2]], dtype=float)
        
        except (TransformException, TypeError, ValueError) as e:
            self.node.get_logger().warn(
                f"TF lookup failed ({target_frame} ← {source_frame}): {e}",
                throttle_duration_sec=5.0)
            return None
        except Exception as e:
            self.node.get_logger().warn(
                f"TF lookup unexpected error ({target_frame} ← {source_frame}): {type(e).__name__}: {e}",
                throttle_duration_sec=5.0,
            )
            return None
    
    def _transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply SE(3) transform to points."""
        from fl_slam_poc.geometry.se3 import rotvec_to_rotmat
        
        R = rotvec_to_rotmat(transform[3:6])
        t = transform[:3]
        
        # points_new = R @ points^T + t
        return (R @ points.T).T + t
    
    def get_nearest_pose(self, stamp_sec: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get pose nearest to timestamp. Returns (pose, dt)."""
        if not self.odom_buffer:
            return None, None
        closest = min(self.odom_buffer, key=lambda x: abs(x[0] - stamp_sec))
        return closest[1], float(stamp_sec - closest[0])
    
    def get_nearest_image(self, stamp_sec: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get RGB image array nearest to timestamp. Returns (rgb_array, dt)."""
        if not self.image_buffer:
            return None, None
        closest = min(self.image_buffer, key=lambda x: abs(x[0] - stamp_sec))
        return closest[1], float(stamp_sec - closest[0])
    
    def get_nearest_depth(self, stamp_sec: float) -> Optional[Tuple[float, Optional[np.ndarray], Optional[np.ndarray], str]]:
        """Get depth data nearest to timestamp. Returns (timestamp, depth_array, points, frame_id)."""
        if not self.depth_buffer:
            return None
        return min(self.depth_buffer, key=lambda x: abs(x[0] - stamp_sec))
    
    def get_synchronized_rgbd(self, stamp_sec: float, max_dt: float = 0.05) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[str]]:
        """
        Get synchronized RGB + depth pair nearest to timestamp.
        
        Args:
            stamp_sec: Target timestamp
            max_dt: Maximum time offset for synchronization (seconds)
        
        Returns:
            (rgb_array, depth_array, dt) or (None, None, None) if no sync found
        """
        if not self.image_buffer or not self.depth_buffer:
            return None, None, None, None
        
        # Find closest depth
        depth_item = min(self.depth_buffer, key=lambda x: abs(x[0] - stamp_sec))
        depth_stamp, depth_array, _, depth_frame = depth_item
        
        # Find closest RGB to the depth timestamp
        rgb_item = min(self.image_buffer, key=lambda x: abs(x[0] - depth_stamp))
        rgb_stamp, rgb_array, _rgb_frame = rgb_item
        
        # Check sync quality
        dt_rgb_depth = abs(rgb_stamp - depth_stamp)
        if dt_rgb_depth > max_dt:
            return None, None, None, None
        
        dt = float(stamp_sec - depth_stamp)
        return rgb_array, depth_array, dt, depth_frame
