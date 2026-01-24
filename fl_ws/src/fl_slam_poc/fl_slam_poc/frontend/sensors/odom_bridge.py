"""
Odometry Bridge Node (Simplified for Golden Child).

Passes through odometry messages, optionally converting frames.
Used to bridge odometry from rosbag to GC backend.

For Golden Child SLAM v2, the backend directly uses absolute odometry,
so this bridge is mainly for topic remapping and frame normalization.
"""

from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class OdomBridge(Node):
    """Simple odometry bridge for GC backend."""
    
    def __init__(self):
        super().__init__("odom_bridge")
        
        # Parameters
        self.declare_parameter("input_topic", "/odom")
        self.declare_parameter("output_topic", "/odom")
        self.declare_parameter("output_frame", "odom")
        self.declare_parameter("child_frame", "base_link")
        self.declare_parameter("qos_reliability", "reliable")
        
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)
        
        qos_reliability = str(self.get_parameter("qos_reliability").value).lower()
        if qos_reliability == "reliable":
            reliability = ReliabilityPolicy.RELIABLE
        elif qos_reliability == "best_effort":
            reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            raise ValueError(f"Unsupported qos_reliability: {qos_reliability}")

        qos = QoSProfile(
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribe with explicit QoS profile
        self.sub = self.create_subscription(
            Odometry, input_topic, self.on_odom, qos
        )

        # Publisher
        self.pub = self.create_publisher(Odometry, output_topic, qos)
        
        self.msg_count = 0
        self.get_logger().info(f"Odom bridge: {input_topic} -> {output_topic}")

    def on_odom(self, msg: Odometry):
        """Pass through odometry with frame normalization."""
        self.msg_count += 1
        
        # Normalize frames if needed
        out = Odometry()
        out.header = msg.header
        if self.output_frame:
            out.header.frame_id = self.output_frame
        out.child_frame_id = self.child_frame
        out.pose = msg.pose
        out.twist = msg.twist
        
        self.pub.publish(out)
        
        if self.msg_count == 1:
            pos = msg.pose.pose.position
            self.get_logger().info(
                f"First odom at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})"
            )


def main():
    rclpy.init()
    node = OdomBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
