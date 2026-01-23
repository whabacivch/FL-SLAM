"""
Backend status monitoring and diagnostics.

Handles periodic status checks, dead-reckoning detection, and status publishing.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from rclpy.node import Node
from std_msgs.msg import String

if TYPE_CHECKING:
    from fl_slam_poc.backend.backend_node import FLBackend


def check_status(
    backend: "FLBackend",
    node_start_time: float,
    odom_count: int,
    loop_factor_count: int,
    anchor_count: int,
    imu_factor_count: int,
    last_loop_time: float | None,
    pending_loop_factors: dict,
    pending_imu_factors: dict,
    anchors: dict,
    sparse_anchors: dict,
    dense_modules: dict,
    pub_status,
) -> None:
    """
    Periodic status check - warns if running dead-reckoning only.
    
    Args:
        backend: Backend node instance
        node_start_time: Node startup timestamp
        odom_count: Number of odometry messages received
        loop_factor_count: Number of loop factors received
        anchor_count: Number of anchors created
        imu_factor_count: Number of IMU segments processed
        last_loop_time: Timestamp of last loop factor (None if none)
        pending_loop_factors: Dict of pending loop factors by anchor_id
        pending_imu_factors: Dict of pending IMU segments by anchor_id
        anchors: Dict of anchors
        sparse_anchors: Dict of sparse anchor modules
        dense_modules: Dict of dense 3D modules
        pub_status: Publisher for /cdwm/backend_status
    """
    elapsed = time.time() - node_start_time
    
    # Compute odom rate
    odom_rate = odom_count / max(elapsed, 1.0)
    
    # Check if we're getting loop factors
    receiving_loops = loop_factor_count > 0
    loops_recent = (last_loop_time is not None and 
                   (time.time() - last_loop_time) < 30.0)
    
    # Determine mode
    if not receiving_loops:
        mode = "DEAD_RECKONING"
    elif loops_recent:
        mode = "SLAM_ACTIVE"
    else:
        mode = "SLAM_STALE"
    
    # Count pending factors
    total_pending = sum(len(v) for v in pending_loop_factors.values())
    total_pending_imu = sum(len(v) for v in pending_imu_factors.values())
    
    status = {
        "timestamp": time.time(),
        "elapsed_sec": elapsed,
        "mode": mode,
        "state_dim": backend.state_dim,
        "odom_count": odom_count,
        "odom_rate_hz": round(odom_rate, 1),
        "loop_factor_count": loop_factor_count,
        "imu_factor_count": imu_factor_count,
        "anchor_count": anchor_count,
        "anchors_stored": len(anchors),
        "pending_loop_factors": total_pending,
        "pending_imu_factors": total_pending_imu,
        "last_loop_age_sec": (time.time() - last_loop_time) if last_loop_time else None,
        "last_imu_age_sec": (time.time() - backend.last_imu_time) if hasattr(backend, "last_imu_time") and backend.last_imu_time else None,
        # Dual-layer statistics
        "sparse_anchors": len(sparse_anchors),
        "dense_modules": len(dense_modules),
        "rgbd_fused_anchors": sum(1 for a in sparse_anchors.values() if a.rgbd_fused),
    }
    
    # Warn if no loop factors after startup period
    if elapsed > 15.0 and not receiving_loops and not hasattr(backend, "warned_no_loops"):
        backend.warned_no_loops = True
        backend.get_logger().warn(
            "=" * 60 + "\n"
            "BACKEND RUNNING DEAD-RECKONING ONLY\n"
            "No loop factors received from frontend.\n"
            "This means: NO SLAM, just accumulating odometry drift.\n"
            "Check: Are sensors connected? Is frontend running?\n"
            f"Stats: odom={odom_count}, anchors={anchor_count}, loops={loop_factor_count}\n"
            "=" * 60
        )
    
    # Periodic status log
    if elapsed > 10.0 and int(elapsed) % 30 == 0:
        backend.get_logger().info(
            f"Backend status: mode={mode}, odom={odom_count} ({odom_rate:.1f}Hz), "
            f"loops={loop_factor_count}, anchors={len(anchors)}"
        )
    
    # Publish status
    msg = String()
    msg.data = json.dumps(status)
    pub_status.publish(msg)
