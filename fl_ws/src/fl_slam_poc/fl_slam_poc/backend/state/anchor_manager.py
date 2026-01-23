"""
Anchor creation and state buffer management.

Handles anchor creation with timestamp alignment, state buffer queries,
and pending factor processing.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from fl_slam_poc.backend.diagnostics import publish_anchor_marker, publish_map, publish_report
from fl_slam_poc.backend.fusion.gaussian_geom import gaussian_frobenius_correction
from fl_slam_poc.backend.fusion.gaussian_info import make_evidence, mean_cov
from fl_slam_poc.common import constants
from fl_slam_poc.common.op_report import OpReport

if TYPE_CHECKING:
    from fl_slam_poc.backend.backend_node import FLBackend
    from fl_slam_poc.msg import AnchorCreate


def get_state_at_stamp(
    backend: "FLBackend",
    stamp_sec: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Get state (mu, cov) at a specific timestamp from state buffer.
    
    Args:
        backend: Backend node instance
        stamp_sec: Timestamp in seconds
        
    Returns:
        Tuple of (mu, cov, dt) where dt is the time difference from closest state
        (None if state_buffer is empty, uses current state)
    """
    if not backend.state_buffer:
        mu, cov = mean_cov(backend.L, backend.h)
        return mu, cov, None
    closest = min(backend.state_buffer, key=lambda item: abs(item[0] - stamp_sec))
    return closest[1], closest[2], float(stamp_sec - closest[0])


def create_anchor(
    backend: "FLBackend",
    msg: "AnchorCreate",
) -> None:
    """
    Create anchor with probabilistic timestamp weighting.
    
    Handles:
    - Timestamp alignment and weighting
    - Anchor storage in anchors dict
    - Keyframe-to-anchor mapping
    - Processing pending loop factors and IMU segments
    - Publishing anchor markers and map updates
    - OpReport generation
    
    Args:
        backend: Backend node instance
        msg: AnchorCreate message from frontend
    """
    backend.anchor_count += 1
    
    anchor_id = int(msg.anchor_id)
    from fl_slam_poc.common.utils import stamp_to_sec
    stamp = stamp_to_sec(msg.header.stamp)
    mu, cov, dt = get_state_at_stamp(backend, stamp)
    
    backend.get_logger().info(f"Backend received anchor {anchor_id} with {len(msg.points)} points")
    
    backend.timestamp_model.update(dt)
    timestamp_weight = backend.timestamp_model.weight(dt)
    
    # Scale covariance by inverse weight
    if timestamp_weight > 1e-6:
        cov_scaled = cov / timestamp_weight
    else:
        cov_scaled = cov * 1e6
    
    # Convert points from message
    points = (
        np.array([[p.x, p.y, p.z] for p in msg.points], dtype=float)
        if len(msg.points) > 0
        else np.empty((0, 3))
    )
    
    L_anchor, h_anchor = make_evidence(mu, cov_scaled)
    backend.anchors[anchor_id] = (
        mu.copy(), cov_scaled.copy(), L_anchor.copy(), h_anchor.copy(), points.copy()
    )
    # Map keyframe id to anchor id (keyframe ids align to anchor ids)
    backend.keyframe_to_anchor[anchor_id] = anchor_id
    publish_anchor_marker(
        backend, anchor_id, mu, backend.odom_frame, backend.pub_loop_markers
    )
    
    # Publish updated map
    publish_map(
        backend, backend.anchors, backend.dense_modules, backend.odom_frame,
        backend.pub_map, backend.PointCloud2, backend.PointField
    )
    
    _, frob_stats = gaussian_frobenius_correction(np.zeros(6, dtype=float))
    
    publish_report(backend, OpReport(
        name="AnchorCreate",
        exact=dt is None or abs(dt) < constants.TIMESTAMP_EPSILON,
        approximation_triggers=["TimestampAlignment"] if dt is not None and abs(dt) >= constants.TIMESTAMP_EPSILON else [],
        family_in="Gaussian",
        family_out="Gaussian",
        closed_form=True,
        frobenius_applied=bool(dt is not None and abs(dt) >= constants.TIMESTAMP_EPSILON),
        frobenius_operator="gaussian_identity_third_order" if (dt is not None and abs(dt) >= constants.TIMESTAMP_EPSILON) else None,
        frobenius_delta_norm=float(frob_stats["delta_norm"]) if (dt is not None and abs(dt) >= constants.TIMESTAMP_EPSILON) else None,
        frobenius_input_stats=dict(frob_stats["input_stats"]) if (dt is not None and abs(dt) >= constants.TIMESTAMP_EPSILON) else None,
        frobenius_output_stats=dict(frob_stats["output_stats"]) if (dt is not None and abs(dt) >= constants.TIMESTAMP_EPSILON) else None,
        metrics={
            "anchor_id": anchor_id,
            "dt_sec": dt,
            "timestamp_weight": timestamp_weight,
        },
        notes="Anchor with probabilistic timestamp weighting.",
    ), backend.pub_report)
    
    # Process any pending loop factors for this anchor (race condition protection)
    if anchor_id in backend.pending_loop_factors:
        pending = backend.pending_loop_factors.pop(anchor_id)
        backend.get_logger().info(
            f"Processing {len(pending)} pending loop factors for anchor {anchor_id}"
        )
        for pending_msg in pending:
            backend.on_loop(pending_msg)
    
    # Process any pending IMU segments waiting on this anchor
    if anchor_id in backend.pending_imu_factors:
        pending_imu = backend.pending_imu_factors.pop(anchor_id)
        backend.get_logger().info(
            f"Processing {len(pending_imu)} pending IMU segments for anchor {anchor_id}"
        )
        for imu_msg in pending_imu:
            backend.on_imu_segment(imu_msg)
