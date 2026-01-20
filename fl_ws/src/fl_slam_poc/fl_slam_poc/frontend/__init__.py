"""
Frontend package for FL-SLAM.

CRITICAL: These modules are ORCHESTRATION ONLY.
All mathematical operations MUST call backend/ operators directly.
NO math duplication, NO heuristic thresholds, NO approximations here.

This is pure I/O and data wrangling.

Subpackages:
- processing/: Sensor I/O, RGB-D processing, status monitoring
- loops/: Loop detection, ICP, point cloud processing
- anchors/: Anchor management, descriptor building
"""

from fl_slam_poc.frontend.processing.sensor_io import SensorIO
from fl_slam_poc.frontend.processing.rgbd_processor import (
    depth_to_pointcloud,
    compute_normals_from_depth,
    rgbd_to_evidence,
    transform_evidence_to_global,
    subsample_evidence_spatially,
)
from fl_slam_poc.frontend.processing.status_monitor import StatusMonitor, SensorStatus
from fl_slam_poc.frontend.anchors.descriptor_builder import DescriptorBuilder
from fl_slam_poc.frontend.anchors.anchor_manager import AnchorManager
from fl_slam_poc.frontend.loops.loop_processor import LoopProcessor

__all__ = [
    # Processing
    "SensorIO",
    "StatusMonitor",
    "SensorStatus",
    # RGB-D processing
    "depth_to_pointcloud",
    "compute_normals_from_depth",
    "rgbd_to_evidence",
    "transform_evidence_to_global",
    "subsample_evidence_spatially",
    # Anchors
    "DescriptorBuilder",
    "AnchorManager",
    # Loops
    "LoopProcessor",
]
