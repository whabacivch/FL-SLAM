"""
Frontend helper modules for FL-SLAM.

CRITICAL: These modules are ORCHESTRATION ONLY.
All mathematical operations MUST call operators/ and models/ directly.
NO math duplication, NO heuristic thresholds, NO approximations here.

This is pure I/O and data wrangling.
"""

from fl_slam_poc.frontend.sensor_io import SensorIO
from fl_slam_poc.frontend.descriptor_builder import DescriptorBuilder
from fl_slam_poc.frontend.anchor_manager import AnchorManager
from fl_slam_poc.frontend.loop_processor import LoopProcessor
from fl_slam_poc.frontend.rgbd_processor import (
    depth_to_pointcloud,
    compute_normals_from_depth,
    rgbd_to_evidence,
    transform_evidence_to_global,
    subsample_evidence_spatially,
)

__all__ = [
    "SensorIO",
    "DescriptorBuilder",
    "AnchorManager",
    "LoopProcessor",
    # RGB-D processing
    "depth_to_pointcloud",
    "compute_normals_from_depth",
    "rgbd_to_evidence",
    "transform_evidence_to_global",
    "subsample_evidence_spatially",
]
