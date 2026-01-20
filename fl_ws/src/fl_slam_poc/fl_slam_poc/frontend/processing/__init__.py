"""
Frontend processing modules for FL-SLAM.

Handles sensor I/O, RGB-D processing, and status monitoring.
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

__all__ = [
    "SensorIO",
    "StatusMonitor",
    "SensorStatus",
    "depth_to_pointcloud",
    "compute_normals_from_depth",
    "rgbd_to_evidence",
    "transform_evidence_to_global",
    "subsample_evidence_spatially",
]
