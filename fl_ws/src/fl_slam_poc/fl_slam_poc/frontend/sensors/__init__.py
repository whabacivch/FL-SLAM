"""
Sensor I/O subpackage.

Handles:
- Sensor message subscription and buffering
- TF lookups and transforms
- Camera intrinsics management
- RGB-D processing
- Livox converter and odometry bridge utilities

Usage:
    from fl_slam_poc.frontend.sensors import (
        SensorIO,
        depth_to_pointcloud,
        compute_normals_from_depth,
        rgbd_to_evidence,
    )
"""

from __future__ import annotations

from fl_slam_poc.frontend.sensors.sensor_io import SensorIO
from fl_slam_poc.frontend.sensors.rgbd_processor import (
    depth_to_pointcloud,
    compute_normals_from_depth,
    rgbd_to_evidence,
    transform_evidence_to_global,
    subsample_evidence_spatially,
)

__all__ = [
    "SensorIO",
    "depth_to_pointcloud",
    "compute_normals_from_depth",
    "rgbd_to_evidence",
    "transform_evidence_to_global",
    "subsample_evidence_spatially",
]
