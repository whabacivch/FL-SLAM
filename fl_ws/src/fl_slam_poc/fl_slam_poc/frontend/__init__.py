"""
Frontend package for FL-SLAM.

ORCHESTRATION ONLY: All mathematical operations call backend/ operators directly.

Subpackages:
- scan/: Point cloud processing, ICP, descriptors, visual features
- keyframes/: Anchor lifecycle management
- imu/: IMU buffering and segment extraction
- sensors/: Sensor I/O, RGB-D processing, converters
- loops/: Loop detection and factor generation
- diagnostics/: Status monitoring and metrics

Shared geometry:
- vmf_geometry: von Mises-Fisher geometry (frontend-specific)
"""

from __future__ import annotations

from fl_slam_poc.frontend.scan import (
    GPUPointCloudProcessor,
    is_gpu_available,
    icp_3d,
    icp_information_weight,
    icp_covariance_tangent,
    DescriptorBuilder,
)
from fl_slam_poc.frontend.keyframes import AnchorManager
from fl_slam_poc.frontend.imu import IMUBuffer, IMUSegment, IMUMeasurement
from fl_slam_poc.frontend.sensors import (
    SensorIO,
    depth_to_pointcloud,
    compute_normals_from_depth,
    rgbd_to_evidence,
    transform_evidence_to_global,
    subsample_evidence_spatially,
)
from fl_slam_poc.frontend.loops import LoopProcessor
from fl_slam_poc.frontend.diagnostics import StatusMonitor, SensorStatus

__all__ = [
    # Scan processing
    "GPUPointCloudProcessor",
    "is_gpu_available",
    "icp_3d",
    "icp_information_weight",
    "icp_covariance_tangent",
    "DescriptorBuilder",
    # Keyframes
    "AnchorManager",
    # IMU
    "IMUBuffer",
    "IMUSegment",
    "IMUMeasurement",
    # Sensors
    "SensorIO",
    "depth_to_pointcloud",
    "compute_normals_from_depth",
    "rgbd_to_evidence",
    "transform_evidence_to_global",
    "subsample_evidence_spatially",
    # Loops
    "LoopProcessor",
    # Diagnostics
    "StatusMonitor",
    "SensorStatus",
]
