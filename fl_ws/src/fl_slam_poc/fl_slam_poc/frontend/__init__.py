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

from __future__ import annotations

from importlib import import_module
from typing import Any

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

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Processing
    "SensorIO": ("fl_slam_poc.frontend.processing.sensor_io", "SensorIO"),
    "StatusMonitor": ("fl_slam_poc.frontend.processing.status_monitor", "StatusMonitor"),
    "SensorStatus": ("fl_slam_poc.frontend.processing.status_monitor", "SensorStatus"),
    # RGB-D processing
    "depth_to_pointcloud": ("fl_slam_poc.frontend.processing.rgbd_processor", "depth_to_pointcloud"),
    "compute_normals_from_depth": ("fl_slam_poc.frontend.processing.rgbd_processor", "compute_normals_from_depth"),
    "rgbd_to_evidence": ("fl_slam_poc.frontend.processing.rgbd_processor", "rgbd_to_evidence"),
    "transform_evidence_to_global": ("fl_slam_poc.frontend.processing.rgbd_processor", "transform_evidence_to_global"),
    "subsample_evidence_spatially": ("fl_slam_poc.frontend.processing.rgbd_processor", "subsample_evidence_spatially"),
    # Anchors
    "DescriptorBuilder": ("fl_slam_poc.frontend.anchors.descriptor_builder", "DescriptorBuilder"),
    "AnchorManager": ("fl_slam_poc.frontend.anchors.anchor_manager", "AnchorManager"),
    # Loops
    "LoopProcessor": ("fl_slam_poc.frontend.loops.loop_processor", "LoopProcessor"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))
