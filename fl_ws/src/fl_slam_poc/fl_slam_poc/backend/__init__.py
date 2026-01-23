"""
Golden Child SLAM v2 Backend.

Branch-free compositional inference backend per docs/GOLDEN_CHILD_INTERFACE_SPEC.md.

Structure:
- operators/: Branch-free operators (predict, fuse, recompose, etc.)
- structures/: Data structures (BinAtlas, UTCache)
- pipeline.py: Main pipeline functions
- backend_node.py: ROS2 node entry point

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md
"""

from fl_slam_poc.backend.pipeline import (
    PipelineConfig,
    RuntimeManifest,
    process_scan,
    process_hypotheses,
)

__all__ = [
    "PipelineConfig",
    "RuntimeManifest",
    "process_scan",
    "process_hypotheses",
]
