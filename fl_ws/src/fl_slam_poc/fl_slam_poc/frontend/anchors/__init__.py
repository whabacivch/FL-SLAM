"""
Frontend anchor management modules for FL-SLAM.

Handles anchor lifecycle and descriptor building.
"""

from fl_slam_poc.frontend.anchors.anchor_manager import AnchorManager, Anchor
from fl_slam_poc.frontend.anchors.descriptor_builder import DescriptorBuilder

__all__ = [
    "AnchorManager",
    "Anchor",
    "DescriptorBuilder",
]
