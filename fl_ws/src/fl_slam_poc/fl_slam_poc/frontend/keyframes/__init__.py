"""
Keyframe management subpackage.

Handles:
- Anchor lifecycle (creation, storage, retrieval)
- Keyframe selection criteria

Usage:
    from fl_slam_poc.frontend.keyframes import AnchorManager
"""

from __future__ import annotations

from fl_slam_poc.frontend.keyframes.anchor_manager import AnchorManager

__all__ = [
    "AnchorManager",
]
