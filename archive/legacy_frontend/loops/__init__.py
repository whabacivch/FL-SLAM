"""
Loop detection subpackage.

Handles:
- Loop closure detection via ICP
- Loop factor generation
- Descriptor matching for candidate selection

Usage:
    from fl_slam_poc.frontend.loops import LoopProcessor
"""

from __future__ import annotations

from fl_slam_poc.frontend.loops.loop_processor import LoopProcessor

__all__ = [
    "LoopProcessor",
]
