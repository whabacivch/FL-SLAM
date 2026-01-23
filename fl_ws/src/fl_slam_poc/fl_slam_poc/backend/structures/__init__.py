"""
Data structures for Golden Child SLAM v2.

This package contains all data structures used in the inference pipeline.
"""

from fl_slam_poc.backend.structures.bin_atlas import (
    BinAtlas,
    MapBinStats,
    create_fibonacci_atlas,
)

__all__ = [
    "BinAtlas",
    "MapBinStats",
    "create_fibonacci_atlas",
]
