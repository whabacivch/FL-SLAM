"""
Common package for FL-SLAM.

Shared utilities and transforms used by both frontend and backend.

Subpackages:
- transforms/: SE(3) geometry operations
"""

from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.common import config
from fl_slam_poc.common import constants

__all__ = [
    "OpReport",
    "config",
    "constants",
]
