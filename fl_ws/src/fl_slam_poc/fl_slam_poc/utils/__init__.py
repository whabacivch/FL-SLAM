"""
Utility modules for FL-SLAM - Legacy compatibility.

NOTE: StatusMonitor has moved to fl_slam_poc.frontend.processing.
This module provides backward compatibility re-exports.
"""

from fl_slam_poc.frontend.processing.status_monitor import StatusMonitor, SensorStatus

__all__ = [
    "StatusMonitor",
    "SensorStatus",
]
