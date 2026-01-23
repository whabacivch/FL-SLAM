"""
Frontend diagnostics subpackage.

Handles:
- Status monitoring and reporting
- Sensor health tracking
- Performance metrics

Usage:
    from fl_slam_poc.frontend.diagnostics import StatusMonitor, SensorStatus
"""

from __future__ import annotations

from fl_slam_poc.frontend.diagnostics.status_monitor import StatusMonitor, SensorStatus

__all__ = [
    "StatusMonitor",
    "SensorStatus",
]
