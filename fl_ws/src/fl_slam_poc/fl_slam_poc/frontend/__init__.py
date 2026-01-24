"""
Frontend package for Golden Child SLAM v2.

Architecture:
    Rosbag (raw topics)
        │
        ▼
    Sensor Hub (gc_sensor_hub or individual nodes)
        - livox_converter:  /livox/mid360/lidar → /gc/sensors/lidar_points
        - odom_normalizer:  /odom → /gc/sensors/odom
        - imu_normalizer:   /livox/mid360/imu → /gc/sensors/imu
        - dead_end_audit:   unused topics → /gc/dead_end_status
        │
        ▼
    /gc/sensors/* (canonical topics for backend)

Submodules:
    - hub: Single-process sensor hub (gc_sensor_hub)
    - sensors: Individual converter/normalizer nodes
    - audit: Dead-end audit for unused topics
"""

# Hub
from fl_slam_poc.frontend.hub.gc_sensor_hub import main as sensor_hub_main

# Sensors
from fl_slam_poc.frontend.sensors.livox_converter import LivoxConverterNode
from fl_slam_poc.frontend.sensors.odom_normalizer import OdomNormalizerNode
from fl_slam_poc.frontend.sensors.imu_normalizer import ImuNormalizerNode

# Audit
from fl_slam_poc.frontend.audit.dead_end_audit_node import DeadEndAuditNode
from fl_slam_poc.frontend.audit.wiring_auditor import WiringAuditorNode

__all__ = [
    # Hub
    "sensor_hub_main",
    # Sensors
    "LivoxConverterNode",
    "OdomNormalizerNode",
    "ImuNormalizerNode",
    # Audit
    "DeadEndAuditNode",
    "WiringAuditorNode",
]
