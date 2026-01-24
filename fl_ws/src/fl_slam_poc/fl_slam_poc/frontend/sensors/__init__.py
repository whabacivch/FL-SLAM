"""
Sensor utility nodes for Golden Child SLAM v2.

Converters:
    - livox_converter: Converts Livox CustomMsg to PointCloud2

Normalizers:
    - odom_normalizer: Normalizes raw odometry for /gc/sensors/odom
    - imu_normalizer: Normalizes raw IMU for /gc/sensors/imu

Legacy:
    - (none)
"""

from fl_slam_poc.frontend.sensors.livox_converter import LivoxConverterNode, main as livox_main
from fl_slam_poc.frontend.sensors.odom_normalizer import OdomNormalizerNode, main as odom_normalizer_main
from fl_slam_poc.frontend.sensors.imu_normalizer import ImuNormalizerNode, main as imu_normalizer_main

__all__ = [
    # Converters
    "LivoxConverterNode",
    "livox_main",
    # Normalizers
    "OdomNormalizerNode",
    "odom_normalizer_main",
    "ImuNormalizerNode",
    "imu_normalizer_main",
]
