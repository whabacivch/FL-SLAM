"""
Frontend package for Geometric Compositional SLAM v2.

Architecture:
    Rosbag → gc_sensor_hub (pointcloud_passthrough)
           → /gc/sensors/* (canonical topics for backend)

ROS-dependent nodes (rclpy) are not imported here so that backend/pipeline can be
imported without a ROS environment (e.g. preflight in run_and_evaluate_gc.sh).
Use submodules directly: fl_slam_poc.frontend.hub.gc_sensor_hub, etc.
"""

__all__ = []
