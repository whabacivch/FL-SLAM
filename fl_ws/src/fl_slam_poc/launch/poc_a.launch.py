"""
POC-A Launch: Ground Truth Simulation with SLAM Pipeline.

This launch file runs:
- sim_world_node: Provides ground truth trajectory and noisy odometry
- tb3_odom_bridge: Converts absolute odom to delta odom for backend
- frontend_node: Processes sensor data (when available from real sim)
- backend_node: Fuses odometry and loop factors

Sensors should come from a real simulator (e.g., Gazebo) for physically
consistent data. Without sensors, the frontend will run but won't create
anchors or loop factors - this is correct behavior.

To run with sensors, either:
1. Launch Gazebo separately with appropriate sensor plugins
2. Use poc_tb3.launch.py for full TurtleBot3 + Gazebo integration
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch configuration
    enable_frontend = LaunchConfiguration("enable_frontend")
    enable_foxglove = LaunchConfiguration("enable_foxglove")
    foxglove_port = LaunchConfiguration("foxglove_port")
    
    # Simulation parameters
    sim_duration = LaunchConfiguration("sim_duration")
    trajectory_type = LaunchConfiguration("trajectory_type")
    linear_velocity = LaunchConfiguration("linear_velocity")
    
    # Visualization
    publish_world_markers = LaunchConfiguration("publish_world_markers")
    
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument("enable_frontend", default_value="true",
            description="Enable frontend node (disable for odom-only testing)"),
        DeclareLaunchArgument("enable_foxglove", default_value="false",
            description="Enable Foxglove bridge for visualization"),
        DeclareLaunchArgument("foxglove_port", default_value="8765",
            description="Foxglove bridge port"),
        
        # Simulation configuration
        DeclareLaunchArgument("sim_duration", default_value="60.0",
            description="Simulation duration in seconds"),
        DeclareLaunchArgument("trajectory_type", default_value="circle",
            description="Trajectory type: circle, figure8, straight"),
        DeclareLaunchArgument("linear_velocity", default_value="0.3",
            description="Linear velocity in m/s"),
        DeclareLaunchArgument("publish_world_markers", default_value="true",
            description="Publish world obstacle markers for visualization"),

        # Static TF: base_link -> camera_link (identity)
        # Required for TF chain: odom -> base_link -> camera_link
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_tf_cam",
            arguments=["0", "0", "0", "0", "0", "0", "base_link", "camera_link"],
            output="screen",
        ),

        # Simulation world node: ground truth + noisy odometry
        # Publishes: /sim/ground_truth, /odom, /cdwm/world_markers
        Node(
            package="fl_slam_poc",
            executable="sim_world_node",
            name="sim_world",
            output="screen",
            parameters=[{
                "sim_duration": sim_duration,
                "trajectory_type": trajectory_type,
                "linear_velocity": linear_velocity,
                "publish_world_markers": publish_world_markers,
            }],
        ),

        # Odometry bridge: converts /odom (absolute) to /sim/odom (delta)
        # Same pattern as TB3 launch for consistency
        Node(
            package="fl_slam_poc",
            executable="tb3_odom_bridge_node",
            name="odom_bridge",
            output="screen",
            parameters=[{
                "input_topic": "/odom",
                "output_topic": "/sim/odom",
            }],
        ),

        # Frontend: processes sensors and publishes loop factors
        # Reads: /odom, /scan, /camera/* (sensors from real sim when available)
        # Publishes: /sim/loop_factor, /sim/anchor_create
        Node(
            package="fl_slam_poc",
            executable="frontend_node",
            name="fl_frontend",
            output="screen",
            parameters=[{
                "odom_topic": "/odom",
                "odom_is_delta": False,
                # Standard sensor topics - will receive data when real sim is running
                "scan_topic": "/scan",
                "depth_topic": "/camera/depth/image_raw",
                "camera_topic": "/camera/image_raw",
                "camera_info_topic": "/camera/depth/camera_info",
            }],
            condition=IfCondition(enable_frontend),
        ),

        # Backend: fuses odometry and loop factors, publishes state estimate
        # Reads: /sim/odom (delta), /sim/loop_factor, /sim/anchor_create
        # Publishes: /cdwm/state, /cdwm/markers, TF odom->base_link
        Node(
            package="fl_slam_poc",
            executable="fl_backend_node",
            name="fl_backend",
            output="screen",
        ),

        # Static TF: map -> odom (identity for now, would be set by localization)
        # Required for complete TF tree: map -> odom -> base_link -> camera_link
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_tf_map_odom",
            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
            output="screen",
        ),

        # Foxglove bridge for visualization
        # See: https://github.com/foxglove/foxglove-sdk/blob/main/ros/src/foxglove_bridge/README.md
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[{
                "port": foxglove_port,
                "address": "0.0.0.0",  # Listen on all interfaces
                "capabilities": [
                    "clientPublish",
                    "parameters",
                    "parametersSubscribe",
                    "services",
                    "connectionGraph",  # Shows node graph in Foxglove
                    "assets",
                ],
                "num_threads": 0,  # Auto (one per core)
                "max_qos_depth": 25,
                "include_hidden": False,
            }],
            condition=IfCondition(enable_foxglove),
        ),
    ])
