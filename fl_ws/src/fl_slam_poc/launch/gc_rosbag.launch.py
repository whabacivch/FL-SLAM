"""
Golden Child SLAM v2 Rosbag Launch File.

Launches the Golden Child backend with a rosbag for evaluation.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Golden Child SLAM evaluation."""
    
    # Declare launch arguments
    bag_arg = DeclareLaunchArgument(
        "bag",
        description="Path to rosbag directory",
    )
    
    trajectory_path_arg = DeclareLaunchArgument(
        "trajectory_export_path",
        default_value="/tmp/gc_slam_trajectory.tum",
        description="Path to export trajectory in TUM format",
    )
    
    # Golden Child backend node
    gc_backend = Node(
        package="fl_slam_poc",
        executable="gc_backend_node",
        name="gc_backend",
        output="screen",
        parameters=[
            {
                "lidar_topic": "/livox/mid360/points",
                "odom_topic": "/odom",
                "imu_topic": "/livox/mid360/imu",
                "trajectory_export_path": LaunchConfiguration("trajectory_export_path"),
                "odom_frame": "odom",
                "base_frame": "base_link",
                "status_check_period_sec": 5.0,
                "forgetting_factor": 0.99,
            }
        ],
    )
    
    # Livox converter (converts Livox CustomMsg to PointCloud2)
    # Input: /livox/mid360/lidar (CustomMsg from bag)
    # Output: /livox/mid360/points (PointCloud2 for gc_backend)
    livox_converter = Node(
        package="fl_slam_poc",
        executable="livox_converter",
        name="livox_converter",
        output="screen",
        parameters=[
            {
                "input_topic": "/livox/mid360/lidar",
                "output_topic": "/livox/mid360/points",
            }
        ],
    )
    
    # Rosbag playback (short delay to let nodes initialize)
    bag_play = TimerAction(
        period=3.0,  # 3 second delay
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "bag", "play",
                    LaunchConfiguration("bag"),
                    "--clock",
                    "--rate", "1.0",
                ],
                output="screen",
            ),
        ],
    )
    
    return LaunchDescription([
        bag_arg,
        trajectory_path_arg,
        gc_backend,
        livox_converter,
        bag_play,
    ])
