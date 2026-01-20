from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enable_foxglove = LaunchConfiguration("enable_foxglove")
    foxglove_port = LaunchConfiguration("foxglove_port")
    enable_frontend = LaunchConfiguration("enable_frontend")
    publish_sensors = LaunchConfiguration("publish_sensors")
    publish_anchor = LaunchConfiguration("publish_anchor")
    publish_loop_factor = LaunchConfiguration("publish_loop_factor")
    publish_world_markers = LaunchConfiguration("publish_world_markers")

    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_foxglove", default_value="true"),
            DeclareLaunchArgument("foxglove_port", default_value="8765"),
            DeclareLaunchArgument("enable_frontend", default_value="true"),
            # Demo defaults: work out-of-the-box for Foxglove
            DeclareLaunchArgument("publish_sensors", default_value="true"),
            DeclareLaunchArgument("publish_anchor", default_value="true"),
            DeclareLaunchArgument("publish_loop_factor", default_value="true"),
            DeclareLaunchArgument("publish_world_markers", default_value="true"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_tf_cam",
                arguments=["0", "0", "0", "0", "0", "0", "base_link", "camera_link"],
                output="screen",
            ),
            Node(
                package="fl_slam_poc",
                executable="sim_world_node",
                name="sim_world",
                output="screen",
                parameters=[
                    {"publish_sensors": publish_sensors},
                    {"publish_anchor": publish_anchor},
                    {"publish_loop_factor": publish_loop_factor},
                    {"publish_world_markers": publish_world_markers},
                ],
            ),
            Node(
                package="fl_slam_poc",
                executable="frontend_node",
                name="fl_frontend",
                output="screen",
                parameters=[
                    {"odom_topic": "/sim/odom"},
                    {"odom_is_delta": True},
                ],
                condition=IfCondition(enable_frontend),
            ),
            Node(
                package="fl_slam_poc",
                executable="fl_backend_node",
                name="fl_backend",
                output="screen",
            ),
            Node(
                package="fl_slam_poc",
                executable="sim_semantics_node",
                name="sim_semantics",
                output="screen",
            ),
            Node(
                package="fl_slam_poc",
                executable="dirichlet_backend_node",
                name="dirichlet_backend",
                output="screen",
            ),
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                output="screen",
                parameters=[{"port": foxglove_port}],
                condition=IfCondition(enable_foxglove),
            ),
        ]
    )
