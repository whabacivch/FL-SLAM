from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
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
        ]
    )
