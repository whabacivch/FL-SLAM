from setuptools import find_packages, setup

package_name = "fl_slam_poc"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/poc_a.launch.py",
                "launch/poc_b.launch.py",
                "launch/poc_all.launch.py",
                "launch/poc_tb3.launch.py",
                "launch/poc_tb3_rosbag.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/config",
            [
                "config/qos_override.yaml",
            ],
        ),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="Frobenius-Legendre compositional inference SLAM POC (ROS 2 Jazzy)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim_world_node = fl_slam_poc.nodes.sim_world_node:main",
            "fl_backend_node = fl_slam_poc.nodes.fl_backend_node:main",
            "sim_semantics_node = fl_slam_poc.nodes.sim_semantics_node:main",
            "dirichlet_backend_node = fl_slam_poc.nodes.dirichlet_backend_node:main",
            "tb3_odom_bridge_node = fl_slam_poc.nodes.tb3_odom_bridge_node:main",
            "frontend_node = fl_slam_poc.nodes.frontend_node:main",
            "image_decompress_node = fl_slam_poc.nodes.image_decompress_node:main",
        ],
    },
)
