# FL-SLAM POC Docker Environment
# Based on ROS 2 Jazzy with all dependencies pre-installed

FROM ros:jazzy-ros-base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-scipy \
    python3-numpy \
    python3-pytest \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-visualization-msgs \
    ros-jazzy-nav-msgs \
    ros-jazzy-sensor-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-std-msgs \
    ros-jazzy-launch-testing \
    ros-jazzy-launch-testing-ros \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /ros2_ws

# Copy package source
COPY fl_ws/src/fl_slam_poc src/fl_slam_poc

# Build the workspace
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --symlink-install --packages-select fl_slam_poc

# Source setup in bashrc for interactive shells
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

# Expose Foxglove WebSocket port
EXPOSE 8765

# Default command: bash
CMD ["/bin/bash"]

