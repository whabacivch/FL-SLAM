#!/usr/bin/env python3
"""
Direct bag inspection without ROS runtime.
Reads the bag database directly to extract frame IDs.
"""

import sqlite3
import sys
from pathlib import Path

try:
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message
except ImportError:
    print("ERROR: This script requires ROS 2 Python libraries.")
    print("Source your ROS 2 workspace first:")
    print("  source fl_ws/install/setup.bash")
    sys.exit(1)


def inspect_bag(bag_path: Path):
    """Inspect bag file and extract frame information."""

    db_file = bag_path / "tb3_slam3d_small_ros2.db3"
    if not db_file.exists():
        print(f"ERROR: Bag database not found at {db_file}")
        return

    print("=" * 60)
    print("TurtleBot3 Bag Frame Inspection (Direct DB Read)")
    print("=" * 60)
    print()

    # Connect to bag database
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    # Get topic metadata
    print("1. Topic Information:")
    print("-" * 60)
    cursor.execute("SELECT id, name, type FROM topics")
    topics = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

    for topic_id, (name, msg_type) in topics.items():
        cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (topic_id,))
        count = cursor.fetchone()[0]
        print(f"  {name:50s} {msg_type:40s} {count:>5d} msgs")
    print()

    # Inspect /scan frame
    print("2. LaserScan Frame IDs:")
    print("-" * 60)
    scan_topic_id = next((tid for tid, (name, _) in topics.items() if name == "/scan"), None)
    if scan_topic_id:
        cursor.execute(
            "SELECT data FROM messages WHERE topic_id = ? LIMIT 5",
            (scan_topic_id,))

        msg_type = get_message("sensor_msgs/msg/LaserScan")
        frame_ids = set()

        for (data,) in cursor.fetchall():
            msg = deserialize_message(data, msg_type)
            frame_ids.add(msg.header.frame_id)

        print(f"  Found frame IDs: {frame_ids}")
    else:
        print("  /scan topic not found")
    print()

    # Inspect /odom frames
    print("3. Odometry Frame IDs:")
    print("-" * 60)
    odom_topic_id = next((tid for tid, (name, _) in topics.items() if name == "/odom"), None)
    if odom_topic_id:
        cursor.execute(
            "SELECT data FROM messages WHERE topic_id = ? LIMIT 5",
            (odom_topic_id,))

        msg_type = get_message("nav_msgs/msg/Odometry")
        header_frames = set()
        child_frames = set()

        for (data,) in cursor.fetchall():
            msg = deserialize_message(data, msg_type)
            header_frames.add(msg.header.frame_id)
            child_frames.add(msg.child_frame_id)

        print(f"  header.frame_id: {header_frames}")
        print(f"  child_frame_id:  {child_frames}")
    else:
        print("  /odom topic not found")
    print()

    # Inspect /tf_static
    print("4. Static TF Frames:")
    print("-" * 60)
    tf_static_id = next((tid for tid, (name, _) in topics.items() if name == "/tf_static"), None)
    if tf_static_id:
        cursor.execute(
            "SELECT data FROM messages WHERE topic_id = ?",
            (tf_static_id,))

        msg_type = get_message("tf2_msgs/msg/TFMessage")
        transforms = []

        for (data,) in cursor.fetchall():
            msg = deserialize_message(data, msg_type)
            for transform in msg.transforms:
                transforms.append(
                    f"{transform.header.frame_id} -> {transform.child_frame_id}")

        print("  Static transforms:")
        for tf in transforms:
            print(f"    {tf}")
    else:
        print("  /tf_static topic not found")
    print()

    # Sample /tf
    print("5. Sample Dynamic TF Frames (first 20):")
    print("-" * 60)
    tf_id = next((tid for tid, (name, _) in topics.items() if name == "/tf"), None)
    if tf_id:
        cursor.execute(
            "SELECT data FROM messages WHERE topic_id = ? LIMIT 20",
            (tf_id,))

        msg_type = get_message("tf2_msgs/msg/TFMessage")
        transforms = set()

        for (data,) in cursor.fetchall():
            msg = deserialize_message(data, msg_type)
            for transform in msg.transforms:
                transforms.add(
                    f"{transform.header.frame_id} -> {transform.child_frame_id}")

        print("  Dynamic transforms:")
        for tf in sorted(transforms):
            print(f"    {tf}")
    else:
        print("  /tf topic not found")
    print()

    conn.close()

    print("=" * 60)
    print("Recommended Launch Parameters:")
    print("=" * 60)
    print()
    print("Based on this inspection, update your launch file if needed:")
    print("  odom_frame:   (check /odom header.frame_id above)")
    print("  base_frame:   (check /odom child_frame_id above)")
    print("  scan_frame:   (check /scan header.frame_id above)")
    print()


if __name__ == "__main__":
    # Default assumes running from Impact Project_v1 directory tree.
    default_bag = Path(__file__).parent.parent / "rosbags" / "tb3_slam3d_small_ros2"
    bag_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_bag

    if not bag_path.exists():
        print(f"ERROR: Bag not found at {bag_path}")
        print("Expected directory structure:")
        print("  Impact Project_v1/")
        print("    rosbags/")
        print("      tb3_slam3d_small_ros2/")
        print("")
        print("Or pass an explicit bag path:")
        print("  python3 scripts/inspect_bag_direct.py /path/to/bag_dir")
        sys.exit(1)

    inspect_bag(bag_path)
