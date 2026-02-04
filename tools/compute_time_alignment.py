#!/usr/bin/env python3
"""
Compute per-stream time alignment (offset + drift) relative to a reference topic.

Outputs a YAML profile:
  time_alignment:
    reference: <topic>
    window_sec: <duration>
    t0_sec: <reference start>
    streams:
      <topic>:
        offset_sec: <median offset>
        drift_sec_per_sec: <slope>
        t0_sec: <reference start>

Usage:
  ./tools/compute_time_alignment.py <rosbag> --duration 60 --reference /acl_jackal/lidar_points \
    --topics /acl_jackal/forward/imu /acl_jackal/jackal_velocity_controller/odom \
    --out /path/to/profile.yaml
"""

import argparse
from collections import deque
import numpy as np
import yaml

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def stamp_to_sec(msg):
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def summarize(pairs):
    if not pairs:
        return None
    times = np.array([p[0] for p in pairs], dtype=np.float64)
    offs = np.array([p[1] for p in pairs], dtype=np.float64)
    x = times - times[0]
    if len(x) >= 2:
        slope, _ = np.polyfit(x, offs, 1)
    else:
        slope = 0.0
    return {
        "offset_sec": float(np.median(offs)),
        "drift_sec_per_sec": float(slope),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bag", help="Path to ros2 bag directory")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--reference", required=True, help="Reference topic (e.g., lidar)")
    p.add_argument("--topics", nargs="+", required=True, help="Topics to align against reference")
    p.add_argument("--out", required=True, help="Output YAML path")
    args = p.parse_args()

    reader = SequentialReader()
    reader.open(StorageOptions(uri=args.bag, storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))

    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.reference not in types:
        raise SystemExit(f"reference topic not found: {args.reference}")
    topics = [args.reference] + [t for t in args.topics if t in types]
    msg_types = {t: get_message(types[t]) for t in topics}

    buffers = {t: deque() for t in topics if t != args.reference}
    offsets = {t: [] for t in topics if t != args.reference}

    t0 = None
    end_t = None

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in topics:
            continue
        msg = deserialize_message(data, msg_types[topic])
        if not hasattr(msg, "header"):
            continue
        t = stamp_to_sec(msg)
        if topic == args.reference:
            if t0 is None:
                t0 = t
                end_t = t0 + args.duration
            if t > end_t:
                break
            for ot, buf in buffers.items():
                if not buf:
                    continue
                while len(buf) > 2 and buf[1] < t:
                    buf.popleft()
                best = min(buf, key=lambda x: abs(x - t))
                offsets[ot].append((t, best - t))
        else:
            if t0 is not None and end_t is not None and t > end_t + 1.0:
                continue
            buffers[topic].append(t)

    out = {
        "time_alignment": {
            "reference": args.reference,
            "window_sec": float(args.duration),
            "t0_sec": float(t0 if t0 is not None else 0.0),
            "streams": {},
        }
    }
    for t, pairs in offsets.items():
        stats = summarize(pairs)
        if stats is None:
            continue
        stats["t0_sec"] = float(t0 if t0 is not None else 0.0)
        out["time_alignment"]["streams"][t] = stats

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)


if __name__ == "__main__":
    main()
