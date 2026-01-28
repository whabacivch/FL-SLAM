#!/usr/bin/env python3
"""
Compare IMU accel (gravity direction) with odom orientation.

For each IMU sample we have accel in Livox IMU frame. We transform to base using
R_base_imu, then compare with expected gravity in body from odom orientation:
  g_body = R_odom @ [0, 0, -9.81]
If accel agrees with odom, a_base (from IMU) should align with g_body when stationary.

Usage:
  .venv/bin/python tools/compare_accel_odom.py docs/raw_sensor_dump/imu_raw_first_300.csv docs/raw_sensor_dump/odom_raw_first_300.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# T_base_imu from launch: [x, y, z, rx, ry, rz]; rotvec in rad
R_BASE_IMU_ROTVEC = np.array([-0.015586, 0.489293, 0.0])
# Expected measured accel in world when level and stationary: +g upward (specific force)
EXPECTED_ACCEL_WORLD = np.array([0.0, 0.0, 9.81])


def load_imu(csv_path: str):
    """Load stamp_sec, accel x,y,z from IMU CSV."""
    stamps, ax, ay, az = [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stamps.append(float(row["stamp_sec"]))
            ax.append(float(row["accel_x"]))
            ay.append(float(row["accel_y"]))
            az.append(float(row["accel_z"]))
    return np.array(stamps), np.array(ax), np.array(ay), np.array(az)


def load_odom(csv_path: str):
    """Load stamp_sec and quat from odom CSV."""
    stamps = []
    quats = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stamps.append(float(row["stamp_sec"]))
            quats.append((
                float(row["qx"]), float(row["qy"]),
                float(row["qz"]), float(row["qw"]),
            ))
    return np.array(stamps), quats


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare IMU accel with odom orientation.")
    ap.add_argument("imu_csv", nargs="?", default="docs/raw_sensor_dump/imu_raw_first_300.csv")
    ap.add_argument("odom_csv", nargs="?", default="docs/raw_sensor_dump/odom_raw_first_300.csv")
    args = ap.parse_args()

    imu_path = Path(args.imu_csv)
    odom_path = Path(args.odom_csv)
    if not imu_path.is_file() or not odom_path.is_file():
        print("Usage: compare_accel_odom.py <imu_csv> <odom_csv>", file=sys.stderr)
        return 1

    t_imu, ax, ay, az = load_imu(str(imu_path))
    t_odom, quats = load_odom(str(odom_path))
    accel_imu = np.stack([ax, ay, az], axis=1)

    R_base_imu = Rotation.from_rotvec(R_BASE_IMU_ROTVEC).as_matrix()

    # For each IMU sample, nearest odom by time
    dot_products = []
    for i in range(len(t_imu)):
        j = np.argmin(np.abs(t_odom - t_imu[i]))
        q = quats[j]
        R_odom = Rotation.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
        a_base = R_base_imu @ accel_imu[i]
        g_body = R_odom @ EXPECTED_ACCEL_WORLD
        a_n = a_base / (np.linalg.norm(a_base) + 1e-12)
        g_n = g_body / (np.linalg.norm(g_body) + 1e-12)
        dot_products.append(float(np.dot(a_n, g_n)))

    dot_products = np.array(dot_products)
    mean_dot = float(np.mean(dot_products))
    min_dot = float(np.min(dot_products))
    max_dot = float(np.max(dot_products))

    print("Accel vs odom (gravity direction agreement)")
    print("  Dot product of normalized a_base (from IMU) and g_body (from odom):")
    print(f"    mean = {mean_dot:.4f}  (1 = perfect agreement)")
    print(f"    min  = {min_dot:.4f}")
    print(f"    max  = {max_dot:.4f}")
    if mean_dot > 0.99:
        print("  -> Accel agrees well with odom orientation.")
    elif mean_dot > 0.95:
        print("  -> Accel mostly agrees; small frame or timing offset.")
    else:
        print("  -> Accel and odom orientation disagree (frame, scale, or timing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
