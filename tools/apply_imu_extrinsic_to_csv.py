#!/usr/bin/env python3
"""
Apply the pipeline's IMU extrinsic (R_base_imu) to raw IMU CSV and write base-frame values.

Uses the same T_base_imu as gc_rosbag.launch / gc_unified.yaml:
  [x, y, z, rx, ry, rz] rotvec (radians) -> R_base_imu
  gyro_base = R_base_imu @ gyro_imu
  accel_base = R_base_imu @ accel_imu

The extrinsic only ROTATES into base frame; it does NOT remove gravity.
When stationary, accel_base ≈ (0, 0, +9.8) (reaction to gravity).
Use --linear to also subtract gravity so linear accel ≈ 0 when stationary:
  linear_accel_base = accel_base + gravity_W  (GC_GRAVITY_W = (0,0,-9.81))

Input CSV: stamp_sec, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z (IMU/sensor frame)
Output CSV: stamp_sec, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z (base frame; or linear accel if --linear)

Usage:
  .venv/bin/python tools/apply_imu_extrinsic_to_csv.py docs/raw_sensor_dump/imu_raw_first_300.csv
  .venv/bin/python tools/apply_imu_extrinsic_to_csv.py docs/raw_sensor_dump/imu_raw_first_300.csv --linear -o docs/raw_sensor_dump/imu_linear_first_300.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Same as gc_rosbag.launch.py / gc_unified.yaml (rotvec radians: rx, ry, rz)
T_BASE_IMU_6D = [0.0, 0.0, 0.0, -0.015586, 0.489293, 0.0]
# Same as constants.GC_GRAVITY_W: Z-up, gravity down (m/s^2)
GRAVITY_W = np.array([0.0, 0.0, -9.81], dtype=np.float64)
# Livox Mid-360 IMU outputs in g's; same as constants.GC_IMU_ACCEL_SCALE
ACCEL_G_TO_MPS2 = 9.81


def parse_T_base_imu() -> np.ndarray:
    """Return (3,3) R_base_imu from 6D [x,y,z,rx,ry,rz] rotvec."""
    v = np.array(T_BASE_IMU_6D, dtype=np.float64).reshape(-1)
    t = v[:3]
    rotvec = v[3:6]
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return R


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Apply IMU extrinsic to raw IMU CSV (output base-frame gyro/accel)."
    )
    ap.add_argument(
        "input_csv",
        help="Input CSV: stamp_sec, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z",
    )
    ap.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path (default: input path with _extrinsic_applied before .csv)",
    )
    ap.add_argument(
        "--linear",
        action="store_true",
        help="Output linear acceleration (accel_base + gravity_W) so stationary -> 0",
    )
    args = ap.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        if "_raw" in stem:
            stem = stem.replace("_raw", "_linear" if args.linear else "_extrinsic_applied")
        else:
            stem = stem + ("_linear" if args.linear else "_extrinsic_applied")
        output_path = input_path.parent / (stem + input_path.suffix)

    R = parse_T_base_imu()

    rows_out = []
    with open(input_path) as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames)
        for row in r:
            stamp = float(row["stamp_sec"])
            gyro_imu = np.array(
                [float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"])],
                dtype=np.float64,
            )
            # Raw bag/CSV for Livox is in g's; convert to m/s^2 to match pipeline
            accel_imu = np.array(
                [
                    float(row["accel_x"]),
                    float(row["accel_y"]),
                    float(row["accel_z"]),
                ],
                dtype=np.float64,
            ) * ACCEL_G_TO_MPS2
            gyro_base = R @ gyro_imu
            accel_base = R @ accel_imu
            if args.linear:
                # Linear accel = specific force + gravity_W -> 0 when stationary
                accel_base = accel_base + GRAVITY_W
            rows_out.append(
                {
                    "stamp_sec": f"{stamp:.9f}",
                    "gyro_x": gyro_base[0],
                    "gyro_y": gyro_base[1],
                    "gyro_z": gyro_base[2],
                    "accel_x": accel_base[0],
                    "accel_y": accel_base[1],
                    "accel_z": accel_base[2],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["stamp_sec", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"],
        )
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
