#!/usr/bin/env python3
"""
Dead-reckon 2D paths: (1) odom vx + wz, (2) odom vx + quat yaw, (3) IMU only (gyro + accel).
Assumes x = forward. Side-by-side; first pose = origin.

Usage:
  .venv/bin/python tools/dead_reckon_odom_dump.py docs/raw_sensor_dump/odom_raw_first_300.csv
  .venv/bin/python tools/dead_reckon_odom_dump.py docs/raw_sensor_dump/odom_raw_first_300.csv --imu-csv docs/raw_sensor_dump/imu_raw_first_300.csv --out docs/raw_sensor_dump/dead_reckon_path.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# T_base_imu from launch [x,y,z,rx,ry,rz]; rotvec in rad
R_BASE_IMU_ROTVEC = np.array([-0.015586, 0.489293, 0.0])


def quat_xyzw_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (rad) from ROS xyzw quaternion, Z-up convention."""
    # siny_cosp = 2 * (qw*qz + qx*qy); cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def load_odom(csv_path: str):
    """Load stamp, vx, wz, and first-pose quat from odom CSV."""
    stamps = []
    vx = []
    wz = []
    quats = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stamps.append(float(row["stamp_sec"]))
            vx.append(float(row["vx"]))
            wz.append(float(row["wz"]))
            quats.append((float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])))
    return (
        np.array(stamps),
        np.array(vx),
        np.array(wz),
        quats,
    )


def dead_reckon(stamp: np.ndarray, vx: np.ndarray, wz: np.ndarray, yaw0: float):
    """First pose as origin. Heading from integrating wz. Returns (x, y, theta)."""
    n = len(stamp)
    x = np.zeros(n)
    y = np.zeros(n)
    theta = np.zeros(n)
    theta[0] = yaw0
    for i in range(1, n):
        dt = stamp[i] - stamp[i - 1]
        if dt <= 0:
            dt = stamp[1] - stamp[0] if i == 1 else (stamp[min(i + 1, n - 1)] - stamp[i - 1]) / 2
        if dt <= 0:
            dt = 0.05
        x[i] = x[i - 1] + vx[i - 1] * dt * np.cos(theta[i - 1])
        y[i] = y[i - 1] + vx[i - 1] * dt * np.sin(theta[i - 1])
        theta[i] = theta[i - 1] + wz[i - 1] * dt
    return x, y, theta


def dead_reckon_quat(stamp: np.ndarray, vx: np.ndarray, quats: list):
    """First pose as origin. Heading from quaternion at each row. Returns (x, y)."""
    n = len(stamp)
    yaw = np.array([quat_xyzw_to_yaw(*q) for q in quats])
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        dt = stamp[i] - stamp[i - 1]
        if dt <= 0:
            dt = stamp[1] - stamp[0] if i == 1 else (stamp[min(i + 1, n - 1)] - stamp[i - 1]) / 2
        if dt <= 0:
            dt = 0.05
        x[i] = x[i - 1] + vx[i - 1] * dt * np.cos(yaw[i - 1])
        y[i] = y[i - 1] + vx[i - 1] * dt * np.sin(yaw[i - 1])
    return x, y


def load_imu(csv_path: str):
    """Load stamp_sec, gyro x,y,z, accel x,y,z from IMU CSV."""
    stamps, gx, gy, gz, ax, ay, az = [], [], [], [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stamps.append(float(row["stamp_sec"]))
            gx.append(float(row["gyro_x"]))
            gy.append(float(row["gyro_y"]))
            gz.append(float(row["gyro_z"]))
            ax.append(float(row["accel_x"]))
            ay.append(float(row["accel_y"]))
            az.append(float(row["accel_z"]))
    return (
        np.array(stamps),
        np.stack([np.array(gx), np.array(gy), np.array(gz)], axis=1),
        np.stack([np.array(ax), np.array(ay), np.array(az)], axis=1),
    )


def dead_reckon_imu(stamp: np.ndarray, gyro: np.ndarray, accel: np.ndarray):
    """
    IMU-only 2D dead reckoning: x = forward.
    Gyro (in IMU frame) -> transform to base, integrate omega_z for yaw.
    Accel (in IMU frame) -> transform to base, use x,y as in-plane acceleration (level approx),
    rotate to world by yaw, integrate for v then x,y.
    First pose: origin, heading 0, zero velocity.
    """
    R_base_imu = Rotation.from_rotvec(R_BASE_IMU_ROTVEC).as_matrix()
    n = len(stamp)
    theta = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        dt = stamp[i] - stamp[i - 1]
        if dt <= 0:
            dt = 0.004  # ~250 Hz fallback
        dt = min(dt, 0.1)
        # Gyro -> body frame, then yaw rate = omega_z (body)
        omega_base = R_base_imu @ gyro[i - 1]
        wz = omega_base[2]
        theta[i] = theta[i - 1] + wz * dt
        # Accel -> base; in-plane (x,y) is horizontal acceleration when level
        a_base = R_base_imu @ accel[i - 1]
        ax_b, ay_b = a_base[0], a_base[1]
        c, s = np.cos(theta[i - 1]), np.sin(theta[i - 1])
        ax_w = c * ax_b - s * ay_b
        ay_w = s * ax_b + c * ay_b
        vx[i] = vx[i - 1] + ax_w * dt
        vy[i] = vy[i - 1] + ay_w * dt
        x[i] = x[i - 1] + vx[i - 1] * dt
        y[i] = y[i - 1] + vy[i - 1] * dt
    return x, y


def main() -> int:
    ap = argparse.ArgumentParser(description="Dead-reckon 2D path: odom (vx+wz, vx+quat) and optionally IMU only.")
    ap.add_argument("csv_path", nargs="?", default="docs/raw_sensor_dump/odom_raw_first_300.csv")
    ap.add_argument("--imu-csv", default=None, help="IMU CSV for IMU-only dead reckoning (e.g. imu_raw_first_300.csv)")
    ap.add_argument("--out", default=None, help="Output plot path")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    stamp, vx, wz, quats = load_odom(str(csv_path))
    if stamp.size == 0:
        print("No rows.", file=sys.stderr)
        return 1

    yaw0 = quat_xyzw_to_yaw(*quats[0])
    x, y, theta = dead_reckon(stamp, vx, wz, yaw0)
    x_q, y_q = dead_reckon_quat(stamp, vx, quats)

    # Rotate path so first pose heading is +X; start stays at (0,0)
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    x_plot = x * c - y * s
    y_plot = x * s + y * c
    x_q_plot = x_q * c - y_q * s
    y_q_plot = x_q * s + y_q * c
    if y_plot.min() < 0:
        y_plot = -y_plot
    if y_q_plot.min() < 0:
        y_q_plot = -y_q_plot

    # IMU-only path: --imu-csv, or same-dir imu_raw_first_3000.csv (~15 s), else imu_raw_first_300.csv
    if args.imu_csv is not None:
        imu_csv = args.imu_csv
    else:
        dir_ = csv_path.parent
        if (dir_ / "imu_raw_first_3000.csv").is_file():
            imu_csv = str(dir_ / "imu_raw_first_3000.csv")
        else:
            imu_csv = str(dir_ / "imu_raw_first_300.csv")
    x_imu_plot = y_imu_plot = None
    if Path(imu_csv).is_file():
        t_imu, gyro, accel = load_imu(imu_csv)
        if t_imu.size > 1:
            x_imu, y_imu = dead_reckon_imu(t_imu, gyro, accel)
            x_imu_plot = x_imu.copy()
            y_imu_plot = y_imu.copy()
            if y_imu_plot.min() < 0:
                y_imu_plot = -y_imu_plot

    def draw_path(ax, y_ax, x_ax, title: str):
        ax.plot(y_ax, x_ax, "b.-", markersize=2, linewidth=0.6, label="Path")
        ax.plot(y_ax[0], x_ax[0], "go", markersize=10, label="Start (0,0)")
        ax.plot(y_ax[-1], x_ax[-1], "r^", markersize=8, label="End")
        ax.set_xlabel("y (m)")
        ax.set_ylabel("x (m) — forward")
        ax.set_title(title)
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-5, 5)
        ax.set_aspect("auto")
        ax.legend()
        ax.grid(True, alpha=0.3)

    n_panels = 3 if x_imu_plot is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
    if n_panels == 2:
        ax_wz, ax_quat = axes
    else:
        ax_wz, ax_quat, ax_imu = axes
    draw_path(ax_wz, y_plot, x_plot, "vx + wz (integrated yaw)")
    draw_path(ax_quat, y_q_plot, x_q_plot, "vx + quaternion yaw")
    if x_imu_plot is not None:
        draw_path(ax_imu, y_imu_plot, x_imu_plot, "IMU only (gyro + accel)")
        # Lateral (y) comparison: IMU often shows much larger |y| than odom (gyro/accel bias)
        print("y (lateral) range: odom_wz [%.4f, %.4f]  odom_quat [%.4f, %.4f]  IMU [%.4f, %.4f]"
              % (y_plot.min(), y_plot.max(), y_q_plot.min(), y_q_plot.max(),
                 y_imu_plot.min(), y_imu_plot.max()))

    out = args.out
    if out is None:
        out = csv_path.parent / "dead_reckon_path.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
