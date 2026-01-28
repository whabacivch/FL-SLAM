#!/usr/bin/env python3
"""
Plot odom wz (yaw rate) from a dumped odom CSV:
  - 2D: heading in XY plane (x=cos θ, y=sin θ) from integrated wz, as points.
  - Polar: angle = time, r = wz.

Usage:
  .venv/bin/python tools/plot_wz_odom.py /tmp/odom_raw_first_300.csv
  .venv/bin/python tools/plot_wz_odom.py /tmp/odom_raw_first_300.csv --out /tmp/wz_plots.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def load_wz(csv_path: str):
    """Load stamp_sec and wz from odom CSV. Returns (stamp_sec, wz)."""
    stamps = []
    wz = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stamps.append(float(row["stamp_sec"]))
            wz.append(float(row["wz"]))
    return np.array(stamps), np.array(wz)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot odom wz: 2D and polar.")
    ap.add_argument("csv_path", nargs="?", default="/tmp/odom_raw_first_300.csv", help="Odom CSV from dump_raw_imu_odom.py")
    ap.add_argument("--out", default=None, help="Output image path (default: show or save next to CSV)")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    stamp, wz = load_wz(str(csv_path))
    if stamp.size == 0:
        print("No rows in CSV.", file=sys.stderr)
        return 1

    # Integrate wz to get yaw θ(t); then heading in XY plane is (cos θ, sin θ).
    # θ[0] = 0; θ[i] = θ[i-1] + wz[i-1] * dt[i-1]
    dt = np.diff(stamp)
    theta = np.zeros_like(stamp)
    for i in range(1, len(stamp)):
        theta[i] = theta[i - 1] + wz[i - 1] * dt[i - 1]
    x_heading = np.cos(theta)
    y_heading = np.sin(theta)

    fig = plt.figure(figsize=(10, 5))
    ax2d = fig.add_subplot(1, 2, 1)
    ax_polar = fig.add_subplot(1, 2, 2, projection="polar")

    # ---- 2D: heading in XY plane (Y = sin θ = "change in 3D Y direction"), as points ----
    ax2d.scatter(x_heading, y_heading, c=np.arange(len(theta)), cmap="viridis", s=8, alpha=0.8)
    ax2d.set_xlabel("cos θ (heading X)")
    ax2d.set_ylabel("sin θ (heading Y)")
    ax2d.set_title("Heading from integrated wz (points = time)")
    ax2d.set_aspect("equal")
    ax2d.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax2d.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax2d.grid(True, alpha=0.3)
    # Unit circle for reference
    circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle=":", linewidth=0.5)
    ax2d.add_patch(circle)

    # ---- Polar: angle = time (wrapped to 2π), radius = wz ----
    # Map time to angle [0, 2π] so full circle = full time span
    angle_time = 2 * np.pi * np.arange(len(wz)) / max(len(wz) - 1, 1)
    r = wz
    ax_polar.plot(angle_time, r, color="C1", linewidth=0.8, label="wz")
    ax_polar.set_title("wz in polar (angle = time, r = wz)")
    ax_polar.set_ylabel("wz (rad/s)")
    ax_polar.legend(loc="upper right")
    ax_polar.grid(True, alpha=0.3)

    plt.tight_layout()

    out = args.out
    if out is None:
        out = csv_path.parent / (csv_path.stem + "_wz_plots.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
