# Raw sensor dump (first 300 messages)

Permanent copy of the first 300 IMU and odom messages from the M3DGR Dynamic01_ros2 bag, for reference and plotting.

## Files

| File | Source topic | ~Duration | Columns |
|------|--------------|----------|---------|
| `imu_raw_first_300.csv` | `/livox/mid360/imu` | ~1.5 s | stamp_sec, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z |
| `imu_raw_first_3000.csv` | `/livox/mid360/imu` | ~15 s | same (for dead-reckon plot aligned with odom) |
| `odom_raw_first_300.csv` | `/odom` | ~15 s | stamp_sec, x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz |

## How generated

```bash
.venv/bin/python tools/dump_raw_imu_odom.py rosbags/m3dgr/Dynamic01_ros2 --max-imu 300 --max-odom 300 --out-dir docs/raw_sensor_dump
```

## Dead-reckon map from vx + wz

With first pose as origin, you can build a 2D path: **vx** gives forward distance (vx × Δt), **wz** gives yaw rate so heading θ = ∫ wz dt; then Δx = vx·Δt·cos(θ), Δy = vx·Δt·sin(θ). The script does this and plots the path:

```bash
.venv/bin/python tools/dead_reckon_odom_dump.py docs/raw_sensor_dump/odom_raw_first_300.csv --out docs/raw_sensor_dump/dead_reckon_path.png
```

Output: `dead_reckon_path.png` — (x,y) with first pose at origin, X = first-pose forward.

## Notes

- Odom pose is **absolute** in the bag (e.g. z ≈ 30 m); the backend uses first-odom-as-origin internally.
- IMU: gyro in rad/s, accel in m/s² (Livox frame).
- To regenerate from a different bag or count, run the same command with different `bag_path` / `--max-imu` / `--max-odom`.
