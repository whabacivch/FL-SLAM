# Archive

This folder contains obsolete files and folders that are no longer used in the current project but are kept for historical reference.

## Contents

### Build Artifacts (Obsolete)
- `build_3d/` - Old 3D-specific build directory (replaced by unified `build/`)
- `install_3d/` - Old 3D-specific install directory (replaced by unified `install/`)
- `log_3d/` - Old 3D-specific log directory (replaced by unified `log/`)

### Code Backups
- `frontend_node_ORIGINAL_BACKUP.py` - Original frontend node implementation before refactoring

### Obsolete Scripts (`obsolete_scripts/`)
- `inspect_bag_direct.py` - Direct SQLite bag inspection (superseded by `scripts/inspect_rosbag_topics.sh`)
- `record_test_bag.sh` - Attempted Gazebo bag recording script (never worked due to headless Gazebo crashes)

### Obsolete Launch Files (`obsolete_launch/`)
- `poc_b.launch.py` - Minimal Dirichlet demo (only `sim_semantics_node` + `dirichlet_backend_node`)
- `poc_all.launch.py` - Full system demo with unused Dirichlet components

### Old Scripts (`old_scripts/`)
- Historical copies of previously-used helper/test scripts (kept for reference only)

## Notes

- Everything is now 3D by default, so the `_3d` suffixes were redundant
- Dirichlet-based semantic SLAM (`dirichlet_backend_node`, `sim_semantics_node`) is experimental and not used in the main pipeline
- These files are kept for reference but should not be used in development
- To restore any file, copy it from this archive back to the appropriate location

## Current Active Files

### Scripts (`scripts/`)
- `run_and_evaluate.sh` - MVP pipeline (M3DGR SLAM + metrics + plots)
- `align_ground_truth.py` - Align M3DGR ground truth timestamps to ROS time
- `evaluate_slam.py` - Compute ATE/RPE metrics and generate plots
- `download_tb3_rosbag.sh` - Download alternative 2D test rosbag (future/optional)
- `download_r2b_dataset.sh` - Download alternative 3D dataset (future/optional)
- `test-integration.sh` - Integration test script (future/optional)
- `inspect_rosbag_topics.sh` - Inspect rosbag topics

### Launch Files (`fl_ws/src/fl_slam_poc/launch/`)
- `poc_m3dgr_rosbag.launch.py` - MVP rosbag launch (M3DGR Dynamic01)
- `poc_tb3_rosbag.launch.py` - Alternative 2D rosbag testing
- `poc_3d_rosbag.launch.py` - Alternative 3D point cloud mode
- `poc_tb3.launch.py` - Gazebo live integration

**Last updated:** January 21, 2026
