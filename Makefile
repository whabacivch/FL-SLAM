# Impact Project v1 — Golden Child SLAM v2
# Run from project root.

.PHONY: eval gc-eval build

# Primary eval: run GC pipeline + evaluation (artifacts under results/gc_*)
eval: gc-eval

gc-eval:
	bash tools/run_and_evaluate_gc.sh

# Build only (no rosbag run)
build:
	cd fl_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select fl_slam_poc && source install/setup.bash
