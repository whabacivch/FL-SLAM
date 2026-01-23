# Ignore launch tests for regular pytest runs (use colcon test for those)
collect_ignore = ['test_end_to_end_launch.py']

import os
import pytest
from typing import Dict, Any

# =============================================================================
# Production Config Fixtures
# =============================================================================
# These fixtures load the actual production configuration used in M3DGR pipeline,
# ensuring tests validate the same code paths as production.


def _load_yaml_file(path: str) -> Dict[str, Any]:
    """Load a YAML config file, handling the ros__parameters wrapper."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    
    # ROS2 YAML files wrap parameters in /**:/ros__parameters:
    if "/**" in data and "ros__parameters" in data.get("/**", {}):
        return data["/**"]["ros__parameters"]
    return data


@pytest.fixture
def prod_config() -> Dict[str, Any]:
    """
    Load production config for M3DGR dataset.
    
    This fixture merges the base config with the M3DGR preset, providing the
    same configuration that would be used in production.
    
    Returns:
        Merged configuration dict
    
    Usage:
        def test_something(prod_config):
            assert prod_config["enable_imu"] == True
    """
    # Try to find config files relative to this test file
    test_dir = os.path.dirname(__file__)
    pkg_root = os.path.dirname(test_dir)
    
    config_dir = os.path.join(pkg_root, "config")
    config_base_path = os.path.join(config_dir, "fl_slam_poc_base.yaml")
    config_preset_path = os.path.join(config_dir, "presets", "m3dgr.yaml")
    
    # Fallback: try installed package location
    if not os.path.exists(config_base_path):
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory("fl_slam_poc")
            config_base_path = os.path.join(pkg_share, "config", "fl_slam_poc_base.yaml")
            config_preset_path = os.path.join(pkg_share, "config", "presets", "m3dgr.yaml")
        except ImportError:
            # ament_index not available (running outside ROS2 environment)
            pass
    
    # Load and merge configs
    base_config = _load_yaml_file(config_base_path) if os.path.exists(config_base_path) else {}
    preset_config = _load_yaml_file(config_preset_path) if os.path.exists(config_preset_path) else {}
    
    # Merge: preset overrides base
    merged = {**base_config, **preset_config}
    return merged


@pytest.fixture
def m3dgr_topic_config() -> Dict[str, str]:
    """
    Return the expected topic names for M3DGR dataset.
    
    These are the parametrized topic names used in production (AUD-002 fix).
    Tests can use this to verify topic wiring.
    """
    return {
        "odom_topic": "/sim/odom",
        "loop_factor_topic": "/sim/loop_factor",
        "anchor_create_topic": "/sim/anchor_create",
        "imu_segment_topic": "/sim/imu_segment",
        "rgbd_evidence_topic": "/sim/rgbd_evidence",
        "pointcloud_topic": "/lidar/points",
        "imu_topic": "/livox/mid360/imu",
    }


@pytest.fixture
def config_manifest() -> Dict[str, Any]:
    """
    Load the config manifest for dataset validation.
    
    This provides the canonical specification of what features and topics
    are required/expected for each dataset.
    """
    import yaml
    
    test_dir = os.path.dirname(__file__)
    pkg_root = os.path.dirname(test_dir)
    manifest_path = os.path.join(pkg_root, "config", "config_manifest.yaml")
    
    if not os.path.exists(manifest_path):
        pytest.skip("config_manifest.yaml not found")
        return {}
    
    with open(manifest_path) as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# Test Utility Fixtures
# =============================================================================

@pytest.fixture
def numpy_seed():
    """Set numpy random seed for reproducible tests."""
    import numpy as np
    np.random.seed(42)
    yield
    # Reset seed after test (optional)


@pytest.fixture
def small_pointcloud():
    """Generate a small test point cloud for ICP tests."""
    import numpy as np
    np.random.seed(42)
    return np.random.randn(100, 3).astype(np.float32)


@pytest.fixture
def identity_pose():
    """Return identity SE(3) pose as 6D vector [x, y, z, rx, ry, rz]."""
    import numpy as np
    return np.zeros(6, dtype=np.float64)


@pytest.fixture
def random_pose():
    """Generate a small random SE(3) pose for testing."""
    import numpy as np
    np.random.seed(42)
    # Small translation and rotation
    trans = np.random.randn(3) * 0.1
    rot = np.random.randn(3) * 0.05
    return np.concatenate([trans, rot])
