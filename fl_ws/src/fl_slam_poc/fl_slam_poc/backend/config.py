"""
Backend configuration model hooks.

Provides utilities for loading, validating, and managing backend configuration
from YAML files and ROS 2 parameters.

This module bridges the gap between:
1. YAML configuration files (config/fl_slam_poc_base.yaml)
2. Pydantic validation models (common/param_models.py)
3. ROS 2 parameter system

Usage:
    from fl_slam_poc.backend.config import load_backend_config, validate_backend_params
    
    # Load from YAML
    config = load_backend_config("/path/to/config.yaml")
    
    # Validate ROS params against model
    params = validate_backend_params(node)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import yaml
from pydantic import ValidationError

from fl_slam_poc.common.param_models import BackendParams

if TYPE_CHECKING:
    from rclpy.node import Node


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Dictionary with configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries (later configs override earlier).
    
    Args:
        configs: Variable number of config dicts to merge
        
    Returns:
        Merged configuration dictionary
    """
    result: Dict[str, Any] = {}
    for config in configs:
        if config:
            _deep_merge(result, config)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Deep merge override dict into base dict (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_backend_config(
    base_path: Optional[str | Path] = None,
    preset_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> BackendParams:
    """
    Load and validate backend configuration from YAML files.
    
    Args:
        base_path: Path to base configuration YAML (fl_slam_poc_base.yaml)
        preset_path: Optional path to preset override YAML (e.g., presets/m3dgr.yaml)
        overrides: Optional dictionary of parameter overrides
        
    Returns:
        Validated BackendParams model
        
    Raises:
        ValidationError: If configuration is invalid
    """
    # Load base config
    base_config: Dict[str, Any] = {}
    if base_path:
        full_config = load_yaml_config(base_path)
        # Extract backend section
        base_config = full_config.get("fl_backend", {}).get("ros__parameters", {})
    
    # Load preset config
    preset_config: Dict[str, Any] = {}
    if preset_path:
        full_preset = load_yaml_config(preset_path)
        preset_config = full_preset.get("fl_backend", {}).get("ros__parameters", {})
    
    # Merge configs: base <- preset <- overrides
    merged = merge_configs(base_config, preset_config, overrides or {})
    
    # Validate against Pydantic model
    return BackendParams(**merged)


def validate_backend_params(node: "Node") -> BackendParams:
    """
    Validate ROS 2 node parameters against BackendParams model.
    
    Extracts all declared parameters from the node and validates them
    against the Pydantic model.
    
    Args:
        node: ROS 2 node with declared parameters
        
    Returns:
        Validated BackendParams model
        
    Raises:
        ValidationError: If parameters are invalid
    """
    values: Dict[str, Any] = {}
    for name in BackendParams.model_fields:
        if node.has_parameter(name):
            values[name] = node.get_parameter(name).value
    
    try:
        return BackendParams(**values)
    except ValidationError as exc:
        node.get_logger().error(f"Invalid backend parameters: {exc}")
        raise


def get_default_config_paths() -> tuple[Path, Path]:
    """
    Get default paths to configuration files relative to package.
    
    Returns:
        Tuple of (base_config_path, presets_dir_path)
    """
    # Find package root by traversing up from this file
    pkg_root = Path(__file__).parent.parent.parent
    base_config = pkg_root / "config" / "fl_slam_poc_base.yaml"
    presets_dir = pkg_root / "config" / "presets"
    return base_config, presets_dir


def get_preset_path(preset_name: str) -> Optional[Path]:
    """
    Get path to a preset configuration file.
    
    Args:
        preset_name: Name of preset (e.g., "m3dgr")
        
    Returns:
        Path to preset file, or None if not found
    """
    _, presets_dir = get_default_config_paths()
    preset_path = presets_dir / f"{preset_name}.yaml"
    return preset_path if preset_path.exists() else None
