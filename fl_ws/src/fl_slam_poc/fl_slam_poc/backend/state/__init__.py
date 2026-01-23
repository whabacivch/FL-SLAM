"""State representation and storage for backend."""

from fl_slam_poc.backend.state.anchor_manager import create_anchor, get_state_at_stamp
from fl_slam_poc.backend.state.modules import Dense3DModule, SparseAnchorModule
from fl_slam_poc.backend.state.rgbd import (
    add_dense_module,
    cull_dense_modules,
    parse_rgbd_evidence,
    process_rgbd_evidence,
)

__all__ = [
    "Dense3DModule",
    "SparseAnchorModule",
    "create_anchor",
    "get_state_at_stamp",
    "parse_rgbd_evidence",
    "process_rgbd_evidence",
    "add_dense_module",
    "cull_dense_modules",
]
