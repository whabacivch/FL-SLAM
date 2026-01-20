# FL-SLAM Package Structure

## Overview

The `fl_slam_poc` package has been reorganized into a clear frontend/backend architecture following the 2026-01-20 restructure.

## Package Structure

```
fl_slam_poc/
├── frontend/                    # Sensor processing → Loop factors
│   ├── frontend_node.py         # Main frontend orchestrator
│   ├── processing/              # Sensor input & preprocessing
│   │   ├── sensor_io.py         # Sensor subscriptions, TF, PointCloud2
│   │   ├── rgbd_processor.py    # RGB-D depth processing
│   │   └── status_monitor.py    # Sensor connectivity monitoring
│   ├── loops/                   # Loop closure detection
│   │   ├── loop_processor.py    # ICP loop detection coordinator
│   │   ├── icp.py               # Iterative Closest Point solver
│   │   ├── pointcloud_gpu.py    # GPU-accelerated point cloud ops
│   │   └── vmf_geometry.py      # von Mises-Fisher directional stats
│   └── anchors/                 # Anchor management
│       ├── anchor_manager.py    # Anchor budget/creation
│       └── descriptor_builder.py # Descriptor extraction
│
├── backend/                     # Information fusion → State estimate
│   ├── backend_node.py          # Main backend ROS node (was fl_backend_node)
│   ├── fusion/                  # Information-geometric fusion
│   │   ├── gaussian_info.py     # Gaussian information form ops
│   │   ├── gaussian_geom.py     # Frobenius correction
│   │   ├── multimodal_fusion.py # Multi-sensor fusion
│   │   └── information_distances.py # Fisher-Rao distances
│   └── parameters/              # Adaptive parameter estimation (was models/)
│       ├── adaptive.py          # Online adaptive parameters
│       ├── birth.py             # Stochastic birth model
│       ├── nig.py               # Normal-Inverse-Gamma model
│       ├── process_noise.py     # Adaptive process noise
│       ├── timestamp.py         # Time alignment model
│       └── weights.py           # Weight combination utilities
│
├── common/                      # Shared by frontend & backend
│   ├── transforms/              # SE(3) transforms (was geometry/)
│   │   └── se3.py               # SE(3) operations, quaternions
│   ├── config.py                # Configuration dataclasses
│   ├── constants.py             # Constants and defaults
│   └── op_report.py             # Operation reporting for audits
│
├── utility_nodes/               # Helper/utility nodes
│   ├── tb3_odom_bridge.py       # TurtleBot3 odometry converter
│   ├── image_decompress.py      # Rosbag image decompression
│   ├── livox_converter.py       # Livox → PointCloud2 conversion
│   └── sim_world.py             # Simulation ground truth provider
│
├── nodes/                       # (Legacy) Experimental nodes
│   ├── dirichlet_backend_node.py # EXPERIMENTAL: Dirichlet semantic SLAM
│   └── sim_semantics_node.py     # EXPERIMENTAL: Semantic simulation
│
├── operators/                   # (Legacy) Re-exports + experimental
│   ├── __init__.py              # Re-exports from new locations
│   └── dirichlet_geom.py        # EXPERIMENTAL: Dirichlet geometry
│
├── models/                      # (Legacy) Re-exports for compatibility
│   └── __init__.py              # Re-exports from backend.parameters
│
├── geometry/                    # (Legacy) Re-exports for compatibility
│   ├── __init__.py              # Re-exports from common.transforms
│   └── se3.py                   # Re-exports from common.transforms.se3
│
└── utils/                       # (Legacy) Re-exports for compatibility
    └── __init__.py              # Re-exports from frontend.processing
```

## Architecture Flow

```
                    ┌─────────────────────────────────────────────────┐
                    │                   FRONTEND                       │
                    ├─────────────────────────────────────────────────┤
                    │                                                  │
Sensors ─────────►  │  processing/   ──►   loops/   ──►   anchors/   │ ──► Loop Factors
(scan, depth,       │  (sensor_io)       (ICP, vMF)    (management)   │
 camera, odom)      │                                                  │
                    └─────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────┐
                    │                   BACKEND                        │
Odometry ─────────► ├─────────────────────────────────────────────────┤
                    │                                                  │
Loop Factors ─────► │  fusion/       ──►   State Estimate + Map       │ ──► /cdwm/pose
                    │  (information       (SE(3) + covariance)        │     /cdwm/map
                    │   geometry)                                      │
                    └─────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────┐
                    │                   COMMON                         │
                    ├─────────────────────────────────────────────────┤
                    │  transforms/se3   config   constants   op_report │
                    └─────────────────────────────────────────────────┘
```

## Node Executables

| New Name | Old Name | Description |
|----------|----------|-------------|
| `frontend_node` | `frontend_node` | Main frontend orchestrator |
| `backend_node` | `fl_backend_node` | Main backend fuser |
| `tb3_odom_bridge` | `tb3_odom_bridge_node` | Odometry converter |
| `image_decompress` | `image_decompress_node` | Image decompression |
| `livox_converter` | `livox_converter_node` | Livox message converter |
| `sim_world` | `sim_world_node` | Simulation world |

Legacy names are preserved as aliases for backward compatibility.

## Import Paths

### New Paths (Recommended)

```python
# Frontend
from fl_slam_poc.frontend.processing import SensorIO, StatusMonitor
from fl_slam_poc.frontend.loops import LoopProcessor, icp_3d
from fl_slam_poc.frontend.anchors import AnchorManager

# Backend
from fl_slam_poc.backend.fusion import make_evidence, fuse_info
from fl_slam_poc.backend.parameters import NIGModel, TimeAlignmentModel

# Common
from fl_slam_poc.common.transforms.se3 import se3_compose, rotmat_to_quat
from fl_slam_poc.common import constants, config
from fl_slam_poc.common.op_report import OpReport
```

### Legacy Paths (Backward Compatible)

The following legacy import paths are preserved for compatibility:

```python
from fl_slam_poc.geometry.se3 import ...   # → common.transforms.se3
from fl_slam_poc.models import ...          # → backend.parameters
from fl_slam_poc.operators import ...       # → frontend.loops + backend.fusion
from fl_slam_poc.utils import ...           # → frontend.processing
```

## Experimental Code

The following modules are tagged as **EXPERIMENTAL** and not part of the main pipeline:

- `nodes/dirichlet_backend_node.py` - Dirichlet semantic SLAM
- `nodes/sim_semantics_node.py` - Semantic category simulation  
- `operators/dirichlet_geom.py` - Dirichlet geometry operations

These implement research directions in semantic SLAM using Dirichlet distributions.

## Test Coverage

- **113/114 tests passing** after restructure
- 1 numerical precision test (GPU ICP) has pre-existing flakiness

Run tests with:
```bash
cd fl_ws/src/fl_slam_poc
pytest test/ -v
```

## Last Updated

2026-01-20: Major restructure for frontend/backend separation
