"""
Data structures for Geometric Compositional SLAM v2.

AtlasMap and MeasurementBatch are the canonical map and measurement types.
IW states (process/measurement noise) live in operators/structures as needed.
"""

from fl_slam_poc.backend.structures.primitive_map import (
    PrimitiveMapView,
    AtlasMapView,
    RenderablePrimitiveBatch,
    extract_primitive_map_view,
    extract_atlas_map_view,
    renderable_batch_from_view,
    primitive_map_fuse,
    primitive_map_insert,
    primitive_map_insert_masked,
    primitive_map_cull,
    primitive_map_forget,
    primitive_map_recency_inflate,
    PrimitiveMapRecencyInflateStats,
    primitive_map_merge_reduce,
    PrimitiveMapTile,
    AtlasMap,
    create_empty_tile,
    create_empty_atlas_map,
)
from fl_slam_poc.backend.structures.measurement_batch import (
    MeasurementBatch,
    create_empty_measurement_batch,
)
from fl_slam_poc.backend.structures.inverse_wishart_jax import (
    ProcessNoiseIWState,
    create_datasheet_process_noise_state,
)
from fl_slam_poc.backend.structures.measurement_noise_iw_jax import (
    MeasurementNoiseIWState,
    create_datasheet_measurement_noise_state,
)

__all__ = [
    "PrimitiveMapView",
    "AtlasMapView",
    "RenderablePrimitiveBatch",
    "extract_primitive_map_view",
    "extract_atlas_map_view",
    "renderable_batch_from_view",
    "primitive_map_fuse",
    "primitive_map_insert",
    "primitive_map_insert_masked",
    "primitive_map_cull",
    "primitive_map_forget",
    "primitive_map_recency_inflate",
    "PrimitiveMapRecencyInflateStats",
    "primitive_map_merge_reduce",
    "PrimitiveMapTile",
    "AtlasMap",
    "create_empty_tile",
    "create_empty_atlas_map",
    # Other structures
    "MeasurementBatch",
    "create_empty_measurement_batch",
    "ProcessNoiseIWState",
    "create_datasheet_process_noise_state",
    "MeasurementNoiseIWState",
    "create_datasheet_measurement_noise_state",
]
