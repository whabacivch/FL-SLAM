import numpy as np
import jax.numpy as jnp

from fl_slam_poc.common import constants
from fl_slam_poc.backend.structures.primitive_map import (
    AtlasMap,
    create_empty_tile,
    primitive_map_insert_masked,
    primitive_map_fuse,
)


def _make_atlas(m_tile: int = 1) -> AtlasMap:
    tile = create_empty_tile(tile_id=0, m_tile=m_tile)
    return AtlasMap(tiles={0: tile}, next_global_id=0, total_count=0, m_tile=m_tile)


def _insert_single(atlas: AtlasMap, color, source) -> AtlasMap:
    b = int(constants.GC_VMF_N_LOBES)
    Lambdas_new = jnp.eye(3, dtype=jnp.float64)[None, :, :]
    thetas_new = jnp.zeros((1, 3), dtype=jnp.float64)
    etas_new = jnp.zeros((1, b, 3), dtype=jnp.float64)
    weights_new = jnp.array([1.0], dtype=jnp.float64)
    colors_new = jnp.array([color], dtype=jnp.float64)
    sources_new = jnp.array([source], dtype=jnp.int32)
    valid_new_mask = jnp.array([True], dtype=bool)
    result, _, _ = primitive_map_insert_masked(
        atlas_map=atlas,
        tile_id=0,
        Lambdas_new=Lambdas_new,
        thetas_new=thetas_new,
        etas_new=etas_new,
        weights_new=weights_new,
        timestamp=0.0,
        scan_seq=0,
        valid_new_mask=valid_new_mask,
        colors_new=colors_new,
        sources_new=sources_new,
    )
    return result.atlas_map


def _fuse_single(atlas: AtlasMap, color, source) -> AtlasMap:
    b = int(constants.GC_VMF_N_LOBES)
    target_slots = jnp.array([0], dtype=jnp.int32)
    Lambdas_meas = jnp.eye(3, dtype=jnp.float64)[None, :, :]
    thetas_meas = jnp.zeros((1, 3), dtype=jnp.float64)
    etas_meas = jnp.zeros((1, b, 3), dtype=jnp.float64)
    weights_meas = jnp.array([1.0], dtype=jnp.float64)
    responsibilities = jnp.array([1.0], dtype=jnp.float64)
    colors_meas = jnp.array([color], dtype=jnp.float64)
    sources_meas = jnp.array([source], dtype=jnp.int32)
    valid_mask = jnp.array([True], dtype=bool)
    result, _, _ = primitive_map_fuse(
        atlas_map=atlas,
        tile_id=0,
        target_slots=target_slots,
        Lambdas_meas=Lambdas_meas,
        thetas_meas=thetas_meas,
        etas_meas=etas_meas,
        weights_meas=weights_meas,
        responsibilities=responsibilities,
        timestamp=1.0,
        scan_seq=1,
        valid_mask=valid_mask,
        colors_meas=colors_meas,
        sources_meas=sources_meas,
    )
    return result.atlas_map


def test_camera_then_lidar_keeps_camera_color():
    atlas = _make_atlas()
    atlas = _insert_single(atlas, color=[1.0, 0.0, 0.0], source=0)
    atlas = _fuse_single(atlas, color=[0.2, 0.2, 0.2], source=1)
    tile = atlas.tiles[0]
    rgb = np.array(tile.rgb[0])
    assert np.allclose(rgb, np.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_lidar_then_camera_switches_to_camera_color():
    atlas = _make_atlas()
    atlas = _insert_single(atlas, color=[0.2, 0.2, 0.2], source=1)
    atlas = _fuse_single(atlas, color=[0.0, 1.0, 0.0], source=0)
    tile = atlas.tiles[0]
    rgb = np.array(tile.rgb[0])
    assert np.allclose(rgb, np.array([0.0, 1.0, 0.0]), atol=1e-6)
