import numpy as np
import jax.numpy as jnp

from fl_slam_poc.common import constants
from fl_slam_poc.backend.structures.primitive_map import (
    PrimitiveMapTile,
    AtlasMap,
    primitive_map_merge_reduce,
)


def _make_tile(tile_id: int = 0) -> PrimitiveMapTile:
    m_tile = 3
    b = constants.GC_VMF_N_LOBES
    Lambdas = jnp.stack([jnp.eye(3), jnp.eye(3), jnp.eye(3)], axis=0)
    mu = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    thetas = jnp.einsum("nij,nj->ni", Lambdas, mu)
    etas = jnp.zeros((m_tile, b, 3), dtype=jnp.float64)
    weights = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)
    timestamps = jnp.zeros((m_tile,), dtype=jnp.float64)
    created_timestamps = jnp.zeros((m_tile,), dtype=jnp.float64)
    last_supported_scan_seq = jnp.zeros((m_tile,), dtype=jnp.int64)
    last_update_scan_seq = jnp.zeros((m_tile,), dtype=jnp.int64)
    primitive_ids = jnp.array([0, 1, 2], dtype=jnp.int64)
    valid_mask = jnp.array([True, True, True], dtype=bool)
    colors = jnp.zeros((m_tile, 3), dtype=jnp.float64)
    return PrimitiveMapTile(
        tile_id=tile_id,
        Lambdas=Lambdas,
        thetas=thetas,
        etas=etas,
        weights=weights,
        timestamps=timestamps,
        created_timestamps=created_timestamps,
        last_supported_scan_seq=last_supported_scan_seq,
        last_update_scan_seq=last_update_scan_seq,
        primitive_ids=primitive_ids,
        valid_mask=valid_mask,
        colors=colors,
        next_local_id=3,
        count=3,
    )


def test_merge_reduce_merges_close_pair():
    tile = _make_tile(tile_id=0)
    atlas = AtlasMap(tiles={0: tile}, next_global_id=3, total_count=3, m_tile=3)

    result, cert, effect = primitive_map_merge_reduce(
        atlas_map=atlas,
        tile_id=0,
        merge_threshold=0.5,
        max_pairs=1,
        max_tile_size=10,
    )

    assert result.n_merged == 1
    new_tile = result.atlas_map.tiles[0]

    weights = np.array(new_tile.weights)
    valid = np.array(new_tile.valid_mask)

    assert bool(valid[0]) is True
    assert bool(valid[1]) is False
    assert np.isclose(weights[0], 2.0)
    assert result.atlas_map.total_count == 2
    assert cert.frobenius_applied is True
    assert effect.realized == 1.0
