"""
Deterministic MA-Hex 3D tiling utilities (GC v2 Phase 6).

Key design:
- Tile addressing is deterministic and unwrapped (no modulo / no global-grid wrap).
- For runtime storage (AtlasMap.tiles dict), we use a stable int tile_id derived from the
  MA-Hex 3D cell coordinates (c1, c2, cz). This keeps the rest of the Python codebase
  (and ROS messages/serialization) simple while remaining deterministic.
- For JAX-fixed-shape tensor code, higher layers should map arbitrary tile_ids to a
  per-scan batch index in [0..N_ACTIVE_TILES).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from fl_slam_poc.common.jax_init import jax, jnp

# Match the MA-Hex basis used by fl_slam_poc.common.ma_hex_web (canonical basis for cell_1/cell_2).
_A1 = np.array([1.0, 0.0], dtype=np.float64)
_A2 = np.array([0.5, 0.5 * np.sqrt(3.0)], dtype=np.float64)


# -----------------------------------------------------------------------------
# MA-Hex 3D cell coordinates
# -----------------------------------------------------------------------------


def ma_hex_cell_3d_from_xyz(xyz: np.ndarray, h_tile: float) -> Tuple[int, int, int]:
    """
    Map world xyz to MA-Hex 3D cell coordinates (c1, c2, cz).

    c1 = floor((a1·[x,y]) / h_tile), c2 = floor((a2·[x,y]) / h_tile), cz = floor(z / h_tile).

    No modulo wrapping (atlas keys must not wrap).
    """
    xyz = np.asarray(xyz, dtype=np.float64).ravel()
    if xyz.shape[0] < 3:
        raise ValueError(f"ma_hex_cell_3d_from_xyz: expected xyz (3,), got shape {xyz.shape}")
    h = max(float(h_tile), 1e-12)
    y = xyz[:2]
    s1 = float(_A1 @ y)
    s2 = float(_A2 @ y)
    c1 = int(np.floor(s1 / h))
    c2 = int(np.floor(s2 / h))
    cz = int(np.floor(float(xyz[2]) / h))
    return (c1, c2, cz)


def ma_hex_cell_3d_from_xyz_batch(XYZ: np.ndarray, h_tile: float) -> np.ndarray:
    """(N,3) xyz -> (N,3) int64 MA-Hex 3D cell coords."""
    XYZ = np.asarray(XYZ, dtype=np.float64).reshape(-1, 3)
    h = max(float(h_tile), 1e-12)
    Y = XYZ[:, :2]
    s1 = Y @ _A1
    s2 = Y @ _A2
    c1 = np.floor(s1 / h).astype(np.int64)
    c2 = np.floor(s2 / h).astype(np.int64)
    cz = np.floor(XYZ[:, 2] / h).astype(np.int64)
    return np.column_stack([c1, c2, cz])


# -----------------------------------------------------------------------------
# Stable packing to an int tile_id for storage/logging
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PackedTileIdSpec:
    """
    Bit packing for (c1,c2,cz) -> signed int tile_id.

    We allocate BITS_PER_AXIS bits per coordinate after a fixed bias.
    This keeps tile_id stable/deterministic across runs.
    """

    BITS_PER_AXIS: int = 21  # 3*21 = 63 fits in signed int64.
    BIAS: int = 1 << 20

    @property
    def MASK(self) -> int:
        return (1 << self.BITS_PER_AXIS) - 1


_PACK_SPEC = PackedTileIdSpec()


def tile_id_from_cell_3d(c1: int, c2: int, cz: int, spec: PackedTileIdSpec = _PACK_SPEC) -> int:
    """
    Pack (c1,c2,cz) into a single signed int (fits in int64).

    This is a deterministic storage key. It is NOT intended for arithmetic.
    """
    b = int(spec.BIAS)
    m = int(spec.MASK)
    # Bias to non-negative then mask.
    u1 = (int(c1) + b) & m
    u2 = (int(c2) + b) & m
    uz = (int(cz) + b) & m
    tile_id = (u1 << (2 * spec.BITS_PER_AXIS)) | (u2 << spec.BITS_PER_AXIS) | uz
    # Keep as Python int; downstream can cast to int64 for serialization if desired.
    return int(tile_id)


def tile_id_from_xyz(xyz: np.ndarray, h_tile: float, spec: PackedTileIdSpec = _PACK_SPEC) -> int:
    c1, c2, cz = ma_hex_cell_3d_from_xyz(xyz, h_tile=h_tile)
    return tile_id_from_cell_3d(c1, c2, cz, spec=spec)


def tile_ids_from_xyz_batch(XYZ: np.ndarray, h_tile: float, spec: PackedTileIdSpec = _PACK_SPEC) -> np.ndarray:
    """(N,3) xyz -> (N,) int64 packed tile_ids (deterministic; unwrapped)."""
    cells = ma_hex_cell_3d_from_xyz_batch(XYZ, h_tile=h_tile)  # (N,3) int64
    # Vectorized packing in Python/NumPy for stability and simplicity.
    b = int(spec.BIAS)
    m = int(spec.MASK)
    u1 = (cells[:, 0].astype(np.int64) + b) & m
    u2 = (cells[:, 1].astype(np.int64) + b) & m
    uz = (cells[:, 2].astype(np.int64) + b) & m
    tile_ids = (u1 << (2 * spec.BITS_PER_AXIS)) | (u2 << spec.BITS_PER_AXIS) | uz
    return tile_ids.astype(np.int64)


def tile_ids_from_xyz_batch_jax(
    XYZ: jnp.ndarray,
    h_tile: float,
    spec: PackedTileIdSpec = _PACK_SPEC,
) -> jnp.ndarray:
    """JAX: (N,3) xyz -> (N,) int64 packed tile_ids (deterministic; unwrapped)."""
    XYZ = jnp.asarray(XYZ, dtype=jnp.float64).reshape(-1, 3)
    h = jnp.maximum(jnp.asarray(h_tile, dtype=jnp.float64), 1e-12)
    s1 = XYZ[:, 0]
    s2 = XYZ[:, 0] * 0.5 + XYZ[:, 1] * (jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float64)) * 0.5)
    sz = XYZ[:, 2]
    c1 = jnp.floor(s1 / h).astype(jnp.int64)
    c2 = jnp.floor(s2 / h).astype(jnp.int64)
    cz = jnp.floor(sz / h).astype(jnp.int64)
    b = jnp.asarray(spec.BIAS, dtype=jnp.int64)
    m = jnp.asarray(spec.MASK, dtype=jnp.int64)
    u1 = (c1 + b) & m
    u2 = (c2 + b) & m
    uz = (cz + b) & m
    return (u1 << (2 * spec.BITS_PER_AXIS)) | (u2 << spec.BITS_PER_AXIS) | uz


def tile_ids_from_cells_jax(
    c1: jnp.ndarray,
    c2: jnp.ndarray,
    cz: jnp.ndarray,
    spec: PackedTileIdSpec = _PACK_SPEC,
) -> jnp.ndarray:
    """JAX: pack MA-Hex 3D cell coords into int64 tile_ids."""
    c1 = jnp.asarray(c1, dtype=jnp.int64)
    c2 = jnp.asarray(c2, dtype=jnp.int64)
    cz = jnp.asarray(cz, dtype=jnp.int64)
    b = jnp.asarray(spec.BIAS, dtype=jnp.int64)
    m = jnp.asarray(spec.MASK, dtype=jnp.int64)
    u1 = (c1 + b) & m
    u2 = (c2 + b) & m
    uz = (cz + b) & m
    return (u1 << (2 * spec.BITS_PER_AXIS)) | (u2 << spec.BITS_PER_AXIS) | uz


# -----------------------------------------------------------------------------
# Deterministic stencil enumeration
# -----------------------------------------------------------------------------


def hex_disk_axial(radius: int) -> List[Tuple[int, int]]:
    """
    Return axial coords (q,r) in a hex disk of radius r (inclusive), deterministic order.

    Standard axial disk: for q in [-r..r], for r in [max(-r, -q-r)..min(r, -q+r)].
    Deterministic order is sorted (q,r).
    """
    r = int(radius)
    out: List[Tuple[int, int]] = []
    for q in range(-r, r + 1):
        r_min = max(-r, -q - r)
        r_max = min(r, -q + r)
        for rr in range(r_min, r_max + 1):
            out.append((q, rr))
    out.sort()
    return out


def ma_hex_stencil_tile_ids(
    center_xyz: np.ndarray,
    h_tile: float,
    radius_xy: int,
    radius_z: int,
    spec: PackedTileIdSpec = _PACK_SPEC,
) -> List[int]:
    """
    Build a deterministic list of tile_ids for a 3D stencil centered at center_xyz.

    The XY stencil is a hex disk in (c1,c2) axial coords; Z is a symmetric slab.
    Order is deterministic: sort by (cz, c1, c2) via sorted axial disk + increasing cz.
    """
    c1, c2, cz = ma_hex_cell_3d_from_xyz(center_xyz, h_tile=h_tile)
    disk = hex_disk_axial(radius_xy)
    ids: List[int] = []
    for dz in range(-int(radius_z), int(radius_z) + 1):
        z = cz + dz
        for dq, dr in disk:
            ids.append(tile_id_from_cell_3d(c1 + dq, c2 + dr, z, spec=spec))
    return ids
