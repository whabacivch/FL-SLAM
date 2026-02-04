"""
PrimitiveMap: Probabilistic primitive atlas for Geometric Compositional SLAM v2.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md

Each primitive j in the map has:
- Geometry: Gaussian in info form (Lambda_j, theta_j) in 3D; optional cached (mu_j, Sigma_j)
- Orientation/appearance: vMF natural parameter(s) eta_j (resultant or B=3)
- Optional payload: color/descriptor summary
- Stable ID + spatial index membership

Map maintenance operators:
- PrimitiveMapInsert: new primitives enter the map
- PrimitiveMapFuse: PoE + Wishart; fuse associated measurement primitives
- PrimitiveMapCull: compute budget operator; mass drop logged as approximation
- PrimitiveMapMergeReduce: mixture reduction with CertBundle + Frobenius
- PrimitiveMapForget: continuous forgetting factor (no if/else)

All operators are fixed-cost and applied every scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import CertBundle, ExpectedEffect, InfluenceCert
from fl_slam_poc.common.primitives import domain_projection_psd_core


# =============================================================================
# Primitive Data Structure
# =============================================================================


@dataclass
class Primitive:
    """
    Single probabilistic primitive in info form.

    Geometry (3D Gaussian):
        Lambda: (3, 3) precision matrix (info form)
        theta: (3,) information vector (= Lambda @ mu)

    Orientation (vMF):
        eta: (3,) natural parameter (= kappa * mu_dir)

    Metadata:
        primitive_id: Stable unique identifier for temporal tracking
        weight: Accumulated mass/ESS (for culling)
        timestamp: Last update time (for staleness)
    """
    # Geometry (Gaussian in info form)
    Lambda: jnp.ndarray  # (3, 3) precision matrix
    theta: jnp.ndarray   # (3,) information vector

    # Orientation/appearance (multi-lobe vMF natural parameters, B=GC_VMF_N_LOBES)
    etas: jnp.ndarray    # (B, 3) = kappa_b * mu_b; resultant eta = sum over lobes

    # Metadata
    primitive_id: int    # Stable unique ID
    weight: float        # Accumulated mass/ESS
    timestamp: float     # Last update time (seconds)

    # Optional: color payload (RGB, normalized)
    color: Optional[jnp.ndarray] = None  # (3,) RGB in [0, 1]

    def mean_position(self, eps_lift: float = constants.GC_EPS_LIFT) -> jnp.ndarray:
        """Compute mean position mu = Lambda^{-1} @ theta."""
        Lambda_reg = self.Lambda + eps_lift * jnp.eye(3, dtype=jnp.float64)
        return jnp.linalg.solve(Lambda_reg, self.theta)

    def covariance(self, eps_lift: float = constants.GC_EPS_LIFT) -> jnp.ndarray:
        """Compute covariance Sigma = Lambda^{-1}."""
        Lambda_reg = self.Lambda + eps_lift * jnp.eye(3, dtype=jnp.float64)
        return jnp.linalg.inv(Lambda_reg)

    def kappa(self) -> float:
        """Resultant vMF concentration kappa = ||sum_b eta_b||."""
        eta_sum = jnp.sum(self.etas, axis=0)
        return float(jnp.linalg.norm(eta_sum))

    def mean_direction(self, eps_mass: float = constants.GC_EPS_MASS) -> jnp.ndarray:
        """Resultant mean direction mu_dir = eta_sum / ||eta_sum||."""
        eta_sum = jnp.sum(self.etas, axis=0)
        norm = jnp.linalg.norm(eta_sum)
        return eta_sum / (norm + eps_mass)


# =============================================================================
# PrimitiveMapTile: Single Tile in the Atlas (Phase 2 foundation)
# =============================================================================


@dataclass
class PrimitiveMapTile:
    """
    Single tile in the primitive atlas.

    Phase 2 foundation: tiles enable spatial partitioning for scalability.
    Each tile has fixed capacity M_TILE; tile addressing is (tile_id, slot).

    Multi-tile support (Phase 6) uses deterministic MA-Hex tile_ids.

    Attributes:
        tile_id: Unique tile identifier (0 for single-tile mode)
        Lambdas: (M_TILE, 3, 3) precision matrices
        thetas: (M_TILE, 3) information vectors
        etas: (M_TILE, B, 3) vMF natural parameters
        weights: (M_TILE,) accumulated mass/ESS
        timestamps: (M_TILE,) last update times
        created_timestamps: (M_TILE,) creation times
        primitive_ids: (M_TILE,) stable unique IDs
        valid_mask: (M_TILE,) bool mask for valid entries
        colors: (M_TILE, 3) RGB colors
        next_local_id: Next available slot index (local to tile)
        count: Number of valid primitives in this tile
    """
    tile_id: int
    Lambdas: jnp.ndarray      # (M_TILE, 3, 3)
    thetas: jnp.ndarray       # (M_TILE, 3)
    etas: jnp.ndarray         # (M_TILE, B, 3)
    weights: jnp.ndarray      # (M_TILE,)
    timestamps: jnp.ndarray   # (M_TILE,)
    created_timestamps: jnp.ndarray  # (M_TILE,)
    last_supported_scan_seq: jnp.ndarray  # (M_TILE,) int64
    last_update_scan_seq: jnp.ndarray  # (M_TILE,) int64
    primitive_ids: jnp.ndarray  # (M_TILE,) int64
    valid_mask: jnp.ndarray   # (M_TILE,) bool
    colors: jnp.ndarray       # (M_TILE, 3)
    next_local_id: int        # Next available slot in this tile
    count: int                # Number of valid primitives in this tile


def create_empty_tile(
    tile_id: int,
    m_tile: int = constants.GC_PRIMITIVE_MAP_MAX_SIZE,
) -> PrimitiveMapTile:
    """Create empty tile with fixed-size arrays."""
    return PrimitiveMapTile(
        tile_id=tile_id,
        Lambdas=jnp.zeros((m_tile, 3, 3), dtype=jnp.float64),
        thetas=jnp.zeros((m_tile, 3), dtype=jnp.float64),
        etas=jnp.zeros((m_tile, constants.GC_VMF_N_LOBES, 3), dtype=jnp.float64),
        weights=jnp.zeros((m_tile,), dtype=jnp.float64),
        timestamps=jnp.zeros((m_tile,), dtype=jnp.float64),
        created_timestamps=jnp.zeros((m_tile,), dtype=jnp.float64),
        last_supported_scan_seq=jnp.zeros((m_tile,), dtype=jnp.int64),
        last_update_scan_seq=jnp.zeros((m_tile,), dtype=jnp.int64),
        primitive_ids=jnp.zeros((m_tile,), dtype=jnp.int64),
        valid_mask=jnp.zeros((m_tile,), dtype=bool),
        colors=jnp.zeros((m_tile, 3), dtype=jnp.float64),
        next_local_id=0,
        count=0,
    )


# =============================================================================
# AtlasMap: Multi-Tile Container (Phase 2 foundation)
# =============================================================================


@dataclass
class AtlasMap:
    """
    Atlas of primitive map tiles.

    AtlasMap wraps multiple PrimitiveMapTiles for spatial scalability.

    Addressing: (tile_id, slot) tuples identify primitives globally.
    Multi-tile mode (Phase 6) uses MA-Hex for deterministic tile selection.

    Attributes:
        tiles: Dict mapping tile_id -> PrimitiveMapTile
        next_global_id: Next available global primitive ID
        total_count: Total primitives across all tiles
        m_tile: Capacity per tile (fixed at creation)
    """
    tiles: dict  # Dict[int, PrimitiveMapTile]
    next_global_id: int
    total_count: int
    m_tile: int

    @property
    def n_tiles(self) -> int:
        """Number of active tiles."""
        return len(self.tiles)

    @property
    def tile_ids(self) -> List[int]:
        """List of active tile IDs."""
        return list(self.tiles.keys())


def create_empty_atlas_map(
    m_tile: int = constants.GC_PRIMITIVE_MAP_MAX_SIZE,
) -> AtlasMap:
    """
    Create an empty atlas (no tiles yet).

    Phase 6: Tiles are created on-demand by deterministic MA-Hex tile_id.
    """
    return AtlasMap(
        tiles={},
        next_global_id=0,
        total_count=0,
        m_tile=m_tile,
    )


# =============================================================================
# PrimitiveMapView: Read-only view for rendering/association (tile-local)
# =============================================================================


@dataclass
class PrimitiveMapView:
    """
    Read-only view of a single PrimitiveMapTile for rendering and association.

    Contains only the data needed for downstream operations, with optional
    downselection for compute budgeting. Addressing is tile-local: slot_indices
    are indices into the tile arrays.
    """
    tile_id: int
    m_tile: int
    slot_indices: jnp.ndarray  # (N,) indices into tile arrays

    # Positions (mean of Gaussian)
    positions: jnp.ndarray    # (N, 3) mean positions
    covariances: jnp.ndarray  # (N, 3, 3) covariances (Sigma = Lambda^{-1})

    # Directions (mean of vMF)
    directions: jnp.ndarray   # (N, 3) mean directions
    kappas: jnp.ndarray       # (N,) concentrations

    # Weights and metadata
    weights: jnp.ndarray      # (N,) accumulated mass
    primitive_ids: jnp.ndarray  # (N,) stable IDs

    # Optional: colors for rendering
    colors: Optional[jnp.ndarray] = None  # (N, 3)
    etas: Optional[jnp.ndarray] = None  # (N, B, 3) vMF natural params

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


@dataclass
class AtlasMapView:
    """
    Fixed-size stitched view over multiple tiles (for association).

    This is the Phase 6 "candidate pool" surface:
      - Provide a fixed-size pool of primitives from a fixed stencil of tiles.
      - Preserve tile-local addressing via parallel arrays (tile_id, slot).
      - Association uses valid_mask (no gating; continuous weights only).
    """

    # Candidate pool addressing
    candidate_tile_ids: jnp.ndarray  # (M_pool,) int64 (packed tile_id)
    candidate_slots: jnp.ndarray  # (M_pool,) int32 (tile-local slot index)
    valid_mask: jnp.ndarray  # (M_pool,) bool
    tile_ids: jnp.ndarray  # (N_tiles,) int64 in atlas view order
    m_tile_view: int  # Fixed per-tile view size

    # Geometry / appearance for OT cost
    positions: jnp.ndarray  # (M_pool, 3)
    covariances: jnp.ndarray  # (M_pool, 3, 3)
    directions: jnp.ndarray  # (M_pool, 3)
    kappas: jnp.ndarray  # (M_pool,)
    weights: jnp.ndarray  # (M_pool,)
    primitive_ids: jnp.ndarray  # (M_pool,) int64
    last_supported_scan_seq: jnp.ndarray  # (M_pool,) int64
    etas: jnp.ndarray  # (M_pool, B, 3)
    colors: jnp.ndarray  # (M_pool, 3)

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


@jax.jit(static_argnames=("k",))
def _select_topk_slots_fixed(
    weights: jnp.ndarray, primitive_ids: jnp.ndarray, valid_mask: jnp.ndarray, k: int
) -> jnp.ndarray:
    """
    Select top-k slots by weight from a fixed-size tile.

    Fixed-cost: always sorts over full M_TILE, then slices k.
    """
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)
    primitive_ids = jnp.asarray(primitive_ids, dtype=jnp.int64).reshape(-1)
    valid_mask = jnp.asarray(valid_mask, dtype=bool).reshape(-1)
    # Use a large negative sentinel so invalid entries never win.
    score = jnp.where(valid_mask, weights, jnp.asarray(-1e30, dtype=jnp.float64))
    idx = jnp.arange(score.shape[0], dtype=jnp.int32)
    # Deterministic tie-break by primitive_id (ascending).
    score_sorted, id_sorted, idx_sorted = jax.lax.sort((-score, primitive_ids, idx), dimension=0)
    _ = score_sorted, id_sorted
    _ = score_sorted
    return idx_sorted[: k].astype(jnp.int32)


@jax.jit(static_argnames=("k",))
def _select_lowest_mass_slots_fixed(
    weights: jnp.ndarray,
    primitive_ids: jnp.ndarray,
    valid_mask: jnp.ndarray,
    last_supported_scan_seq: jnp.ndarray,
    scan_seq: int,
    recency_decay_lambda: float,
    k: int,
) -> jnp.ndarray:
    """
    Select lowest-retention slots with deterministic tie-break by primitive_id.

    Empty slots are treated as mass = -inf so they are selected first.
    """
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)
    primitive_ids = jnp.asarray(primitive_ids, dtype=jnp.int64).reshape(-1)
    valid_mask = jnp.asarray(valid_mask, dtype=bool).reshape(-1)
    last_supported_scan_seq = jnp.asarray(last_supported_scan_seq, dtype=jnp.int64).reshape(-1)
    scan_seq = jnp.asarray(scan_seq, dtype=jnp.int64)
    dt = jnp.maximum(jnp.asarray(0, dtype=jnp.int64), scan_seq - last_supported_scan_seq)
    decay = jnp.exp(-jnp.asarray(recency_decay_lambda, dtype=jnp.float64) * dt.astype(jnp.float64))
    retention = weights * decay
    mass_key = jnp.where(valid_mask, retention, jnp.asarray(-jnp.inf, dtype=jnp.float64))
    idx = jnp.arange(mass_key.shape[0], dtype=jnp.int32)
    # Sort by (mass_key asc, primitive_id asc, idx asc) deterministically.
    mass_sorted, id_sorted, idx_sorted = jax.lax.sort((mass_key, primitive_ids, idx), dimension=0)
    _ = mass_sorted, id_sorted
    return idx_sorted[: k].astype(jnp.int32)


def extract_atlas_map_view(
    atlas_map: AtlasMap,
    tile_ids: List[int],
    m_tile_view: int,
    eps_lift: float = constants.GC_EPS_LIFT,
    eps_mass: float = constants.GC_EPS_MASS,
) -> AtlasMapView:
    """
    Extract a fixed-size candidate pool from a fixed list of tiles.

    For each tile_id in tile_ids:
      - If missing, treat it as an empty tile.
      - Select top M_TILE_VIEW slots by weight (fixed-cost).
      - Stitch into a single pool of size len(tile_ids) * M_TILE_VIEW.
    """
    if m_tile_view <= 0:
        raise ValueError(f"extract_atlas_map_view: m_tile_view must be > 0, got {m_tile_view}")

    tiles: List[PrimitiveMapTile] = []
    for tid in tile_ids:
        tile = atlas_map.tiles.get(int(tid))
        if tile is None:
            tile = create_empty_tile(tile_id=int(tid), m_tile=int(atlas_map.m_tile))
        tiles.append(tile)

    k = int(m_tile_view)
    eps_lift_j = jnp.asarray(eps_lift, dtype=jnp.float64)
    eps_mass_j = jnp.asarray(eps_mass, dtype=jnp.float64)

    # Gather per-tile fixed views then stitch.
    cand_tile_ids = []
    cand_slots = []
    valid_masks = []
    Lambdas_list = []
    thetas_list = []
    etas_list = []
    weights_list = []
    prim_ids_list = []
    last_supported_list = []
    colors_list = []

    for tile in tiles:
        slots = _select_topk_slots_fixed(tile.weights, tile.primitive_ids, tile.valid_mask, k=k)
        cand_slots.append(slots)
        cand_tile_ids.append(jnp.full((k,), int(tile.tile_id), dtype=jnp.int64))
        valid_masks.append(tile.valid_mask[slots])
        Lambdas_list.append(tile.Lambdas[slots])
        thetas_list.append(tile.thetas[slots])
        etas_list.append(tile.etas[slots])
        weights_list.append(tile.weights[slots])
        prim_ids_list.append(tile.primitive_ids[slots])
        last_supported_list.append(tile.last_supported_scan_seq[slots])
        colors_list.append(tile.colors[slots])

    Lambdas = jnp.concatenate(Lambdas_list, axis=0)
    thetas = jnp.concatenate(thetas_list, axis=0)
    etas = jnp.concatenate(etas_list, axis=0)
    weights = jnp.concatenate(weights_list, axis=0)
    primitive_ids = jnp.concatenate(prim_ids_list, axis=0)
    last_supported_scan_seq = jnp.concatenate(last_supported_list, axis=0)
    colors = jnp.concatenate(colors_list, axis=0)
    valid_mask = jnp.concatenate(valid_masks, axis=0).astype(bool)
    candidate_tile_ids = jnp.concatenate(cand_tile_ids, axis=0)
    candidate_slots = jnp.concatenate(cand_slots, axis=0)

    positions, covariances, directions, kappas, weights_out, prim_ids_out, colors_out, etas_out = (
        _extract_primitive_map_view_core(
            Lambdas,
            thetas,
            etas,
            weights,
            primitive_ids,
            colors,
            valid_mask,
            eps_lift_j,
            eps_mass_j,
        )
    )

    return AtlasMapView(
        candidate_tile_ids=candidate_tile_ids,
        candidate_slots=candidate_slots,
        valid_mask=valid_mask,
        tile_ids=jnp.asarray(tile_ids, dtype=jnp.int64),
        m_tile_view=int(m_tile_view),
        positions=positions,
        covariances=covariances,
        directions=directions,
        kappas=kappas,
        weights=weights_out,
        primitive_ids=prim_ids_out,
        last_supported_scan_seq=last_supported_scan_seq,
        etas=etas_out,
        colors=colors_out,
    )


@dataclass
class RenderablePrimitiveBatch:
    """
    Canonical renderable primitive batch (world frame).

    Fields required by Phase 4:
      - mu_world: (N, 3)
      - Sigma_world: (N, 3, 3) or Lambda_world: (N, 3, 3)
      - eta: (N, B, 3) vMF natural parameters (multi-lobe)
      - mass: (N,)
      - color: (N, 3)
    """
    mu_world: np.ndarray
    Sigma_world: Optional[np.ndarray]
    Lambda_world: Optional[np.ndarray]
    eta: np.ndarray
    mass: np.ndarray
    color: np.ndarray
    primitive_ids: Optional[np.ndarray] = None


@jax.jit
def _extract_primitive_map_view_core(
    Lambdas: jnp.ndarray,
    thetas: jnp.ndarray,
    etas: jnp.ndarray,
    weights: jnp.ndarray,
    primitive_ids: jnp.ndarray,
    colors: jnp.ndarray,
    valid_mask: jnp.ndarray,
    eps_lift: jnp.ndarray,
    eps_mass: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT'd core for extracting PrimitiveMapView from batched arrays."""
    # Compute means and covariances from info form
    Lambda_reg = Lambdas + eps_lift * jnp.eye(3, dtype=jnp.float64)[None, :, :]
    # Vectorized solve: mu_i = Lambda_i^{-1} @ theta_i
    positions = jax.vmap(jnp.linalg.solve)(Lambda_reg, thetas)
    covariances = jax.vmap(jnp.linalg.inv)(Lambda_reg)

    # Compute resultant directions and kappas from multi-lobe vMF
    eta_sum = jnp.sum(etas, axis=1)  # (N, 3)
    kappas = jnp.linalg.norm(eta_sum, axis=1)
    directions = eta_sum / (kappas[:, None] + eps_mass)

    return positions, covariances, directions, kappas, weights, primitive_ids, colors, etas


def extract_primitive_map_view(
    tile: PrimitiveMapTile,
    max_primitives: Optional[int] = None,
    eps_lift: float = constants.GC_EPS_LIFT,
    eps_mass: float = constants.GC_EPS_MASS,
) -> PrimitiveMapView:
    """
    Extract read-only view from a single PrimitiveMapTile.

    Optionally downselects to top max_primitives by weight for compute budgeting.
    This is a declared budgeting operation (not hidden).

    Args:
        tile: Source primitive map tile
        max_primitives: Optional limit on primitives (by weight)
        eps_lift: Regularization for matrix inversion
        eps_mass: Regularization for direction normalization

    Returns:
        PrimitiveMapView with positions, covariances, directions, kappas, etc.
    """
    if tile.count == 0:
        return PrimitiveMapView(
            tile_id=int(tile.tile_id),
            m_tile=int(tile.Lambdas.shape[0]),
            slot_indices=jnp.zeros((0,), dtype=jnp.int32),
            positions=jnp.zeros((0, 3), dtype=jnp.float64),
            covariances=jnp.zeros((0, 3, 3), dtype=jnp.float64),
            directions=jnp.zeros((0, 3), dtype=jnp.float64),
            kappas=jnp.zeros((0,), dtype=jnp.float64),
            weights=jnp.zeros((0,), dtype=jnp.float64),
            primitive_ids=jnp.zeros((0,), dtype=jnp.int64),
            colors=jnp.zeros((0, 3), dtype=jnp.float64),
            etas=jnp.zeros((0, constants.GC_VMF_N_LOBES, 3), dtype=jnp.float64),
        )

    # Get valid indices
    slot_indices = jnp.where(tile.valid_mask, size=tile.count)[0]

    # Optional downselection by weight
    if max_primitives is not None and tile.count > max_primitives:
        # Sort by weight (descending) and take top max_primitives
        weights_valid = tile.weights[slot_indices]
        top_indices = jnp.argsort(-weights_valid)[:max_primitives]
        slot_indices = slot_indices[top_indices]

    # Extract valid entries
    Lambdas = tile.Lambdas[slot_indices]
    thetas = tile.thetas[slot_indices]
    etas = tile.etas[slot_indices]
    weights = tile.weights[slot_indices]
    primitive_ids = tile.primitive_ids[slot_indices]
    colors = tile.colors[slot_indices]
    valid_mask = jnp.ones(len(slot_indices), dtype=bool)

    eps_lift_j = jnp.asarray(eps_lift, dtype=jnp.float64)
    eps_mass_j = jnp.asarray(eps_mass, dtype=jnp.float64)

    positions, covariances, directions, kappas, weights, prim_ids, colors, etas_out = \
        _extract_primitive_map_view_core(
            Lambdas, thetas, etas, weights, primitive_ids, colors, valid_mask,
            eps_lift_j, eps_mass_j,
        )

    return PrimitiveMapView(
        tile_id=int(tile.tile_id),
        m_tile=int(tile.Lambdas.shape[0]),
        slot_indices=slot_indices.astype(jnp.int32),
        positions=positions,
        covariances=covariances,
        directions=directions,
        kappas=kappas,
        weights=weights,
        primitive_ids=prim_ids,
        colors=colors,
        etas=etas_out,
    )


def renderable_batch_from_view(
    view: PrimitiveMapView,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> RenderablePrimitiveBatch:
    """
    Build RenderablePrimitiveBatch from a PrimitiveMapView (world frame).
    """
    mu_world = np.asarray(view.positions, dtype=np.float64)
    Sigma_world = np.asarray(view.covariances, dtype=np.float64)
    eta = (
        np.asarray(view.etas, dtype=np.float64)
        if view.etas is not None
        else np.zeros((mu_world.shape[0], constants.GC_VMF_N_LOBES, 3), dtype=np.float64)
    )
    mass = np.asarray(view.weights, dtype=np.float64)
    color = (
        np.asarray(view.colors, dtype=np.float64)
        if view.colors is not None
        else np.zeros((mu_world.shape[0], 3), dtype=np.float64)
    )
    Lambda_world = None
    if Sigma_world.size:
        reg = eps_lift * np.eye(3, dtype=np.float64)[None, :, :]
        Sigma_reg = Sigma_world + reg
        try:
            Lambda_world = np.linalg.inv(Sigma_reg)
        except np.linalg.LinAlgError:
            Lambda_world = None
    return RenderablePrimitiveBatch(
        mu_world=mu_world,
        Sigma_world=Sigma_world,
        Lambda_world=Lambda_world,
        eta=eta,
        mass=mass,
        color=color,
        primitive_ids=np.asarray(view.primitive_ids, dtype=np.int64),
    )


# =============================================================================
# Map Maintenance Operators
# =============================================================================


@dataclass
class PrimitiveMapInsertResult:
    """Result of PrimitiveMapInsert operator."""
    atlas_map: AtlasMap
    tile_id: int
    n_inserted: int
    new_ids: jnp.ndarray  # IDs assigned to new primitives


def primitive_map_insert(
    atlas_map: AtlasMap,
    tile_id: int,
    Lambdas_new: jnp.ndarray,   # (M, 3, 3)
    thetas_new: jnp.ndarray,    # (M, 3)
    etas_new: jnp.ndarray,      # (M, B, 3)
    weights_new: jnp.ndarray,   # (M,)
    timestamp: float,
    scan_seq: int = 0,
    colors_new: Optional[jnp.ndarray] = None,  # (M, 3)
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map",
) -> Tuple[PrimitiveMapInsertResult, CertBundle, ExpectedEffect]:
    """
    Insert new primitives into a map tile.

    Fixed-cost operator. If map is full, returns without inserting
    (caller should call cull first).

    Args:
        atlas_map: Current atlas map
        tile_id: Target tile ID
        Lambdas_new: Precision matrices for new primitives
        thetas_new: Information vectors
        etas_new: vMF natural parameters
        weights_new: Initial weights
        timestamp: Current time
        colors_new: Optional RGB colors

    Returns:
        (result, CertBundle, ExpectedEffect)
    """
    tile_id = int(tile_id)
    tile = atlas_map.tiles.get(tile_id)
    if tile is None:
        tile = create_empty_tile(tile_id=tile_id, m_tile=int(atlas_map.m_tile))
    M = Lambdas_new.shape[0]
    max_size = tile.Lambdas.shape[0]
    available = max_size - tile.count
    n_to_insert = min(M, available)

    if n_to_insert == 0:
        # No space - return unchanged
        result = PrimitiveMapInsertResult(
            atlas_map=atlas_map,
            tile_id=tile_id,
            n_inserted=0,
            new_ids=jnp.array([], dtype=jnp.int64),
        )
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(
            objective_name="primitive_map_insert",
            predicted=float(M),
            realized=0.0,
        )
        return result, cert, effect

    # Find empty slots
    empty_indices = jnp.where(~tile.valid_mask, size=n_to_insert)[0]

    # Assign new IDs
    new_ids = jnp.arange(
        atlas_map.next_global_id,
        atlas_map.next_global_id + n_to_insert,
        dtype=jnp.int64,
    )

    # Update arrays
    Lambdas = tile.Lambdas.at[empty_indices].set(Lambdas_new[:n_to_insert])
    thetas = tile.thetas.at[empty_indices].set(thetas_new[:n_to_insert])
    etas = tile.etas.at[empty_indices].set(etas_new[:n_to_insert])
    weights = tile.weights.at[empty_indices].set(weights_new[:n_to_insert])
    timestamps = tile.timestamps.at[empty_indices].set(timestamp)
    created_timestamps = tile.created_timestamps.at[empty_indices].set(timestamp)
    last_supported_scan_seq = tile.last_supported_scan_seq.at[empty_indices].set(
        jnp.asarray(scan_seq, dtype=jnp.int64)
    )
    last_update_scan_seq = tile.last_update_scan_seq.at[empty_indices].set(
        jnp.asarray(scan_seq, dtype=jnp.int64)
    )
    primitive_ids = tile.primitive_ids.at[empty_indices].set(new_ids)
    valid_mask = tile.valid_mask.at[empty_indices].set(True)

    if colors_new is not None:
        colors = tile.colors.at[empty_indices].set(colors_new[:n_to_insert])
    else:
        colors = tile.colors

    new_tile = PrimitiveMapTile(
        tile_id=tile.tile_id,
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
        next_local_id=max(int(tile.next_local_id), int(jnp.max(empty_indices)) + 1),
        count=tile.count + n_to_insert,
    )
    tiles = dict(atlas_map.tiles)
    tiles[tile_id] = new_tile
    new_atlas = AtlasMap(
        tiles=tiles,
        next_global_id=atlas_map.next_global_id + n_to_insert,
        total_count=atlas_map.total_count + n_to_insert,
        m_tile=atlas_map.m_tile,
    )

    result = PrimitiveMapInsertResult(
        atlas_map=new_atlas,
        tile_id=tile_id,
        n_inserted=n_to_insert,
        new_ids=new_ids,
    )
    cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
    effect = ExpectedEffect(
        objective_name="primitive_map_insert",
        predicted=float(M),
        realized=float(n_to_insert),
    )
    return result, cert, effect


def primitive_map_insert_masked(
    atlas_map: AtlasMap,
    tile_id: int,
    Lambdas_new: jnp.ndarray,   # (K, 3, 3)
    thetas_new: jnp.ndarray,    # (K, 3)
    etas_new: jnp.ndarray,      # (K, B, 3)
    weights_new: jnp.ndarray,   # (K,)
    timestamp: float,
    valid_new_mask: jnp.ndarray,  # (K,) bool; proposals with mask=0 are ignored (fixed-cost).
    scan_seq: int = 0,
    recency_decay_lambda: float = constants.GC_RECENCY_DECAY_LAMBDA,
    colors_new: Optional[jnp.ndarray] = None,  # (K, 3)
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map_insert_masked",
) -> Tuple[PrimitiveMapInsertResult, CertBundle, ExpectedEffect]:
    """
    Fixed-cost insertion of up to K new primitives into a tile, using a boolean proposal mask.

    This is the Phase 6 insertion surface: callers pass a fixed K=K_INSERT_TILE proposals
    and a mask indicating which are real. The operator inserts only masked proposals, and
    logs any truncation due to lack of free slots as an explicit budgeting approximation.
    """
    tile_id = int(tile_id)
    tile = atlas_map.tiles.get(tile_id)
    if tile is None:
        tile = create_empty_tile(tile_id=tile_id, m_tile=int(atlas_map.m_tile))

    Lambdas_new = jnp.asarray(Lambdas_new, dtype=jnp.float64)
    thetas_new = jnp.asarray(thetas_new, dtype=jnp.float64)
    etas_new = jnp.asarray(etas_new, dtype=jnp.float64)
    weights_new = jnp.asarray(weights_new, dtype=jnp.float64).reshape(-1)
    valid_new_mask = jnp.asarray(valid_new_mask).reshape(-1).astype(bool)
    K = int(Lambdas_new.shape[0])

    # Select K eviction targets (empty slots first, then lowest-mass with primitive_id tie-break).
    target_slots = _select_lowest_mass_slots_fixed(
        weights=tile.weights,
        primitive_ids=tile.primitive_ids,
        valid_mask=tile.valid_mask,
        last_supported_scan_seq=tile.last_supported_scan_seq,
        scan_seq=scan_seq,
        recency_decay_lambda=recency_decay_lambda,
        k=K,
    )
    do_insert = valid_new_mask

    n_inserted = int(jnp.sum(do_insert.astype(jnp.int32)))
    # Assign contiguous IDs for inserted proposals only (fixed-cost prefix-sum).
    prefix = jnp.cumsum(do_insert.astype(jnp.int64)) - 1
    new_ids_full = jnp.where(
        do_insert,
        jnp.asarray(atlas_map.next_global_id, dtype=jnp.int64) + prefix,
        jnp.asarray(-1, dtype=jnp.int64),
    )

    # Masked updates into the selected empty slots.
    Lambdas_prev = tile.Lambdas[target_slots]
    thetas_prev = tile.thetas[target_slots]
    etas_prev = tile.etas[target_slots]
    weights_prev = tile.weights[target_slots]
    prim_ids_prev = tile.primitive_ids[target_slots]
    colors_prev = tile.colors[target_slots]
    valid_prev = tile.valid_mask[target_slots]

    Lambdas_set = jnp.where(do_insert[:, None, None], Lambdas_new, Lambdas_prev)
    thetas_set = jnp.where(do_insert[:, None], thetas_new, thetas_prev)
    etas_set = jnp.where(do_insert[:, None, None], etas_new, etas_prev)
    weights_set = jnp.where(do_insert, weights_new, weights_prev)
    prim_ids_set = jnp.where(do_insert, new_ids_full, prim_ids_prev)
    valid_set = valid_prev | do_insert

    if colors_new is not None:
        colors_new = jnp.asarray(colors_new, dtype=jnp.float64)
        colors_set = jnp.where(do_insert[:, None], colors_new, colors_prev)
    else:
        colors_set = colors_prev

    Lambdas = tile.Lambdas.at[target_slots].set(Lambdas_set)
    thetas = tile.thetas.at[target_slots].set(thetas_set)
    etas = tile.etas.at[target_slots].set(etas_set)
    weights = tile.weights.at[target_slots].set(weights_set)
    timestamps = tile.timestamps.at[target_slots].set(
        jnp.where(do_insert, float(timestamp), tile.timestamps[target_slots])
    )
    created_timestamps = tile.created_timestamps.at[target_slots].set(
        jnp.where(do_insert, float(timestamp), tile.created_timestamps[target_slots])
    )
    last_supported_scan_seq = tile.last_supported_scan_seq.at[target_slots].set(
        jnp.where(do_insert, jnp.asarray(scan_seq, dtype=jnp.int64), tile.last_supported_scan_seq[target_slots])
    )
    last_update_scan_seq = tile.last_update_scan_seq.at[target_slots].set(
        jnp.where(do_insert, jnp.asarray(scan_seq, dtype=jnp.int64), tile.last_update_scan_seq[target_slots])
    )
    primitive_ids = tile.primitive_ids.at[target_slots].set(prim_ids_set)
    valid_mask = tile.valid_mask.at[target_slots].set(valid_set)
    colors = tile.colors.at[target_slots].set(colors_set)

    new_tile = PrimitiveMapTile(
        tile_id=tile.tile_id,
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
        next_local_id=int(tile.next_local_id),
        count=int(jnp.sum(valid_mask.astype(jnp.int32))),
    )
    tiles = dict(atlas_map.tiles)
    tiles[tile_id] = new_tile
    new_atlas = AtlasMap(
        tiles=tiles,
        next_global_id=int(atlas_map.next_global_id + n_inserted),
        total_count=int(atlas_map.total_count + n_inserted),
        m_tile=atlas_map.m_tile,
    )

    triggers: List[str] = []
    dropped = int(jnp.sum((~valid_new_mask).astype(jnp.int32)))
    if dropped > 0:
        triggers.append("insert_unfilled_budget")
    cert = (
        CertBundle.create_approx(chart_id=chart_id, anchor_id=anchor_id, triggers=triggers, frobenius_applied=False)
        if triggers
        else CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
    )
    effect = ExpectedEffect(
        objective_name="primitive_map_insert_masked",
        predicted=float(int(jnp.sum(valid_new_mask.astype(jnp.int32)))),
        realized=float(n_inserted),
    )
    result = PrimitiveMapInsertResult(atlas_map=new_atlas, tile_id=tile_id, n_inserted=n_inserted, new_ids=new_ids_full)
    return result, cert, effect


@dataclass
class PrimitiveMapFuseResult:
    """Result of PrimitiveMapFuse operator."""
    atlas_map: AtlasMap
    tile_id: int
    n_fused: int


def primitive_map_fuse(
    atlas_map: AtlasMap,
    tile_id: int,
    target_slots: jnp.ndarray,       # (K,) slot indices into tile
    Lambdas_meas: jnp.ndarray,         # (K, 3, 3) measurement precisions
    thetas_meas: jnp.ndarray,          # (K, 3) measurement info vectors
    etas_meas: jnp.ndarray,            # (K, B, 3) measurement vMF params
    weights_meas: jnp.ndarray,         # (K,) measurement weights
    responsibilities: jnp.ndarray,     # (K,) soft association weights
    timestamp: float,
    scan_seq: int = 0,
    valid_mask: Optional[jnp.ndarray] = None,  # (K,) bool; when provided, zero out invalid before segment_sum
    colors_meas: Optional[jnp.ndarray] = None,  # (K, 3) measurement RGB; when provided, weighted blend per target
    eps_psd: float = constants.GC_EPS_PSD,
    eps_mass: float = constants.GC_EPS_MASS,
    fuse_chunk_size: int = constants.GC_FUSE_CHUNK_SIZE,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map",
) -> Tuple[PrimitiveMapFuseResult, CertBundle, ExpectedEffect]:
    """
    Fuse measurement primitives into a map tile via Product-of-Experts.

    Gaussian info fusion: Lambda_post = Lambda_prior + sum_k r_k * Lambda_meas_k
    vMF natural param addition: eta_post = eta_prior + sum_k r_k * eta_meas_k
    Colors: when colors_meas provided, map color at each target = responsibility-weighted mean of meas colors.

    Fixed-cost operator. Always applies (no gates).
    When valid_mask is provided (e.g. from JAX-only flatten), invalid entries contribute zero.
    """
    tile_id = int(tile_id)
    tile = atlas_map.tiles.get(tile_id)
    if tile is None:
        tile = create_empty_tile(tile_id=tile_id, m_tile=int(atlas_map.m_tile))
    K = target_slots.shape[0]
    if K == 0:
        result = PrimitiveMapFuseResult(atlas_map=atlas_map, tile_id=tile_id, n_fused=0)
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(objective_name="primitive_map_fuse", predicted=0.0, realized=0.0)
        return result, cert, effect

    # Streaming reduce-by-key: scatter-add into map-sized accumulators (fixed-cost chunks).
    map_size = tile.Lambdas.shape[0]
    d_Lambdas = jnp.zeros((map_size, 3, 3), dtype=jnp.float64)
    d_thetas = jnp.zeros((map_size, 3), dtype=jnp.float64)
    d_etas = jnp.zeros((map_size, constants.GC_VMF_N_LOBES, 3), dtype=jnp.float64)
    d_weights = jnp.zeros((map_size,), dtype=jnp.float64)
    d_color_sum = jnp.zeros((map_size, 3), dtype=jnp.float64)
    d_resp_sum = jnp.zeros((map_size,), dtype=jnp.float64)

    chunk = int(max(1, fuse_chunk_size))
    num_chunks = (K + chunk - 1) // chunk
    for i in range(num_chunks):
        start = i * chunk
        end = min(start + chunk, K)
        if start >= end:
            break

        idx = target_slots[start:end]
        resp = responsibilities[start:end]
        w_meas = weights_meas[start:end]
        Lambdas_c = Lambdas_meas[start:end]
        thetas_c = thetas_meas[start:end]
        etas_c = etas_meas[start:end]

        if valid_mask is not None:
            m_vec = valid_mask[start:end].astype(jnp.float64)
            resp_vec = (resp * m_vec).astype(jnp.float64)
        else:
            resp_vec = resp.astype(jnp.float64)

        r = resp_vec[:, None, None]
        r_vec = resp_vec[:, None]

        d_Lambdas = d_Lambdas.at[idx].add(r * Lambdas_c)
        d_thetas = d_thetas.at[idx].add(r_vec * thetas_c)
        # r is already (chunk, 1, 1); multiplying again would create extra dims and break scatter_add.
        d_etas = d_etas.at[idx].add(r * etas_c)
        d_weights = d_weights.at[idx].add(resp_vec * w_meas)
        d_resp_sum = d_resp_sum.at[idx].add(resp_vec)

        if colors_meas is not None and colors_meas.shape[0] >= end:
            colors_c = jnp.clip(colors_meas[start:end], 0.0, 1.0)
            d_color_sum = d_color_sum.at[idx].add(colors_c * r_vec)

    # Color: responsibility-weighted mean per target (only when colors_meas provided)
    if colors_meas is not None and colors_meas.shape[0] >= K:
        sum_r_safe = jnp.maximum(d_resp_sum[:, None], eps_mass)
        color_agg = jnp.clip(d_color_sum / sum_r_safe, 0.0, 1.0)
        updated = d_resp_sum > 0.0
        colors = jnp.where(updated[:, None], color_agg, tile.colors)
    else:
        colors = tile.colors

    # Update map (single add per field)
    Lambdas = tile.Lambdas + d_Lambdas
    thetas = tile.thetas + d_thetas
    etas = tile.etas + d_etas
    weights = tile.weights + d_weights
    timestamps = tile.timestamps.at[jnp.unique(target_slots)].set(timestamp)
    updated = d_resp_sum > 0.0
    last_supported_scan_seq = jnp.where(
        updated,
        jnp.asarray(scan_seq, dtype=jnp.int64),
        tile.last_supported_scan_seq,
    )
    last_update_scan_seq = jnp.where(
        updated,
        jnp.asarray(scan_seq, dtype=jnp.int64),
        tile.last_update_scan_seq,
    )

    new_tile = PrimitiveMapTile(
        tile_id=tile.tile_id,
        Lambdas=Lambdas,
        thetas=thetas,
        etas=etas,
        weights=weights,
        timestamps=timestamps,
        created_timestamps=tile.created_timestamps,
        last_supported_scan_seq=last_supported_scan_seq,
        last_update_scan_seq=last_update_scan_seq,
        primitive_ids=tile.primitive_ids,
        valid_mask=tile.valid_mask,
        colors=colors,
        next_local_id=tile.next_local_id,
        count=tile.count,
    )
    tiles = dict(atlas_map.tiles)
    tiles[tile_id] = new_tile
    new_atlas = AtlasMap(
        tiles=tiles,
        next_global_id=atlas_map.next_global_id,
        total_count=atlas_map.total_count,
        m_tile=atlas_map.m_tile,
    )

    n_unique = int(jnp.unique(target_slots).shape[0])
    result = PrimitiveMapFuseResult(atlas_map=new_atlas, tile_id=tile_id, n_fused=n_unique)
    cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
    effect = ExpectedEffect(
        objective_name="primitive_map_fuse",
        predicted=float(K),
        realized=float(n_unique),
    )
    return result, cert, effect


@dataclass
class PrimitiveMapCullResult:
    """Result of PrimitiveMapCull operator."""
    atlas_map: AtlasMap
    tile_id: int
    n_culled: int
    mass_dropped: float  # Total weight removed (logged as approximation)


def primitive_map_cull(
    atlas_map: AtlasMap,
    tile_id: int,
    weight_threshold: float = constants.GC_PRIMITIVE_CULL_WEIGHT_THRESHOLD,
    max_primitives: Optional[int] = None,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map",
) -> Tuple[PrimitiveMapCullResult, CertBundle, ExpectedEffect]:
    """
    Cull low-weight primitives from the map (explicit budgeting operator).

    Objective: resource constraint (retain only primitives with weight >= threshold
    and/or up to max_primitives); mass_dropped is logged. No hidden gate; declared
    budgeting operator. If mixture/family changes, Frobenius applied and logged.

    Args:
        prim_map: Current primitive map
        weight_threshold: Minimum weight to retain (tau)
        max_primitives: Optional hard limit (keeps highest weight)

    Returns:
        (result, CertBundle, ExpectedEffect) with mass_dropped logged
    """
    tile_id = int(tile_id)
    tile = atlas_map.tiles.get(tile_id)
    if tile is None:
        tile = create_empty_tile(tile_id=tile_id, m_tile=int(atlas_map.m_tile))
    if tile.count == 0:
        result = PrimitiveMapCullResult(atlas_map=atlas_map, tile_id=tile_id, n_culled=0, mass_dropped=0.0)
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(objective_name="primitive_map_cull", predicted=0.0, realized=0.0)
        return result, cert, effect

    # Find primitives below threshold
    below_threshold = tile.valid_mask & (tile.weights < weight_threshold)

    # If max_primitives specified, also cull excess
    n_to_keep = tile.count - int(jnp.sum(below_threshold))
    if max_primitives is not None and n_to_keep > max_primitives:
        # Need to cull more - keep top max_primitives by weight
        # Set threshold to weight of (max_primitives+1)th highest
        sorted_weights = jnp.sort(tile.weights * tile.valid_mask.astype(jnp.float64))[::-1]
        if max_primitives < len(sorted_weights):
            effective_threshold = float(sorted_weights[max_primitives])
            below_threshold = tile.valid_mask & (tile.weights < effective_threshold)

    # Compute mass dropped
    mass_dropped = float(jnp.sum(tile.weights * below_threshold.astype(jnp.float64)))
    n_culled = int(jnp.sum(below_threshold))

    if n_culled == 0:
        result = PrimitiveMapCullResult(atlas_map=atlas_map, tile_id=tile_id, n_culled=0, mass_dropped=0.0)
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(objective_name="primitive_map_cull", predicted=0.0, realized=0.0)
        return result, cert, effect

    # Clear culled entries
    valid_mask = tile.valid_mask & ~below_threshold
    new_count = tile.count - n_culled

    new_tile = PrimitiveMapTile(
        tile_id=tile.tile_id,
        Lambdas=tile.Lambdas,
        thetas=tile.thetas,
        etas=tile.etas,
        weights=tile.weights,
        timestamps=tile.timestamps,
        created_timestamps=tile.created_timestamps,
        last_supported_scan_seq=tile.last_supported_scan_seq,
        last_update_scan_seq=tile.last_update_scan_seq,
        primitive_ids=tile.primitive_ids,
        valid_mask=valid_mask,
        colors=tile.colors,
        next_local_id=tile.next_local_id,
        count=new_count,
    )
    tiles = dict(atlas_map.tiles)
    tiles[tile_id] = new_tile
    new_atlas = AtlasMap(
        tiles=tiles,
        next_global_id=atlas_map.next_global_id,
        total_count=atlas_map.total_count - n_culled,
        m_tile=atlas_map.m_tile,
    )

    result = PrimitiveMapCullResult(
        atlas_map=new_atlas,
        tile_id=tile_id,
        n_culled=n_culled,
        mass_dropped=mass_dropped,
    )

    # Log as approximation: budgeting operator; mass dropped is logged
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["budgeting", "mass_drop"],
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=mass_dropped / (jnp.sum(tile.weights) + constants.GC_EPS_MASS),
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    effect = ExpectedEffect(
        objective_name="primitive_map_cull",
        predicted=float(n_culled),
        realized=float(n_culled),
    )
    return result, cert, effect


@dataclass
class PrimitiveMapForgetResult:
    """Result of PrimitiveMapForget operator."""
    atlas_map: AtlasMap
    tile_id: int


def primitive_map_forget(
    atlas_map: AtlasMap,
    tile_id: int,
    forgetting_factor: float = constants.GC_PRIMITIVE_FORGETTING_FACTOR,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map",
) -> Tuple[PrimitiveMapForgetResult, CertBundle, ExpectedEffect]:
    """
    Apply continuous forgetting to primitive weights.

    Fixed-cost operator applied every scan. No if/else.

    Args:
        atlas_map: Current atlas map
        tile_id: Target tile ID
        forgetting_factor: Decay factor in (0, 1), closer to 1 = slower

    Returns:
        (result, CertBundle, ExpectedEffect)
    """
    tile_id = int(tile_id)
    tile = atlas_map.tiles.get(tile_id)
    if tile is None:
        result = PrimitiveMapForgetResult(atlas_map=atlas_map, tile_id=tile_id)
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(objective_name="primitive_map_forget", predicted=0.0, realized=0.0)
        return result, cert, effect
    gamma = float(forgetting_factor)

    new_tile = PrimitiveMapTile(
        tile_id=tile.tile_id,
        Lambdas=tile.Lambdas,
        thetas=tile.thetas,
        etas=tile.etas,
        weights=gamma * tile.weights,
        timestamps=tile.timestamps,
        created_timestamps=tile.created_timestamps,
        last_supported_scan_seq=tile.last_supported_scan_seq,
        last_update_scan_seq=tile.last_update_scan_seq,
        primitive_ids=tile.primitive_ids,
        valid_mask=tile.valid_mask,
        colors=tile.colors,
        next_local_id=tile.next_local_id,
        count=tile.count,
    )
    tiles = dict(atlas_map.tiles)
    tiles[tile_id] = new_tile
    new_atlas = AtlasMap(
        tiles=tiles,
        next_global_id=atlas_map.next_global_id,
        total_count=atlas_map.total_count,
        m_tile=atlas_map.m_tile,
    )

    result = PrimitiveMapForgetResult(atlas_map=new_atlas, tile_id=tile_id)
    cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
    effect = ExpectedEffect(
        objective_name="primitive_map_forget",
        predicted=1.0 - gamma,
        realized=1.0 - gamma,
    )
    return result, cert, effect


# =============================================================================
# Recency-based uncertainty inflation (continuous, fixed-cost)
# =============================================================================


@dataclass
class PrimitiveMapRecencyInflateStats:
    """Stats for recency-based inflation (for cert logging)."""
    staleness_inflation_strength: float
    staleness_cov_inflation_trace: float
    stale_precision_downscale_total: float


def primitive_map_recency_inflate(
    atlas_map: AtlasMap,
    tile_ids: List[int],
    scan_seq: int,
    recency_decay_lambda: float = constants.GC_RECENCY_DECAY_LAMBDA,
    min_scale: float = constants.GC_RECENCY_MIN_SCALE,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map_recency_inflate",
) -> Tuple[AtlasMap, CertBundle, ExpectedEffect, PrimitiveMapRecencyInflateStats]:
    """
    Inflate uncertainty for stale primitives by downscaling precision.

    Continuous, fixed-cost: apply to all primitives in the specified tiles.
    Scaling preserves mean (Lambda, theta scaled together).
    """
    tiles = dict(atlas_map.tiles)
    total_downscale = 0.0
    total_inflation_trace = 0.0
    n_valid_total = 0.0

    for tid in tile_ids:
        tid_i = int(tid)
        tile = tiles.get(tid_i)
        if tile is None:
            continue
        valid = tile.valid_mask.astype(jnp.float64)
        dt = jnp.maximum(
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(scan_seq, dtype=jnp.int64) - tile.last_supported_scan_seq,
        )
        decay = jnp.exp(-jnp.asarray(recency_decay_lambda, dtype=jnp.float64) * dt.astype(jnp.float64))
        decay = jnp.clip(decay, float(min_scale), 1.0)
        decay = jnp.where(tile.valid_mask, decay, 1.0)

        Lambdas = tile.Lambdas * decay[:, None, None]
        thetas = tile.thetas * decay[:, None]
        # etas are vMF parameters; do not scale.
        new_tile = PrimitiveMapTile(
            tile_id=tile.tile_id,
            Lambdas=Lambdas,
            thetas=thetas,
            etas=tile.etas,
            weights=tile.weights,
            timestamps=tile.timestamps,
            created_timestamps=tile.created_timestamps,
            last_supported_scan_seq=tile.last_supported_scan_seq,
            last_update_scan_seq=tile.last_update_scan_seq,
            primitive_ids=tile.primitive_ids,
            valid_mask=tile.valid_mask,
            colors=tile.colors,
            next_local_id=tile.next_local_id,
            count=tile.count,
        )
        tiles[tid_i] = new_tile

        n_valid = float(jnp.sum(valid))
        n_valid_total += n_valid
        total_downscale += float(jnp.sum((1.0 - decay) * valid))
        total_inflation_trace += float(jnp.sum(((1.0 / decay) - 1.0) * valid))

    new_atlas = AtlasMap(
        tiles=tiles,
        next_global_id=atlas_map.next_global_id,
        total_count=atlas_map.total_count,
        m_tile=atlas_map.m_tile,
    )

    strength = total_downscale / max(n_valid_total, 1.0)
    stats = PrimitiveMapRecencyInflateStats(
        staleness_inflation_strength=float(strength),
        staleness_cov_inflation_trace=float(total_inflation_trace),
        stale_precision_downscale_total=float(total_downscale),
    )
    cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
    effect = ExpectedEffect(
        objective_name="primitive_map_recency_inflate",
        predicted=float(n_valid_total),
        realized=float(n_valid_total),
    )
    return new_atlas, cert, effect, stats


# =============================================================================
# Merge-Reduce (Mixture Reduction)
# =============================================================================


@dataclass
class PrimitiveMapMergeReduceResult:
    """Result of PrimitiveMapMergeReduce operator."""
    atlas_map: AtlasMap
    tile_id: int
    n_merged: int
    frobenius_correction: float  # Applied when out-of-family


def primitive_map_merge_reduce(
    atlas_map: AtlasMap,
    tile_id: int,
    merge_threshold: float = constants.GC_PRIMITIVE_MERGE_THRESHOLD,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map",
) -> Tuple[PrimitiveMapMergeReduceResult, CertBundle, ExpectedEffect]:
    """
    Merge nearby primitives via mixture reduction.

    Uses Bhattacharyya distance to identify merge candidates.
    Merged primitive = weighted combination of Gaussians (moment matching).

    Fixed-cost operator with CertBundle + Frobenius correction.

    Args:
        atlas_map: Current atlas map
        tile_id: Target tile ID
        merge_threshold: Distance below which to merge
        eps_psd: PSD regularization
        eps_lift: Matrix inversion regularization

    Returns:
        (result, CertBundle, ExpectedEffect)
    """
    # For now, implement a simple version that doesn't merge
    # Full implementation would require nearest-neighbor search
    # and careful handling of the Gaussian/vMF mixture reduction

    # TODO: Implement actual merge logic with:
    # 1. Build k-d tree or spatial hash of primitive positions
    # 2. Find pairs with Bhattacharyya distance < threshold
    # 3. Merge via moment matching (weight-preserving)
    # 4. Log Frobenius correction for out-of-family approximation

    tile_id = int(tile_id)
    if tile_id not in atlas_map.tiles:
        result = PrimitiveMapMergeReduceResult(
            atlas_map=atlas_map,
            tile_id=tile_id,
            n_merged=0,
            frobenius_correction=0.0,
        )
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(
            objective_name="primitive_map_merge_reduce",
            predicted=0.0,
            realized=0.0,
        )
        return result, cert, effect

    result = PrimitiveMapMergeReduceResult(
        atlas_map=atlas_map,
        tile_id=tile_id,
        n_merged=0,
        frobenius_correction=0.0,
    )
    cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
    effect = ExpectedEffect(
        objective_name="primitive_map_merge_reduce",
        predicted=0.0,
        realized=0.0,
    )
    return result, cert, effect
