# Map Structure, Build, Visualization, and Known Issues

This document explains the **current map architecture** (tiles, atlas, MA-Hex), **how the map is built** per scan (measurements → association → fuse/insert → cull/forget/merge), **how it is visualized** (live and post-run), and **known issues and design tradeoffs**. It complements `docs/MAP_AND_VISUALIZATION.md` and `docs/GC_SLAM.md` (§5.7).

---

## 1. Current Structure

### 1.1 Primitive (One Splat)

Each map primitive is a 3D Gaussian in **information form** plus vMF orientation and optional color:

| Field | Meaning |
|-------|--------|
| **Lambda, theta** | Precision (3×3) and information vector (3×1); μ = Λ⁻¹θ, Σ = Λ⁻¹ |
| **etas** | vMF natural params (B lobes, B=3); resultant κ = ‖Σ η_b‖, mean direction = η/κ |
| **weight, timestamp, primitive_id** | Mass/ESS, last update time, stable ID |
| **color** | (3,) RGB in [0,1]; comes from measurement fusion/insert |

**Code:** `fl_slam_poc/backend/structures/primitive_map.py` — `Primitive`, `PrimitiveMapTile` (per-slot arrays).

### 1.2 Tiles and Atlas

- **PrimitiveMapTile:** One tile = a **fixed-size array** of slots. Each slot holds one primitive (Lambda, theta, etas, weight, timestamps, primitive_id, valid_mask, color). Capacity per tile is **M_TILE** (e.g. 50,000 in config `primitive_map_max_size`). Only slots with `valid_mask[i] == True` count toward `tile.count`.
- **AtlasMap:** Global map = `Dict[tile_id → PrimitiveMapTile]` plus `next_global_id` and `total_count`. Tiles are created on demand when a primitive is first inserted into that tile.
- **Frame:** All primitives are stored in **world frame**. The tile is an indexing and budgeting structure only.

**Code:** `primitive_map.py` — `PrimitiveMapTile`, `AtlasMap`, `create_empty_tile`, `create_empty_atlas_map`.

### 1.3 MA-Hex 3D: Tile ID and “Nearby”

- **Tile ID** is determined **only by 3D position** via the **MA-Hex 3D** grid:
  - XY: hex grid with basis A1=[1,0], A2=[0.5, √3/2]. Cell coords: `c1 = floor((A1·[x,y])/H_TILE)`, `c2 = floor((A2·[x,y])/H_TILE)`.
  - Z: `cz = floor(z / H_TILE)`.
  - `(c1, c2, cz)` is packed into a single integer `tile_id` (deterministic, no wrap).
- **No overlap:** Each point in space belongs to **exactly one** tile (one cell). Tiles partition space; they do not overlap.
- **Stencil:** Given a center (e.g. robot position), `ma_hex_stencil_tile_ids(center_xyz, H_TILE, radius_xy, radius_z)` returns the tile under the center plus a **fixed-size neighborhood** (hex disk in XY, slab in Z). That list is used as “nearby” for association and updates.
- **Active set:** The tiles that are **updated** this scan = stencil around current robot pose with radii `R_ACTIVE_TILES_XY`, `R_ACTIVE_TILES_Z`. Size = **N_ACTIVE_TILES** (constant, e.g. 7).
- **Stencil (candidates):** Slightly larger neighborhood `R_STENCIL_TILES_XY/Z` used to build the **candidate pool** for OT association. Size = **N_STENCIL_TILES** (e.g. 7).

So the **MA-Hex structure is already the spatial index**: “nearby” = list of tile IDs in the stencil; “which tile does this point belong to?” = `tile_id_from_xyz(xyz, H_TILE)`.

**Code:** `fl_slam_poc/common/tiling.py` — `ma_hex_cell_3d_from_xyz`, `tile_id_from_xyz`, `tile_ids_from_xyz_batch_jax`, `ma_hex_stencil_tile_ids`; `constants.py` — `GC_H_TILE`, `GC_R_ACTIVE_TILES_*`, `GC_N_ACTIVE_TILES`, etc.

### 1.4 Key Constants (Config vs Code Defaults)

| Constant | Meaning | Example (gc_unified.yaml / code default) |
|----------|---------|------------------------------------------|
| **H_TILE** | Tile scale (m); hex cell size | 2.0 |
| **N_ACTIVE_TILES** | Number of tiles updated per scan (stencil around robot) | 7 |
| **N_STENCIL_TILES** | Number of tiles in candidate stencil | 7 |
| **primitive_map_max_size** / **M_TILE** | Max primitives **per tile** | 50,000 |
| **M_TILE_VIEW** | Max primitives per tile exposed to association (top by weight) | 1024 |
| **k_insert_tile** | Max new primitives **per active tile** per scan | 64 |
| **n_feat** | Camera feature budget (also sets C++ visual_feature_node max_features) | 256 |
| **n_surfel** | LiDAR surfel budget per scan | 512 |

So **max primitives added per scan** = `k_insert_tile × N_ACTIVE_TILES` (e.g. 64×7 = **448**). The map only receives new primitives in **active** tiles (robot neighborhood). Tiles the robot never approaches never get primitives.

---

## 2. How the Map Is Built (Per Scan)

### 2.1 Measurement Batch (Camera + LiDAR)

- **Camera slice:** Up to **n_feat** visual features (from C++ `visual_feature_node`; ORB keypoints, depth, covariance, **color** sampled at (u,v) from the image). Built by `feature_list_to_camera_batch` from a `Feature3D` list (after `splat_prep_fused` with LiDAR depth evidence). Indices `0 .. n_feat-1`.
- **LiDAR slice:** Up to **n_surfel** surfels (from deskewed points → MA-hex voxels, plane fit, precision, vMF normals). **No RGB:** LiDAR has no color; when `colors_lidar` is not provided, colors are set to **grayscale from normal.z** (0.25–0.75) via `_lidar_default_colors_from_normals`. Indices `n_feat .. n_feat+n_surfel-1`.
- **Merged batch:** One `MeasurementBatch` with `n_feat + n_surfel` rows (e.g. 256 + 512 = 768). Camera rows have real RGB when features carry color; LiDAR rows have grayscale.

**Code:** `backend/camera_batch_utils.py`, `backend/structures/measurement_batch.py`, `backend_node.py` (camera_batch + extract_lidar_surfels with base_batch=camera_batch), `pipeline.py` (measurement_batch_local).

### 2.2 Per-Scan Map Update Sequence

1. **Recency inflate (active tiles only)**  
   Primitives that haven’t been supported recently get precision downscaled (uncertainty inflated). No gates.

2. **Extract map view (candidate pool)**  
   From **stencil** tiles, take top **M_TILE_VIEW** primitives per tile by weight → single **AtlasMapView** (candidate pool for association). Fixed size.

3. **Association (OT)**  
   Optimal transport between **MeasurementBatch** (camera + LiDAR rows) and **AtlasMapView**. Output: transport plan (responsibilities) — which measurement rows assign to which map slots. **K_ASSOC** candidates per measurement; Sinkhorn fixed iterations.

4. **Fuse**  
   For each map primitive that received mass: Gaussian and vMF natural params are **added** (PoE); **color** = responsibility-weighted mean of the **measurement row colors** that were associated to that primitive. So each primitive’s color comes from the **single measurement row** (or blend of rows) it was associated with.

5. **Insert**  
   “Novel” measurements (high mass not explained by existing map) are proposed as new primitives. Per **active** tile: select top **k_insert_tile** by novelty score; call `primitive_map_insert_masked` to write into (possibly evicted) slots. **Color** = `measurement_batch.colors[ins_idx_j]` — again, the color of the measurement row that was chosen for insert.

6. **Cull**  
   Remove primitives with weight below threshold (per active tile). Mass dropped is logged.

7. **Forget**  
   Multiply all primitive weights in active tiles by forgetting factor (< 1). Every scan.

8. **Merge-reduce**  
   Merge nearby primitives (Bhattacharyya distance below threshold) into one per tile (fixed budget of pairs). Frobenius correction logged.

**Code:** `pipeline.py` — map branch: recency inflate, extract_atlas_map_view, associate_primitives_ot, then per-tile fuse, insert, cull, forget, merge_reduce over `active_tile_ids`.

### 2.3 Why the Map Does Not “Continuously Fill”

- **Only active tiles are updated:** Fuse/insert/cull/forget/merge run only for tiles in the **active stencil** (robot neighborhood). Tiles the robot never visits never get primitives.
- **Fixed insert budget per scan:** At most `k_insert_tile × N_ACTIVE_TILES` new primitives per scan (e.g. 448). The rest of the measurement batch does not create new map primitives this scan.
- **Per-tile capacity:** Each tile has at most **M_TILE** primitives (e.g. 50,000). When full, insert evicts low-retention slots; total per tile is capped.
- **No overlap:** Tiles partition space; there is no “overlap” of tiles to “fill in” the same area twice.

So the map is **local and budgeted**: it fills only where the robot has been (active set), with a fixed insert rate and a per-tile cap. It is **not** a single unbounded surface that fills everywhere. The design was chosen to enforce **fixed-cost** per scan (bounded candidate pool, bounded insert count) and **local modularity** (updates stay in a fixed set of tiles). See §4 for the alternative (same MA-Hex index, different cap policy).

---

## 3. How the Map Is Visualized

### 3.1 Live (During Run)

- **ROS:** `PrimitiveMapPublisher` publishes the map as **PointCloud2** on `/gc/map/points` (x, y, z, intensity = weight). Frame_id = odom (or config). Optionally ellipsoid markers on `/gc/map/ellipsoids` if enabled.
- **Rerun:** If `use_rerun` is true, `RerunVisualizer.log_map(positions, weights, colors, time_sec)` logs `gc/map/points` as **Points3D** (radii from weights; colors if present). Trajectory is logged as `gc/trajectory` (LineStrips3D). No blueprint tabs are sent by the live backend; the viewer shows a single 3D view.
- **Subset:** The published map can be limited to a set of tile IDs (e.g. active stencil) so the live view is local.

**Code:** `backend/map_publisher.py`, `backend/rerun_visualizer.py`, `backend_node.py` (_publish_state_from_pose).

### 3.2 Post-Run: Splat Export and Rerun Build

- **Splat export (on shutdown):** If `splat_export_path` is set and the primitive map has primitives, the backend writes **splat_export.npz** containing: positions, covariances, directions, kappas, weights, **colors**, primitive_ids, timestamps, created_timestamps (and `n`). One file per run (e.g. `results/gc_YYYYMMDD_HHMMSS/splat_export.npz`).
- **Build Rerun from splat:** `tools/build_rerun_from_splat.py` reads `splat_export.npz` (and optional trajectory TUM) and builds a `.rrd` file. It logs:
  - **gc/map/splats/colored** — Points3D with positions + colors + radii (from covariance scale). Blueprint tab: **“Dense Color”**.
  - **gc/map/splats/ellipsoids** — Ellipsoids3D from covariances. Tab: **“Uncertainty Ellipsoids”**.
  - **gc/map/splats/normals** — Arrows3D (vMF directions scaled by κ). Tab: **“Normals (vMF)”**.
  - **gc/map/splats/weights** — Points3D colored by weight (viridis LUT). Tab: **“Weights”**.
  - **gc/trajectory** — LineStrips3D from TUM. Tab: **“Trajectory”**.
- Colors in the NPZ are **whatever the map had at export time** (from fuse/insert). The script applies optional vMF+fBm shading for BEV15; the main “Dense Color” view uses the NPZ colors directly (or shaded once at t=0).

So the **“Dense Color”** view in the post-run Rerun recording is **not** live rendering; it is the **splat_export.npz** geometry and color, built once after the run. If the run had few scans or no camera color in the map, the view will look sparse and/or grayscale.

**Code:** `backend_node.py` (shutdown: splat_export_path), `tools/build_rerun_from_splat.py` (build_rrd, _log_splat_views_at_t0, blueprint).

### 3.3 Rendering Module (Backend)

`fl_slam_poc/backend/rendering.py` implements EWA splatting, multi-lobe vMF shading, and fBm for view-stable texture. It is used by `build_rerun_from_splat.py` for BEV15 shaded colors. Live backend does not currently produce pixel renderings from this module; live viz is point cloud / Points3D (and optionally ellipsoids).

---

## 4. Known Issues and Design Tradeoffs

### 4.1 Map Color: Camera vs LiDAR

- **Observation:** The map often looks **grayscale** in the “Dense Color” view even when the camera provides RGB.
- **Cause:** Color is taken from the **measurement row** that **won** the association (or was chosen for insert). The merged batch has **n_feat** camera rows (with RGB) and **n_surfel** LiDAR rows (grayscale from normals). There are fewer camera rows than LiDAR rows, and most associations are to LiDAR geometry. So **most** map primitives get their color from **LiDAR** (grayscale). Camera RGB is used only for the minority of primitives associated with camera feature rows. In addition, later scans can **overwrite** a primitive’s color when it is fused with a LiDAR row.
- **Summary:** The pipeline **does** use camera color when a primitive is associated with a camera row; the **imbalance** (more LiDAR rows, geometry driven by LiDAR) means most primitives end up with LiDAR grayscale. LiDAR does not provide RGB; the only way to get dense color in the map would be to prefer or propagate camera color when it exists for the same 3D region (e.g. paint LiDAR rows with camera color when co-located, or a separate color-from-camera pass).

### 4.2 Visual Feature Count

- **Observation:** “Why are we not extracting more visual features?”
- **Cause:** The C++ `visual_feature_node`’s **max_features** is set from the backend config key **n_feat** at launch (e.g. 256 in gc_unified.yaml). So extraction is **capped at n_feat** per frame. ORB may also return fewer keypoints (e.g. low texture); then the batch has fewer than n_feat valid features.
- **Summary:** To get more visual features per frame, increase **n_feat** in config (and ensure the C++ node is receiving it via launch). Whether extraction is “working” can be checked by inspecting `/gc/sensors/visual_features` (e.g. `count` field) and backend logs.

### 4.3 Sparse or Black Post-Run View

- **Observation:** Post-run Rerun “Dense Color” view sometimes shows **sparse black clusters**.
- **Causes:**  
  1. **Sparse:** The run wrote few primitives (e.g. very few scans processed, bag path wrong, or backend exited early). Splat export only writes what is in the map at shutdown.  
  2. **Black:** Primitives with **zero or near-zero color**. New tiles start with `colors = 0`; color is updated only when measurements with non-zero color are fused or inserted. If the measurement batch had no camera and LiDAR default was not applied (e.g. empty batch), or only a few scans ran, many primitives can remain black.
- **Check:** Inspect `splat_export.npz`: `n = data['positions'].shape[0]`, `data['colors'].min()/max()`. A healthy run has many primitives and non-zero colors (at least grayscale).

### 4.4 Map Not “Continuously Filling”

- **Observation:** The map does not behave like a single surface that fills all observed space; it fills only in **local patches** (active tiles) with **fixed insert budget** and **per-tile caps**.
- **Design reason:** The spec (§5.7, §1.3) requires **fixed-cost** per scan: bounded candidate pool (stencil × M_TILE_VIEW), bounded insert count (N_ACTIVE_TILES × K_INSERT_TILE), tile-local fuse/insert. Tiles + active set enforce that.
- **Alternative:** Fixed per-scan cost **does not strictly require** tiles. One could use the **same MA-Hex** as a spatial index (“nearby” = stencil of cells), maintain a **single global** (or per-cell) primitive store, and cap growth by **global prune** (e.g. by recency, distance from robot, or total count). Then “nearby” is still bounded (stencil size), and cost stays bounded. Tiles are one way to get locality and caps; **hex + prune** is another. The current implementation chose tiles for explicit per-tile budgets and certificates; a “continuously filling” feel could be achieved by relaxing per-tile caps and using prune, or by increasing insert budget and tile capacity (still only filling where the robot has been).

### 4.5 Tiles Do Not Overlap

- Tiles partition space (one tile per MA-Hex cell). They do **not** overlap to “fill in” the same area twice. To get more primitives in a region, you increase per-tile capacity or decrease H_TILE (more tiles in the same area), not overlap.

---

## 5. Code Anchors (Quick Reference)

| Topic | Location |
|-------|----------|
| Primitive, tile, atlas structures | `fl_slam_poc/backend/structures/primitive_map.py` |
| Map operators (fuse, insert, cull, forget, merge, recency inflate) | `primitive_map.py` |
| MA-Hex tiling, tile_id, stencil | `fl_slam_poc/common/tiling.py` |
| Constants (H_TILE, N_ACTIVE_TILES, k_insert_tile, n_feat, etc.) | `fl_slam_poc/common/constants.py`, `config/gc_unified.yaml` |
| Pipeline map branch (association, fuse, insert) | `fl_slam_poc/backend/pipeline.py` |
| Measurement batch (camera + LiDAR, colors) | `camera_batch_utils.py`, `structures/measurement_batch.py` |
| Map publisher, Rerun live | `backend/map_publisher.py`, `backend/rerun_visualizer.py` |
| Splat export at shutdown | `backend/backend_node.py` |
| Post-run Rerun from NPZ + trajectory | `tools/build_rerun_from_splat.py` |
| Spec (atlas, fixed-cost) | `docs/GC_SLAM.md` §5.7, §1.3 |
| Map components and viz (conceptual) | `docs/MAP_AND_VISUALIZATION.md` |

---

## 6. Summary

- **Structure:** Map = atlas of tiles; each tile = fixed-size array of primitives (Gaussian + vMF + color). Tile ID = MA-Hex 3D cell from position (no overlap). Active set = stencil around robot; only those tiles are updated.
- **Build:** Each scan: recency inflate → extract map view (stencil) → OT association (measurement batch vs map view) → fuse (PoE + color from associated measurement rows) → insert (novelty-driven, fixed budget per active tile) → cull, forget, merge-reduce.
- **Visualization:** Live = PointCloud2 + Rerun Points3D (and trajectory). Post-run = splat_export.npz + `build_rerun_from_splat.py` → .rrd with Dense Color, Ellipsoids, Normals, Weights, Trajectory tabs.
- **Issues:** Map color is dominated by LiDAR (grayscale) because most associations are to LiDAR rows; visual feature count is capped by n_feat; sparse/black view usually means few scans or no color in the batch; map is local/budgeted by design (fixed-cost), but the same MA-Hex index could support a different cap policy (e.g. global prune) for a more “continuously filling” behavior without scaling cost.
