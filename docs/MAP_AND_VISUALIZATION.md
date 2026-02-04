# Map Components and Map Visualization (In-Depth)

This document explains the **components of the GC SLAM map**, how the map is **built and maintained**, and how **map visualization** works (live and post-run). It aligns with the code and with `docs/GC_SLAM.md` (§5.7, §11).

---

## 1. What the Map Is

The map is a **probabilistic primitive atlas**: a collection of 3D Gaussian “splats” (primitives) with orientation/appearance (vMF) and optional color, organized into **tiles** for fixed-cost updates.

### 1.1 Single Primitive (One Splat)

Each primitive \(j\) in the map has:

| Field | Meaning | Storage |
|-------|--------|---------|
| **Geometry (3D Gaussian)** | Position uncertainty in **information form** | `Lambda_j` (3×3 precision), `theta_j` (3×1 info vector); \(\mu_j = \Lambda_j^{-1} \theta_j\), \(\Sigma_j = \Lambda_j^{-1}\) |
| **Orientation / appearance** | Directional concentration on the sphere (vMF) | `etas` (B×3) = \(\kappa_b \mu_b\) for B lobes (B=3); resultant \(\eta = \sum_b \eta_b\), \(\kappa = \|\eta\|\), \(\mu_{\mathrm{dir}} = \eta/\kappa\) |
| **Metadata** | Identity and mass | `primitive_id`, `weight` (accumulated mass/ESS), `timestamp`, `last_supported_scan_seq` |
| **Color** | Optional RGB for rendering | `color` (3,) in [0,1]; fused responsibility-weighted from measurements |

**Code:** `fl_slam_poc/backend/structures/primitive_map.py` — `Primitive` dataclass and helpers (`mean_position`, `covariance`, `kappa`, `mean_direction`).

### 1.2 Tile (Fixed-Size Container)

A **PrimitiveMapTile** is one tile in the atlas:

- **Fixed capacity:** `M_TILE` slots (e.g. 4096 or from config `primitive_map_max_size`).
- **Per-slot arrays:** `Lambdas` (M_TILE, 3, 3), `thetas` (M_TILE, 3), `etas` (M_TILE, B, 3), `weights`, `timestamps`, `created_timestamps`, `last_supported_scan_seq`, `last_update_scan_seq`, `primitive_ids`, `valid_mask`, `colors`.
- **Tile-local addressing:** Primitives are referred to by `(tile_id, slot)`; `slot` is an index into these arrays.
- **Valid subset:** Only entries with `valid_mask[i] == True` count toward `tile.count`.

Tiles are created on demand. Tile ID comes from **MA-Hex 3D**: a deterministic function of 3D position (e.g. robot center) and scale `H_TILE`, so the same world region always maps to the same tile ID.

**Code:** `primitive_map.py` — `PrimitiveMapTile`, `create_empty_tile`.

### 1.3 Atlas (Global Map Container)

The **AtlasMap** is the full map:

- **tiles:** `Dict[tile_id → PrimitiveMapTile]`.
- **next_global_id:** Monotonically increasing ID for new primitives.
- **total_count:** Sum of valid primitives over all tiles.
- **m_tile:** Capacity per tile (fixed at creation).

Primitives are stored in **world frame**. The tile is only an indexing and budgeting structure, not a coordinate frame.

**Code:** `primitive_map.py` — `AtlasMap`, `create_empty_atlas_map`.

---

## 2. How the Map Is Built and Updated (Pipeline)

Map updates happen **per scan** inside `process_scan_single_hypothesis` in `pipeline.py`. The pipeline receives the current `primitive_map` (AtlasMap) and returns an updated one in `ScanPipelineResult.primitive_map_updated`.

### 2.1 Inputs to the Map Branch

- **LiDAR:** Deskewed points → **LiDAR surfels** (MA-hex voxels, plane fit, Wishart-regularized precision, vMF normals) → `MeasurementBatch` (LiDAR slice).
- **Camera (if enabled):** Features + depth (camera + LiDAR depth fusion, PoE) → **camera splats** → `MeasurementBatch` (camera slice).
- **Pose:** Predicted pose from belief (for “where the robot is”) and for transforming measurements into world frame.
- **Active / stencil tiles:** From current pose via MA-Hex:
  - **Active tiles:** Fixed-size neighborhood (e.g. `N_ACTIVE_TILES`) — only these are **updated** (fuse/insert/cull/forget/merge).
  - **Stencil tiles:** Slightly larger neighborhood — used to build the **candidate pool** for association.

### 2.2 Per-Scan Map Update Sequence

1. **Recency inflate (active tiles only)**  
   For primitives that haven’t been supported recently, precision is downscaled (uncertainty inflated). Continuous, no gates.  
   **Operator:** `primitive_map_recency_inflate`.

2. **Candidate pool**  
   From the **stencil** tiles, take top-`M_TILE_VIEW` primitives per tile by weight and stitch into a fixed-size **AtlasMapView** (candidate pool for association).  
   **Code:** `extract_atlas_map_view`.

3. **Association (OT)**  
   Between current **MeasurementBatch** (camera + LiDAR primitives) and **AtlasMapView** (map candidates). Output is a **transport plan** \(\pi[i,k]\) (responsibilities): how much of measurement \(i\) goes to map candidate \(k\).  
   **Operator:** `associate_primitives_ot` (Sinkhorn, fixed iterations).  
   **Code:** `primitive_association.py`.

4. **Fuse (into existing map primitives)**  
   For each map primitive that received mass from the transport plan:
   - **Gaussian:** \(\Lambda_{\mathrm{post}} = \Lambda_{\mathrm{prior}} + \sum_k r_k \Lambda_{\mathrm{meas},k}\), \(\theta_{\mathrm{post}} = \theta_{\mathrm{prior}} + \sum_k r_k \theta_{\mathrm{meas},k}\) (PoE in natural parameters).
   - **vMF:** \(\eta_{\mathrm{post}} = \eta_{\mathrm{prior}} + \sum_k r_k \eta_{\mathrm{meas},k}\).
   - **Color:** Responsibility-weighted mean of measurement colors (when provided).
   - **Weights / scan seq:** Updated from responsibilities and scan index.  
   **Operator:** `primitive_map_fuse` (tile-local, scatter-add into tile arrays).  
   **Code:** `primitive_map.py` — `primitive_map_fuse`.

5. **Insert (new primitives)**  
   Measurements that did not associate strongly to existing map primitives are proposed as **new** primitives. A fixed number of **eviction slots** per active tile are chosen (e.g. lowest retention by weight×recency decay). New proposals (masked) are written into those slots; IDs assigned; valid_mask set.  
   **Operator:** `primitive_map_insert_masked`.  
   **Code:** `primitive_map.py` — `primitive_map_insert_masked`.

6. **Cull (per active tile)**  
   Remove primitives whose weight is below a threshold (and optionally enforce a max count per tile). **Mass dropped** is logged (explicit budgeting; CertBundle reports it).  
   **Operator:** `primitive_map_cull`.

7. **Forget (per active tile)**  
   Multiply all primitive weights by a forgetting factor \(< 1\). Applied every scan, no gates.  
   **Operator:** `primitive_map_forget`.

8. **Merge-reduce (per active tile)**  
   Merge **nearby** primitives (Bhattacharyya distance below threshold) into one via moment matching (Gaussian + vMF blend). Reduces count; **Frobenius correction** is applied and logged (mixture reduction is an approximation).  
   **Operator:** `primitive_map_merge_reduce`.

All of the above are **fixed-cost** (fixed tile set, fixed caps, fixed iteration counts). No “if first scan then X else Y”; insertion is additive every scan subject to the tile budget.

**Code:** `pipeline.py` — map branch in `_compute_map_branch`, then fuse/insert/cull/forget/merge_reduce loop over `active_tile_ids`; result in `primitive_map_updated`.

---

## 3. Map Visualization — Conceptual Split

Visualization is **output-only**: it consumes the map (and trajectory) and never feeds back into inference. Per GC_SLAM.md §11: topology and rendering are **derived**; they reflect belief quality but do not change it.

Two main paths:

1. **Live (during run):** Backend publishes map + trajectory; Rerun (and optionally ROS) show them.
2. **Post-run:** Splat export (NPZ) + trajectory (TUM) are written at shutdown; tools build Rerun recordings (and optional BEV15 view layer) from that.

---

## 4. Live Map Visualization

### 4.1 What Gets Published Each Cycle

Whenever the backend publishes state (after a scan that produced a new pose), it:

1. **Publishes trajectory**  
   Path as ROS `Path` and, if Rerun is enabled, `log_trajectory` (LineStrips3D).

2. **Publishes map**  
   The **PrimitiveMapPublisher** is called with the current `primitive_map` and (optionally) a list of tile IDs to include (e.g. active stencil around current pose). It:
   - For each tile (or all tiles if no list): **extract** a **PrimitiveMapView** (`extract_primitive_map_view`) — positions, covariances, directions, kappas, weights, colors from the tile’s info-form arrays.
   - Concatenate views, optionally sort by recency/primitive_id.
   - **PointCloud2:** Build a ROS message with fields `x, y, z, intensity` (intensity = weight), frame_id = odom (or config), and publish to `/gc/map/points`.
   - **Rerun:** If a `RerunVisualizer` is attached, call `log_map(positions, weights, colors, time_sec)` → logs `gc/map/points` as **Points3D** (radii from weights). Trajectory is logged as `gc/trajectory` (LineStrips3D).

So **live** map viz is **point cloud style**: positions + weight (and color if present). Covariances are not drawn in the default Rerun path; they are available in the publisher’s returned `RenderablePrimitiveBatch` but not yet sent as ellipsoids to Rerun in the current code.

**Code:**  
- `backend_node.py` — `_publish_state_from_pose`: calls `map_publisher.publish(primitive_map, stamp_sec, tile_ids=...)`; trajectory → `rerun_visualizer.log_trajectory`.  
- `map_publisher.py` — `PrimitiveMapPublisher.publish`: extracts views, builds PointCloud2, publishes; calls `rerun_visualizer.log_map`.  
- `rerun_visualizer.py` — `RerunVisualizer.log_map` / `log_trajectory`.

### 4.2 Rerun Initialization

If launch is started with `use_rerun:=true` and optionally a recording path, the backend creates a `RerunVisualizer`, calls `init()` (and optionally `rr.save(path)` for saving to `.rrd`). All subsequent `log_map` and `log_trajectory` go to that recording and/or live viewer.

---

## 5. Post-Run Map Visualization (Splat Export + Rerun Build)

### 5.1 Splat Export (Shutdown)

When the backend is shutting down (or when explicitly requested), if `splat_export_path` is set and the primitive map has primitives:

- For **each tile** in the atlas, it calls `extract_primitive_map_view` (no max_primitives limit for export).
- It concatenates **positions**, **covariances**, **directions**, **kappas**, **weights**, **colors**, **primitive_ids**, **timestamps** (and optionally **created_timestamps**) into NumPy arrays and saves them to a single **NPZ** file (e.g. `results/gc_YYYYMMDD_HHMMSS/splat_export.npz`).

That file is the **full geometric + appearance snapshot** of the map (world frame), suitable for offline rendering or analysis. No Rerun or GPU renderer is required to produce it.

**Code:** `backend_node.py` — shutdown handler: `splat_export_path`, loop over `primitive_map.tiles`, `extract_primitive_map_view`, then `np.savez_compressed(...)`.

### 5.2 Building a Rerun Recording From Splat Export

The script **`tools/build_rerun_from_splat.py`** builds one or two `.rrd` files **after** a run:

- **Inputs:**  
  - `splat_export.npz` (positions, covariances, colors, weights, directions, kappas, …).  
  - Optional trajectory TUM file (stamps, xyz, quat).

- **Main Rerun recording (e.g. `gc_slam.rrd`):**  
  - **Map as Ellipsoids3D:** For each primitive, covariance \(\Sigma\) is turned into an ellipsoid: eigendecomposition \(\Sigma = R D R^T\) → half-axes = \(\sqrt{\text{eigenvalues}}\), rotation from \(R\) (quaternion xyzw). Same convention as 3D Gaussian principal axes (EWA). Logged under a single entity (e.g. `gc/map/ellipsoids` or similar).  
  - **Trajectory:** LineStrips3D from TUM xyz; optional Transform3D per pose (from TUM quat).  
  - **Time:** Uses trajectory timestamps when available.

- **Optional BEV15 recording (e.g. `gc_bev15.rrd`):**  
  - 15 oblique view directions (BEV15); for each view, 2D projected positions (and shaded colors) are logged as Points2D per view.  
  - **Shaded colors:** `_shaded_colors` uses **vMF shading** (view direction vs. primitive direction, multi-lobe) and **fBm** at splat (x,y) for view-stable texture — same math as in `fl_slam_poc.backend.rendering` (vmf_shading_multi_lobe, fbm_at_splat_positions).  
  - This is view-only, not used in the runtime pipeline.

So **post-run** visualization:
- Uses the **same** map representation (splat export) and **same** rendering ideas (vMF + fBm) as the backend.
- Ellipsoids give proper 3D uncertainty; BEV15 gives a fixed set of 2D oblique views with shading.

**Code:** `tools/build_rerun_from_splat.py` — `_covariance_to_ellipsoid`, `_shaded_colors` (imports from `fl_slam_poc.backend.rendering`), `build_rrd`, `_build_bev15_rrd`.

---

## 6. Rendering Module (Backend Side — Not Live Pixel Output Today)

The file **`fl_slam_poc/backend/rendering.py`** implements the **canonical** rendering model for splats (EWA + vMF + fBm). It is used:

- By **build_rerun_from_splat.py** for BEV15 shaded colors (vMF + fBm).
- Not yet by the **live** backend to produce an image (live viz is point cloud / ellipsoids via Rerun).

Rendering is defined as:

- **EWA splatting:** Weight at pixel \(p\) from a primitive: \(\alpha \exp(-\frac{1}{2}(p-\mu)^T \Sigma^{-1} (p-\mu))\), with log-domain clipping for stability.  
  - **Tile binning:** For each image tile, a fixed cap of splats overlapping that tile (by 2\(\sigma\) bbox) are selected; cost is O(pixels × cap).  
- **Multi-lobe vMF shading:** Shading = \(\sum_b \pi_b \exp(\kappa_b (\mu_{n,b}^T v - 1))\), energy-normalized. Optionally κ is modulated by intensity (e.g. LiDAR reflectivity).  
- **fBm:** Value noise at each splat’s world (x,y) to modulate color — view-stable (no shimmer when camera moves).  
- **Opacity:** Can be derived from precision (e.g. logdet) with a soft floor.

So the **map** carries full geometry (Λ, θ) and appearance (η, color); **visualization** turns that into:
- **Points** (live: positions + weight/color),
- **Ellipsoids** (post-run: Σ → half-axes + quat),
- **Shaded 2D/3D views** (post-run BEV15 and, in principle, any EWA render from `rendering.py`).

---

## 7. How It Should Behave End-to-End

1. **Map growth:** Each scan, active tiles get **fuse** (measurements → existing primitives via OT responsibilities) and **insert** (new primitives into eviction slots). Cull/forget/merge-reduce keep the map within budget and merge duplicates. So the map **grows** where there is new evidence and **shrinks** where mass is low or merged.

2. **Consistency:** All map state is in **world frame**. Tile IDs are deterministic from position (MA-Hex). Fusion is **additive in natural parameters** (Gaussian + vMF); no hidden gates.

3. **Live viz:** Subset of map (e.g. active stencil) is published as PointCloud2 and Rerun Points3D (and trajectory as LineStrips3D). What you see is the current belief over the environment and path.

4. **Post-run viz:** Splat export is the single source of truth for “final map.” Rerun recordings (main + optional BEV15) are **derived** from that NPZ + trajectory so you can inspect geometry (ellipsoids), trajectory, and shaded views without re-running SLAM.

5. **No feedback:** Visualization and topology (e.g. nerve) are **derived** outputs. They do not affect which primitives exist, which tiles are active, or any inference step.

---

## 8. Code Anchors (Quick Reference)

| Concept | Primary location |
|--------|-------------------|
| Primitive / tile / atlas structures | `fl_slam_poc/backend/structures/primitive_map.py` |
| Map maintenance operators (fuse, insert, cull, forget, merge-reduce, recency inflate) | `primitive_map.py` |
| Pipeline map branch (surfel extraction, association, fuse, insert, cull, forget, merge) | `fl_slam_poc/backend/pipeline.py` |
| Map publisher (PointCloud2 + Rerun log_map) | `fl_slam_poc/backend/map_publisher.py` |
| Rerun live logging (map points, trajectory) | `fl_slam_poc/backend/rerun_visualizer.py` |
| Backend: when map is published / splat export at shutdown | `fl_slam_poc/backend/backend_node.py` |
| EWA / vMF / fBm rendering (canonical) | `fl_slam_poc/backend/rendering.py` |
| Post-run Rerun from splat NPZ + trajectory | `tools/build_rerun_from_splat.py` |
| Spec (map architecture, topology, rendering contract) | `docs/GC_SLAM.md` (§5.7, §11) |

This should give a complete in-depth picture of map components, map update flow, and how map visualization is supposed to work in your codebase.
