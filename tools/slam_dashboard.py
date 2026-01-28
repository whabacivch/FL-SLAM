#!/usr/bin/env python3
"""
Golden Child SLAM v2 Debugging Dashboard.

Interactive Plotly dashboard for visualizing per-scan diagnostics.

Four panels:
- Panel A: Timeline scrubber with diagnostic scalar plots
- Panel B: Evidence inspector heatmap (22x22 L matrix)
- Panel C: 3D trajectory view with local evidence field glyphs
- Panel D: Excitation & Fusion diagnostics

Usage:
    # Auto-open in browser (Wayland and X11 compatible)
    .venv/bin/python tools/slam_dashboard.py /tmp/gc_slam_diagnostics.npz
    
    # Save to file and open manually
    .venv/bin/python tools/slam_dashboard.py /tmp/gc_slam_diagnostics.npz --output dashboard.html
    
    # Start at specific scan
    .venv/bin/python tools/slam_dashboard.py /tmp/gc_slam_diagnostics.npz --scan 50
    
    # With ground truth in 3D view (auto if diagnostics are in results dir with ground_truth_aligned.tum)
    .venv/bin/python tools/slam_dashboard.py results/gc_20260128_172635/diagnostics.npz --output dashboard.html
    .venv/bin/python tools/slam_dashboard.py diagnostics.npz --ground-truth path/to/ground_truth_aligned.tum
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

import numpy as np

# Add the package to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
FL_WS_SRC = PROJECT_ROOT / "fl_ws" / "src" / "fl_slam_poc"
sys.path.insert(0, str(FL_WS_SRC))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: plotly not installed. Install with: pip install plotly")
    sys.exit(1)

try:
    from fl_slam_poc.backend.diagnostics import DiagnosticsLog
except ImportError:
    print("Warning: Could not import DiagnosticsLog, using standalone loader")
    DiagnosticsLog = None


def load_diagnostics_npz(path: str) -> dict:
    """Load diagnostics from NPZ file into a dictionary."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_tum_positions(path: str):
    """
    Load x, y, z positions from a TUM trajectory file.
    TUM format: timestamp x y z qx qy qz qw (space-separated, # for comments).
    Returns (x, y, z) as 1D numpy arrays, or None if file missing/unreadable.
    """
    try:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
        if not rows:
            return None
        arr = np.array(rows, dtype=np.float64)
        return arr[:, 0], arr[:, 1], arr[:, 2]
    except (OSError, ValueError) as e:
        print(f"Warning: Could not load ground truth from {path}: {e}")
        return None


def numpy_to_json(obj):
    """Convert numpy arrays to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_json(v) for v in obj]
    return obj


def open_browser_wayland_compatible(file_path: str) -> bool:
    """
    Open a file in the default browser, compatible with Wayland and X11.
    
    Tries multiple methods:
    1. xdg-open (works on both Wayland and X11)
    2. $BROWSER environment variable
    3. webbrowser module (fallback)
    
    Returns True if successful, False otherwise.
    """
    file_path = os.path.abspath(file_path)
    file_url = f"file://{urllib.parse.quote(file_path, safe='/')}"
    
    # Method 1: Try xdg-open (works on Wayland and X11)
    try:
        subprocess.Popen(
            ["xdg-open", file_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        return True
    except (FileNotFoundError, OSError):
        pass
    
    # Method 2: Try $BROWSER environment variable
    browser = os.environ.get("BROWSER")
    if browser:
        try:
            # Handle browsers that need the URL as an argument
            if "%s" in browser or "%u" in browser:
                cmd = browser.replace("%s", file_url).replace("%u", file_url).split()
            else:
                cmd = [browser, file_url]
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            return True
        except (FileNotFoundError, OSError):
            pass
    
    # Method 3: Fallback to webbrowser module
    try:
        import webbrowser
        webbrowser.open(file_url)
        return True
    except Exception:
        pass
    
    return False


def create_full_dashboard(
    data: dict,
    selected_scan: int = 0,
    output_path: str = None,
    ground_truth_path: str = None,
) -> str:
    """
    Create and display the fully interactive dashboard.

    All panels update dynamically when the slider is moved.
    
    Args:
        data: Diagnostics data dictionary
        selected_scan: Initial scan index to display
        output_path: Optional path to save HTML file. If None, uses temp file.
        ground_truth_path: Optional path to ground truth TUM file. If provided,
            loads GT and adds it to the 3D trajectory plot, both trajectories
            shifted so origin = first estimated pose.
    
    Returns:
        Path to the created HTML file
    """
    n_scans = int(data.get("n_scans", 0))
    if n_scans == 0:
        print("No scan data found in file")
        return None

    # Prepare all data for JavaScript
    scan_idx = list(range(n_scans))

    # Timeline data - handle missing keys gracefully
    def safe_list(key, default_val=0.0):
        if key in data:
            arr = np.array(data[key])
            return arr.tolist() if hasattr(arr, 'tolist') else list(arr)
        return [default_val] * n_scans

    timeline_data = {
        "n_scans": n_scans,
        "scan_idx": scan_idx,
        "logdet_L_total": safe_list("logdet_L_total"),
        "trace_L_total": safe_list("trace_L_total"),
        "L_dt": safe_list("L_dt"),
        "trace_L_ex": safe_list("trace_L_ex"),
        "psd_delta_fro": safe_list("psd_delta_fro"),
        "psd_min_eig_after": safe_list("psd_min_eig_after"),
        "trace_Q_mode": safe_list("trace_Q_mode"),
        "trace_Sigma_lidar_mode": safe_list("trace_Sigma_lidar_mode"),
        "s_dt": safe_list("s_dt"),
        "s_ex": safe_list("s_ex"),
        "fusion_alpha": safe_list("fusion_alpha"),
        "dt_secs": safe_list("dt_secs", 0.1),
        "dt_scan": safe_list("dt_scan", 0.1),
        "dt_int": safe_list("dt_int", 0.1),
        "num_imu_samples": safe_list("num_imu_samples", 0),
        "wahba_cost": safe_list("wahba_cost"),
        "translation_residual_norm": safe_list("translation_residual_norm"),
        # Conditioning (prefer pose6 observable subspace; fall back to full if missing)
        "conditioning_number": safe_list("conditioning_number", 1.0),  # full 22x22 (may be dominated by null dirs)
        "conditioning_pose6": safe_list("conditioning_pose6", 1.0),
        # Rotation binding diagnostics (if present)
        "rot_err_lidar_deg_pred": safe_list("rot_err_lidar_deg_pred", 0.0),
        "rot_err_lidar_deg_post": safe_list("rot_err_lidar_deg_post", 0.0),
        "rot_err_odom_deg_pred": safe_list("rot_err_odom_deg_pred", 0.0),
        "rot_err_odom_deg_post": safe_list("rot_err_odom_deg_post", 0.0),
    }

    # Precompute log10 conditioning for visualization stability
    cond_src = "conditioning_pose6" if ("conditioning_pose6" in data) else "conditioning_number"
    cond = np.asarray(timeline_data.get(cond_src, timeline_data["conditioning_number"]), dtype=float)
    cond = np.where(np.isfinite(cond) & (cond > 1.0), cond, 1.0)
    timeline_data["log10_cond_pose6"] = np.log10(cond).tolist()

    # Bin statistics
    N_bins = data.get("N_bins", np.zeros((n_scans, 48)))
    kappa_bins = data.get("kappa_bins", np.zeros((n_scans, 48)))
    if hasattr(N_bins, 'tolist'):
        N_bins = np.array(N_bins)
    if hasattr(kappa_bins, 'tolist'):
        kappa_bins = np.array(kappa_bins)

    timeline_data["sum_N"] = np.sum(N_bins, axis=1).tolist()
    timeline_data["mean_kappa"] = np.mean(kappa_bins, axis=1).tolist()
    timeline_data["mf_kappa"] = np.mean(kappa_bins, axis=1).tolist()

    # MF + degeneracy + posterior subspace health (Panel A)
    L_total_arr = np.array(data.get("L_total", np.zeros((n_scans, 22, 22))))
    s1_list, s2_list, s3_list = [], [], []
    logdet_L_pose6_list, eigmin_L_pose6_list = [], []
    for i in range(n_scans):
        L_pose6 = L_total_arr[i, 0:6, 0:6]
        eigvals = np.linalg.eigvalsh(L_pose6)
        eigvals = np.sort(eigvals)[::-1]  # descending
        s1_list.append(float(eigvals[0]) if len(eigvals) > 0 else 0.0)
        s2_list.append(float(eigvals[1]) if len(eigvals) > 1 else 0.0)
        s3_list.append(float(eigvals[2]) if len(eigvals) > 2 else 0.0)
        eigvals_pos = np.maximum(eigvals, 1e-12)
        logdet_L_pose6_list.append(float(np.sum(np.log(eigvals_pos))))
        eigmin_L_pose6_list.append(float(np.min(eigvals)))
    timeline_data["s1"] = s1_list
    timeline_data["s2"] = s2_list
    timeline_data["s3"] = s3_list
    timeline_data["logdet_L_pose6"] = logdet_L_pose6_list
    timeline_data["eigmin_L_pose6"] = eigmin_L_pose6_list
    timeline_data["log10_mf_cond"] = timeline_data["log10_cond_pose6"]

    # Factor influence (trace of pose6 block per factor) for Panel D stacked area
    for key, name in [("L_lidar", "trace_L_pose6_lidar"), ("L_odom", "trace_L_pose6_odom"),
                      ("L_imu", "trace_L_pose6_imu"), ("L_gyro", "trace_L_pose6_gyro")]:
        if key in data:
            L_fac = np.array(data[key])  # (n_scans, 22, 22)
            timeline_data[name] = [float(np.trace(L_fac[i, 0:6, 0:6])) for i in range(n_scans)]
        else:
            timeline_data[name] = [0.0] * n_scans

    # Trajectory data (estimated from diagnostics)
    p_W = data.get("p_W", np.zeros((n_scans, 3)))
    if hasattr(p_W, 'tolist'):
        p_W = np.array(p_W)
    origin = np.array(p_W[0], dtype=np.float64)
    est_x = (p_W[:, 0] - origin[0]).tolist()
    est_y = (p_W[:, 1] - origin[1]).tolist()
    est_z = (p_W[:, 2] - origin[2]).tolist()
    trajectory_data = {
        "x": est_x,
        "y": est_y,
        "z": est_z,
        "logdet": timeline_data["logdet_L_total"],
    }
    # Optional ground truth (same origin = first estimated pose)
    if ground_truth_path:
        gt_xyz = load_tum_positions(ground_truth_path)
        if gt_xyz is not None:
            gt_x_arr = gt_xyz[0] - origin[0]
            gt_y_arr = gt_xyz[1] - origin[1]
            gt_z_arr = gt_xyz[2] - origin[2]
            trajectory_data["gt_x"] = gt_x_arr.tolist()
            trajectory_data["gt_y"] = gt_y_arr.tolist()
            trajectory_data["gt_z"] = gt_z_arr.tolist()

    # L matrices for heatmap (all scans)
    L_total = data.get("L_total", np.zeros((n_scans, 22, 22)))
    if hasattr(L_total, 'tolist'):
        L_total = np.array(L_total)

    # S_bins and R_WL for direction glyphs
    S_bins = data.get("S_bins", np.zeros((n_scans, 48, 3)))
    R_WL = data.get("R_WL", np.tile(np.eye(3), (n_scans, 1, 1)))
    if hasattr(S_bins, 'tolist'):
        S_bins = np.array(S_bins)
    if hasattr(R_WL, 'tolist'):
        R_WL = np.array(R_WL)

    # Create HTML with embedded data and interactive JavaScript
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GC SLAM v2 Diagnostics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            text-align: center;
            color: #00d4ff;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            max-width: 1900px;
            margin: 0 auto;
        }}
        .panel {{
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .controls {{
            text-align: center;
            margin: 15px 0;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }}
        .controls input[type="range"] {{
            width: 400px;
            height: 8px;
            -webkit-appearance: none;
            background: #0f3460;
            border-radius: 4px;
            outline: none;
        }}
        .controls input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
        }}
        .controls label {{
            font-weight: bold;
            font-size: 16px;
        }}
        .scan-info {{
            background: #0f3460;
            padding: 8px 15px;
            border-radius: 5px;
            font-family: monospace;
        }}
        .scan-info span {{
            color: #00d4ff;
            font-weight: bold;
        }}
        #z-leak-display {{
            font-size: 1.1rem;
            font-weight: bold;
            padding: 8px 12px;
            margin-bottom: 8px;
            background: #0f3460;
            border-radius: 6px;
            color: #f7b731;
        }}
        #z-leak-display .value {{
            color: #4ecdc4;
        }}
        #scan-display {{
            font-size: 24px;
            color: #00d4ff;
            min-width: 60px;
            display: inline-block;
            text-align: center;
        }}
        .panel-title {{
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stats-row {{
            display: flex;
            gap: 15px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        .stat-box {{
            background: #0f3460;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 12px;
        }}
        .stat-box .label {{
            color: #888;
        }}
        .stat-box .value {{
            color: #00d4ff;
            font-weight: bold;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <h1>Golden Child SLAM v2 Diagnostics Dashboard</h1>
    <p class="subtitle">Interactive per-scan pipeline diagnostics | {n_scans} scans loaded</p>

	    <div class="controls">
	        <label>Scan:</label>
	        <span id="scan-display">{selected_scan}</span>
	        <input type="range" id="scan-slider" min="0" max="{n_scans - 1}" value="{selected_scan}">
	        <div class="scan-info">
	            dt: <span id="info-dt">--</span>s |
	            α: <span id="info-alpha">--</span> |
	            log|L|: <span id="info-logdet">--</span> |
	            log10 κ_pose6: <span id="info-cond">--</span> |
	            rot_lidar_post: <span id="info-rot-lidar">--</span>°
	        </div>
	    </div>

    <div class="dashboard-grid">
        <div class="panel full-width">
            <div class="panel-title">Panel A: MF + degeneracy + posterior subspace health</div>
            <div id="timeline"></div>
        </div>
        <div class="panel">
            <div class="panel-title">Panel B: L_pose6 / L_xy_yaw (toggle) + Z leak</div>
            <div id="z-leak-display"><span id="z-leak-value"></span></div>
            <div id="heatmap"></div>
        </div>
        <div class="panel">
            <div class="panel-title">Panel C: 3D Trajectory</div>
            <div id="trajectory"></div>
        </div>
        <div class="panel full-width">
            <div class="panel-title">Panel D: Factor influence ledger + top-K bin anatomy</div>
            <div id="factor-stacked"></div>
            <div id="topk-bins"></div>
        </div>
    </div>

    <script>
    // Embedded data
    const timelineData = {json.dumps(numpy_to_json(timeline_data))};
    const trajectoryData = {json.dumps(numpy_to_json(trajectory_data))};
    const L_matrices = {json.dumps(L_total.tolist())};
    const S_bins = {json.dumps(S_bins.tolist())};
    const R_WL = {json.dumps(R_WL.tolist())};
    const kappa_bins = {json.dumps(kappa_bins.tolist())};
    const N_bins = {json.dumps(N_bins.tolist())};

    let currentScan = {selected_scan};
    const nScans = {n_scans};

    // Block labels for pose heatmap (6x6: translation + rotation)
    const poseBlockBoundaries = [0, 3, 6];
    const poseBlockLabels = ['trans', 'rot'];
    let heatmapMode = 'pose6';  // 'pose6' | 'xy_yaw'

    // Dark theme layout
    const darkLayout = {{
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#0f3460',
        font: {{ color: '#eee' }},
        xaxis: {{ gridcolor: '#1a1a2e', zerolinecolor: '#1a1a2e' }},
        yaxis: {{ gridcolor: '#1a1a2e', zerolinecolor: '#1a1a2e' }},
    }};

    // =====================================================================
    // Panel A: MF + degeneracy + posterior subspace health
    // =====================================================================
	    function createTimeline() {{
	        const traces = [
            // Row 1: s1, s2, s3 (singular/value strength)
            {{ x: timelineData.scan_idx, y: timelineData.s1, name: 's1', line: {{color: '#00d4ff'}}, xaxis: 'x', yaxis: 'y' }},
            {{ x: timelineData.scan_idx, y: timelineData.s2, name: 's2', line: {{color: '#4ecdc4'}}, xaxis: 'x', yaxis: 'y' }},
            {{ x: timelineData.scan_idx, y: timelineData.s3, name: 's3', line: {{color: '#45aaf2'}}, xaxis: 'x', yaxis: 'y' }},
            // Row 2: log10(mf_cond), mf_kappa, logdet(L_pose6), eigmin(L_pose6)
            {{ x: timelineData.scan_idx, y: timelineData.log10_mf_cond, name: 'log10(mf_cond)', line: {{color: '#a55eea'}}, xaxis: 'x2', yaxis: 'y2' }},
            {{ x: timelineData.scan_idx, y: timelineData.mf_kappa, name: 'mf_kappa', line: {{color: '#f7b731'}}, xaxis: 'x2', yaxis: 'y2' }},
            {{ x: timelineData.scan_idx, y: timelineData.logdet_L_pose6, name: 'logdet(L_pose6)', line: {{color: '#26de81'}}, xaxis: 'x2', yaxis: 'y2' }},
            {{ x: timelineData.scan_idx, y: timelineData.eigmin_L_pose6, name: 'eigmin(L_pose6)', line: {{color: '#ff6b6b'}}, xaxis: 'x2', yaxis: 'y2' }},
        ];

        const layout = {{
            ...darkLayout,
            height: 420,
            showlegend: true,
            legend: {{ orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center' }},
            grid: {{ rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' }},
            xaxis: {{ ...darkLayout.xaxis, anchor: 'y', domain: [0, 1], showticklabels: false }},
            xaxis2: {{ ...darkLayout.xaxis, anchor: 'y2', domain: [0, 1], title: 'Scan Index' }},
            yaxis: {{ ...darkLayout.yaxis, domain: [0.5, 1], title: 's1, s2, s3' }},
            yaxis2: {{ ...darkLayout.yaxis, domain: [0, 0.46], title: 'log10(mf_cond), mf_κ, logdet, λmin' }},
            margin: {{ t: 50, b: 40, l: 60, r: 40 }},
            shapes: createVerticalLines(currentScan, 2),
        }};

        Plotly.newPlot('timeline', traces, layout, {{responsive: true}});
    }}

    function createVerticalLines(scanIdx, numRows) {{
        const shapes = [];
        for (let i = 0; i < numRows; i++) {{
            shapes.push({{
                type: 'line',
                x0: scanIdx, x1: scanIdx,
                y0: 0, y1: 1,
                xref: 'x' + (i === 0 ? '' : (i + 1)),
                yref: 'paper',
                line: {{ color: '#ff6b6b', width: 2, dash: 'dash' }}
            }});
        }}
        return shapes;
    }}

    // =====================================================================
    // Panel B: L_pose6 (6x6) / L_xy_yaw (3x3) toggle + Z leak indicator
    // =====================================================================
    function updateZLeakDisplay(scanIdx) {{
        const L_full = L_matrices[scanIdx];
        const L_pose = [];
        for (let i = 0; i < 6; i++) L_pose.push(L_full[i].slice(0, 6));
        const Lzz = L_pose[2][2];
        const Lxx = L_pose[0][0], Lyy = L_pose[1][1];
        const xy_mean = (Lxx + Lyy) / 2 + 1e-10;
        const zLeakRatio = Lzz / xy_mean;
        const el = document.getElementById('z-leak-value');
        if (el) {{
            el.innerHTML = 'Z leak (L[z,z]/mean(L[x,x],L[y,y])): <span class="value">' + zLeakRatio.toFixed(4) + '</span> &nbsp; L_zz=' + Lzz.toFixed(4);
        }}
    }}

    function createHeatmap(scanIdx) {{
        const L_full = L_matrices[scanIdx];
        let L_plot, axisLabels, title, rows, cols;
        const idx_xy_yaw = [0, 1, 5];  // x, y, yaw

        if (heatmapMode === 'xy_yaw') {{
            rows = 3; cols = 3;
            L_plot = [];
            for (let i = 0; i < 3; i++) {{
                const row = [];
                for (let j = 0; j < 3; j++) row.push(L_full[idx_xy_yaw[i]][idx_xy_yaw[j]]);
                L_plot.push(row);
            }}
            axisLabels = ['x', 'y', 'yaw'];
            title = 'L_xy_yaw (3×3) - Scan ' + scanIdx;
        }} else {{
            rows = 6; cols = 6;
            L_plot = [];
            for (let i = 0; i < 6; i++) L_plot.push(L_full[i].slice(0, 6));
            axisLabels = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz'];
            title = 'L_pose6 (6×6) - Scan ' + scanIdx;
        }}

        const annotations = [];
        for (let i = 0; i < rows; i++) {{
            annotations.push({{ x: i, y: -0.8, text: axisLabels[i], showarrow: false, font: {{size: 9, color: '#aaa'}} }});
            annotations.push({{ x: -0.8, y: i, text: axisLabels[i], showarrow: false, font: {{size: 9, color: '#aaa'}} }});
        }}

        const trace = {{
            z: L_plot,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmid: 0,
            colorbar: {{ title: 'Info', tickfont: {{color: '#eee'}} }}
        }};

        const layout = {{
            ...darkLayout,
            height: 420,
            title: {{ text: title, font: {{color: '#00d4ff'}} }},
            xaxis: {{ ...darkLayout.xaxis, title: 'Column', scaleanchor: 'y', constrain: 'domain' }},
            yaxis: {{ ...darkLayout.yaxis, title: 'Row', autorange: 'reversed', constrain: 'domain' }},
            annotations: annotations,
            margin: {{ t: 50, b: 50, l: 50, r: 30 }},
        }};

        Plotly.react('heatmap', [trace], layout, {{responsive: true}});
        updateZLeakDisplay(scanIdx);
    }}

    function setupHeatmapToggle() {{
        const container = document.getElementById('z-leak-display');
        if (!container) return;
        const btn = document.createElement('button');
        btn.textContent = 'Toggle L_pose6 / L_xy_yaw';
        btn.style.marginLeft = '12px';
        btn.style.padding = '4px 8px';
        btn.style.cursor = 'pointer';
        btn.style.background = '#0f3460';
        btn.style.color = '#00d4ff';
        btn.style.border = '1px solid #00d4ff';
        btn.style.borderRadius = '4px';
        btn.onclick = function() {{
            heatmapMode = heatmapMode === 'pose6' ? 'xy_yaw' : 'pose6';
            createHeatmap(currentScan);
        }};
        container.appendChild(btn);
    }}

    // =====================================================================
    // Panel C: 3D Trajectory
    // =====================================================================
    function createTrajectory(scanIdx) {{
        const traces = [];
        // Ground truth (if present; draw first so it appears behind estimated)
        if (trajectoryData.gt_x && trajectoryData.gt_y && trajectoryData.gt_z) {{
            traces.push({{
                x: trajectoryData.gt_x,
                y: trajectoryData.gt_y,
                z: trajectoryData.gt_z,
                mode: 'lines',
                type: 'scatter3d',
                line: {{ color: '#4ecdc4', width: 4 }},
                name: 'Ground truth',
                hovertemplate: 'Ground truth<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
            }});
        }}
        // Estimated trajectory
        traces.push({{
            x: trajectoryData.x, y: trajectoryData.y, z: trajectoryData.z,
            mode: 'lines+markers',
            type: 'scatter3d',
            marker: {{
                size: 3,
                color: trajectoryData.logdet,
                colorscale: 'Viridis',
                colorbar: {{ title: 'log|L|', x: 1.02, tickfont: {{color: '#eee'}} }},
                showscale: true
            }},
            line: {{ color: '#555', width: 2 }},
            name: 'Estimated',
            hovertemplate: 'Scan %{{text}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>',
            text: timelineData.scan_idx.map(i => i.toString())
        }});
        // Selected point marker
        traces.push({{
            x: [trajectoryData.x[scanIdx]],
            y: [trajectoryData.y[scanIdx]],
            z: [trajectoryData.z[scanIdx]],
            mode: 'markers',
            type: 'scatter3d',
            marker: {{ size: 10, color: '#ff6b6b', symbol: 'diamond' }},
            name: `Scan ${{scanIdx}}`
        }});

        // Add direction glyphs for top-6 bins by kappa
        const kappaArr = kappa_bins[scanIdx];
        const indices = [...Array(48).keys()].sort((a, b) => kappaArr[b] - kappaArr[a]).slice(0, 6);
        const S_scan = S_bins[scanIdx];
        const R_scan = R_WL[scanIdx];
        const p_scan = [trajectoryData.x[scanIdx], trajectoryData.y[scanIdx], trajectoryData.z[scanIdx]];

        for (const b of indices) {{
            const S_b = S_scan[b];
            const norm = Math.sqrt(S_b[0]*S_b[0] + S_b[1]*S_b[1] + S_b[2]*S_b[2]);
            if (norm < 1e-6) continue;

            const d_body = [S_b[0]/norm, S_b[1]/norm, S_b[2]/norm];
            // Rotate to world frame: d_world = R_scan @ d_body
            const d_world = [
                R_scan[0][0]*d_body[0] + R_scan[0][1]*d_body[1] + R_scan[0][2]*d_body[2],
                R_scan[1][0]*d_body[0] + R_scan[1][1]*d_body[1] + R_scan[1][2]*d_body[2],
                R_scan[2][0]*d_body[0] + R_scan[2][1]*d_body[1] + R_scan[2][2]*d_body[2]
            ];

            const length = 0.3 * Math.log(1 + kappaArr[b]);
            const endX = p_scan[0] + length * d_world[0];
            const endY = p_scan[1] + length * d_world[1];
            const endZ = p_scan[2] + length * d_world[2];

            traces.push({{
                x: [p_scan[0], endX],
                y: [p_scan[1], endY],
                z: [p_scan[2], endZ],
                mode: 'lines',
                type: 'scatter3d',
                line: {{ color: '#f7b731', width: 4 }},
                showlegend: false,
                hoverinfo: 'skip'
            }});
        }}

        const layout = {{
            ...darkLayout,
            height: 450,
            title: {{ text: `3D Trajectory (Scan ${{scanIdx}} selected)`, font: {{color: '#00d4ff'}} }},
            scene: {{
                xaxis: {{ title: 'X (m)', gridcolor: '#1a1a2e', backgroundcolor: '#0f3460' }},
                yaxis: {{ title: 'Y (m)', gridcolor: '#1a1a2e', backgroundcolor: '#0f3460' }},
                zaxis: {{ title: 'Z (m)', gridcolor: '#1a1a2e', backgroundcolor: '#0f3460' }},
                bgcolor: '#0f3460',
                aspectmode: 'data'
            }},
            margin: {{ t: 50, b: 30, l: 30, r: 30 }},
            legend: {{ x: 0, y: 1, bgcolor: 'rgba(22, 33, 62, 0.8)' }}
        }};

        Plotly.react('trajectory', traces, layout, {{responsive: true}});
    }}

    // =====================================================================
    // Panel D: Factor influence (stacked area) + top-K bins bar chart
    // =====================================================================
    const TOP_K_BINS = 12;
	    function createFactorStacked() {{
	        const hasLidar = timelineData.trace_L_pose6_lidar && timelineData.trace_L_pose6_lidar.some(v => v !== 0);
	        const hasOdom = timelineData.trace_L_pose6_odom && timelineData.trace_L_pose6_odom.some(v => v !== 0);
	        const hasImu = timelineData.trace_L_pose6_imu && timelineData.trace_L_pose6_imu.some(v => v !== 0);
	        const hasGyro = timelineData.trace_L_pose6_gyro && timelineData.trace_L_pose6_gyro.some(v => v !== 0);
	        const traces = [];
	        if (hasLidar) traces.push({{ x: timelineData.scan_idx, y: timelineData.trace_L_pose6_lidar, name: 'Lidar (tr L_pose6)', stackgroup: 'one', fill: 'tozeroy', line: {{color: '#00d4ff', width: 1}}, mode: 'lines' }});
	        if (hasOdom) traces.push({{ x: timelineData.scan_idx, y: timelineData.trace_L_pose6_odom, name: 'Odom', stackgroup: 'one', fill: 'tonexty', line: {{color: '#4ecdc4', width: 1}}, mode: 'lines' }});
	        if (hasImu) traces.push({{ x: timelineData.scan_idx, y: timelineData.trace_L_pose6_imu, name: 'IMU', stackgroup: 'one', fill: 'tonexty', line: {{color: '#f7b731', width: 1}}, mode: 'lines' }});
	        if (hasGyro) traces.push({{ x: timelineData.scan_idx, y: timelineData.trace_L_pose6_gyro, name: 'Gyro', stackgroup: 'one', fill: 'tonexty', line: {{color: '#ff6b6b', width: 1}}, mode: 'lines' }});
	        if (traces.length === 0) traces.push({{ x: timelineData.scan_idx, y: timelineData.scan_idx.map(() => 0), name: 'No factor data', line: {{color: '#888'}} }});
	        const layout = {{
	            ...darkLayout,
	            height: 280,
	            title: {{ text: 'Info added per factor (tr L_pose6)', font: {{color: '#00d4ff'}} }},
	            xaxis: {{ ...darkLayout.xaxis, title: 'Scan Index' }},
	            yaxis: {{ ...darkLayout.yaxis, title: 'tr(L_pose6)' }},
	            showlegend: true,
	            legend: {{ orientation: 'h', y: 1.02 }},
	            margin: {{ t: 40, b: 40, l: 50, r: 30 }},
	        }};
	        Plotly.newPlot('factor-stacked', traces, layout, {{responsive: true}});
	    }}

	    function createTopKBins(scanIdx) {{
	        const N = N_bins[scanIdx] || Array(48).fill(0);
	        const kappa = kappa_bins[scanIdx] || Array(48).fill(0);
	        const w_b = N.map((n, i) => n * kappa[i]);
	        const indices = [...Array(48).keys()].sort((a, b) => w_b[b] - w_b[a]).slice(0, TOP_K_BINS);
	        const binLabels = indices.map(i => 'b' + i);
	        const traceW = {{ x: binLabels, y: indices.map(i => w_b[i]), name: 'w_b (N×κ)', type: 'bar', marker: {{ color: '#00d4ff' }} }};
	        const traceN = {{ x: binLabels, y: indices.map(i => N[i]), name: 'N', type: 'bar', marker: {{ color: '#4ecdc4' }} }};
	        const traceK = {{ x: binLabels, y: indices.map(i => kappa[i]), name: 'κ', type: 'bar', marker: {{ color: '#f7b731' }} }};
	        const layout = {{
	            ...darkLayout,
	            height: 280,
	            title: {{ text: 'Top-' + TOP_K_BINS + ' bins (scan ' + scanIdx + ') — w_b, N, κ', font: {{color: '#00d4ff'}} }},
	            xaxis: {{ ...darkLayout.xaxis, title: 'Bin' }},
	            yaxis: {{ ...darkLayout.yaxis, title: 'Value' }},
	            barmode: 'group',
	            showlegend: true,
	            legend: {{ orientation: 'h', y: 1.02 }},
	            margin: {{ t: 40, b: 40, l: 50, r: 30 }},
	        }};
	        Plotly.react('topk-bins', [traceW, traceN, traceK], layout, {{responsive: true}});
	    }}

    // =====================================================================
    // Update all panels when slider changes
    // =====================================================================
	    function updateAllPanels(scanIdx) {{
        currentScan = scanIdx;

        // Update info display
        document.getElementById('scan-display').textContent = scanIdx;
	        document.getElementById('info-dt').textContent = timelineData.dt_secs[scanIdx].toFixed(3);
	        document.getElementById('info-alpha').textContent = timelineData.fusion_alpha[scanIdx].toFixed(3);
	        document.getElementById('info-logdet').textContent = timelineData.logdet_L_total[scanIdx].toFixed(1);
	        document.getElementById('info-cond').textContent = timelineData.log10_cond_pose6[scanIdx].toFixed(2);
	        document.getElementById('info-rot-lidar').textContent = timelineData.rot_err_lidar_deg_post[scanIdx].toFixed(2);

        // Update timeline vertical lines
        Plotly.relayout('timeline', {{ shapes: createVerticalLines(scanIdx, 5) }});

        // Update heatmap and Z leak
        createHeatmap(scanIdx);

        // Update trajectory
        createTrajectory(scanIdx);

        // Update top-K bins bar chart
        createTopKBins(scanIdx);
    }}

    // =====================================================================
    // Initialize
    // =====================================================================
	    document.addEventListener('DOMContentLoaded', function() {{
        createTimeline();
        setupHeatmapToggle();
        createHeatmap(currentScan);
        createTrajectory(currentScan);
        createFactorStacked();
        createTopKBins(currentScan);

        // Update info display
	        document.getElementById('info-dt').textContent = timelineData.dt_secs[currentScan].toFixed(3);
	        document.getElementById('info-alpha').textContent = timelineData.fusion_alpha[currentScan].toFixed(3);
	        document.getElementById('info-logdet').textContent = timelineData.logdet_L_total[currentScan].toFixed(1);
	        document.getElementById('info-cond').textContent = timelineData.log10_cond_pose6[currentScan].toFixed(2);
	        document.getElementById('info-rot-lidar').textContent = timelineData.rot_err_lidar_deg_post[currentScan].toFixed(2);

        // Slider event
        const slider = document.getElementById('scan-slider');
        slider.addEventListener('input', function(e) {{
            updateAllPanels(parseInt(e.target.value));
        }});

        // Click on timeline to select scan
        document.getElementById('timeline').on('plotly_click', function(data) {{
            if (data.points && data.points.length > 0) {{
                const scanIdx = data.points[0].x;
                slider.value = scanIdx;
                updateAllPanels(scanIdx);
            }}
        }});

        // Click on timeline or factor-stacked to select scan
        document.getElementById('factor-stacked').on('plotly_click', function(data) {{
            if (data.points && data.points.length > 0) {{
                const scanIdx = data.points[0].x;
                slider.value = scanIdx;
                updateAllPanels(scanIdx);
            }}
        }});
    }});
    </script>
</body>
</html>
"""

    # Write HTML file
    if output_path:
        html_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(html_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        # Use temp file
        temp_fd, html_path = tempfile.mkstemp(suffix=".html", text=True)
        os.close(temp_fd)  # Close the file descriptor, we'll open it for writing below
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Dashboard saved to: {html_path}")
    
    # Try to open in browser (only if not explicitly saving to file)
    if not output_path:
        if open_browser_wayland_compatible(html_path):
            print("Dashboard opened in browser")
        else:
            print("")
            print("Could not automatically open browser.")
            print(f"Please manually open: {html_path}")
            print("")
            print("Or use --output to save to a specific location:")
            print(f"  .venv/bin/python tools/slam_dashboard.py <diagnostics.npz> --output dashboard.html")
    
    return html_path


def main():
    parser = argparse.ArgumentParser(
        description="Golden Child SLAM v2 Debugging Dashboard"
    )
    parser.add_argument(
        "diagnostics_file",
        type=str,
        help="Path to diagnostics NPZ file (e.g., /tmp/gc_slam_diagnostics.npz)",
    )
    parser.add_argument(
        "--scan",
        type=int,
        default=0,
        help="Initial selected scan index (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save HTML file to specified path (does not auto-open browser)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground truth TUM file. If omitted and diagnostics are in a results dir, uses ground_truth_aligned.tum in the same directory.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.diagnostics_file):
        print(f"Error: File not found: {args.diagnostics_file}")
        print("")
        print("Expected file: diagnostics NPZ file from a SLAM run")
        print("This file is created by the backend node during operation.")
        print("")
        print("To generate this file, run the SLAM pipeline:")
        print("  ./tools/run_and_evaluate_gc.sh")
        print("")
        print("Or check if a previous run created it at:")
        print("  /tmp/gc_slam_diagnostics.npz")
        print("  results/gc_*/diagnostics.npz")
        sys.exit(1)

    print(f"Loading diagnostics from: {args.diagnostics_file}")
    data = load_diagnostics_npz(args.diagnostics_file)

    n_scans = int(data.get("n_scans", 0))
    print(f"Loaded {n_scans} scans")

    # Print available keys for debugging
    print(f"Available data keys: {sorted(data.keys())}")

    if n_scans == 0:
        print("")
        print("ERROR: No scan data found in file!")
        print("The diagnostics file exists but contains 0 scans.")
        print("")
        print("This could mean:")
        print("  1. The SLAM run didn't complete successfully")
        print("  2. No LiDAR scans were processed")
        print("  3. The diagnostics weren't saved properly")
        print("")
        print("Check the SLAM log for errors.")
        sys.exit(1)

    # Validate selected scan
    if args.scan >= n_scans:
        print(f"Warning: Requested scan {args.scan} >= n_scans {n_scans}, using scan 0")
        args.scan = 0

    # Resolve ground truth path (explicit, or auto from same dir as diagnostics)
    ground_truth_path = args.ground_truth
    if not ground_truth_path:
        results_dir = Path(args.diagnostics_file).resolve().parent
        auto_gt = results_dir / "ground_truth_aligned.tum"
        if auto_gt.exists():
            ground_truth_path = str(auto_gt)
            print(f"Using ground truth: {ground_truth_path}")
    elif not os.path.exists(ground_truth_path):
        print(f"Warning: Ground truth file not found: {ground_truth_path}")
        ground_truth_path = None

    # Create dashboard
    html_path = create_full_dashboard(
        data, args.scan, output_path=args.output, ground_truth_path=ground_truth_path
    )
    
    if html_path:
        print(f"\n✓ Dashboard ready at: {html_path}")
        if args.output:
            print(f"  Saved to: {os.path.abspath(args.output)}")
            print(f"  Open manually in your browser or use: xdg-open {html_path}")


if __name__ == "__main__":
    main()
