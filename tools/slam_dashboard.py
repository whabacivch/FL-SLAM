#!/usr/bin/env python3
"""
Geometric Compositional SLAM v2 diagnostics dashboard (minimal tape + cert summary).

This dashboard is designed for the minimal diagnostics tape produced by the backend.
It focuses on high-signal, certificate-derived metrics plus the pose6 evidence block
and trajectory visualization.

Usage:
  tools/slam_dashboard.py <diagnostics.npz> [--output dashboard.html] [--scan N] [--ground-truth path.tum]
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

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
FL_WS_SRC = PROJECT_ROOT / "fl_ws" / "src" / "fl_slam_poc"
sys.path.insert(0, str(FL_WS_SRC))


def load_diagnostics_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_tum_positions(path: str):
    try:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
        if not rows:
            return None
        arr = np.array(rows, dtype=np.float64)
        return arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 0]
    except (OSError, ValueError) as e:
        print(f"Warning: Could not load trajectory from {path}: {e}")
        return None


def interpolate_gt_at_times(gt_x, gt_y, gt_z, gt_ts, query_ts):
    if gt_ts.size < 2 or query_ts.size == 0:
        if gt_ts.size == 1 and query_ts.size > 0:
            return np.full_like(query_ts, gt_x[0]), np.full_like(query_ts, gt_y[0]), np.full_like(query_ts, gt_z[0])
        return np.array([]), np.array([]), np.array([])
    x_out = np.interp(query_ts, gt_ts, gt_x)
    y_out = np.interp(query_ts, gt_ts, gt_y)
    z_out = np.interp(query_ts, gt_ts, gt_z)
    return x_out, y_out, z_out


def numpy_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_json(v) for v in obj]
    return obj


def open_browser_wayland_compatible(file_path: str) -> bool:
    file_path = os.path.abspath(file_path)
    file_url = f"file://{urllib.parse.quote(file_path, safe='/')}"
    try:
        subprocess.Popen(
            ["xdg-open", file_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except (FileNotFoundError, OSError):
        pass

    browser = os.environ.get("BROWSER")
    if browser:
        try:
            if "%s" in browser or "%u" in browser:
                cmd = browser.replace("%s", file_url).replace("%u", file_url).split()
            else:
                cmd = [browser, file_url]
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except (FileNotFoundError, OSError):
            pass

    try:
        import webbrowser
        webbrowser.open(file_url)
        return True
    except Exception:
        return False


def _safe_array(data: dict, key: str, n: int, default: float = 0.0) -> np.ndarray:
    if key in data:
        arr = np.asarray(data[key])
        return arr.astype(np.float64)
    return np.full((n,), default, dtype=np.float64)


def create_dashboard(
    data: dict,
    diagnostics_path: str,
    selected_scan: int = 0,
    output_path: str | None = None,
    ground_truth_path: str | None = None,
    gt_y_flip: bool = False,
) -> str | None:
    n_scans = int(data.get("n_scans", 0))
    if n_scans == 0:
        print("No scan data found in file")
        return None

    if str(data.get("format", "")) != "minimal_tape":
        print("Warning: diagnostics file is not minimal_tape format; attempting best-effort load.")

    timestamps = np.asarray(data.get("timestamps", np.arange(n_scans)), dtype=np.float64)
    scan_idx = np.arange(n_scans, dtype=np.int32)

    cond_pose6 = _safe_array(data, "cond_pose6", n_scans, default=1.0)
    eigmin_pose6 = _safe_array(data, "eigmin_pose6", n_scans, default=1e-12)
    log10_cond_pose6 = np.log10(np.maximum(cond_pose6, 1.0))
    log10_eigmin_pose6 = np.log10(np.maximum(eigmin_pose6, 1e-12))

    support_frac = _safe_array(data, "support_frac", n_scans, default=0.0)
    support_ess = _safe_array(data, "support_ess_total", n_scans, default=0.0)
    log10_support_ess = np.log10(np.maximum(support_ess, 1e-12))
    mismatch_nll = _safe_array(data, "mismatch_nll_per_ess", n_scans, default=0.0)

    fusion_alpha = _safe_array(data, "fusion_alpha", n_scans, default=1.0)
    trust_alpha = _safe_array(data, "influence_trust_alpha", n_scans, default=1.0)
    power_beta = _safe_array(data, "influence_power_beta", n_scans, default=1.0)
    dt_scale = _safe_array(data, "influence_dt_scale", n_scans, default=1.0)
    ex_scale = _safe_array(data, "influence_extrinsic_scale", n_scans, default=1.0)

    total_trigger = _safe_array(data, "total_trigger_magnitude", n_scans, default=0.0)
    n_triggers = _safe_array(data, "cert_n_triggers", n_scans, default=0.0)
    frob = _safe_array(data, "cert_frobenius_applied", n_scans, default=0.0)

    over_dt_asym = _safe_array(data, "overconfidence_dt_asymmetry", n_scans, default=0.0)
    over_z_ratio = _safe_array(data, "overconfidence_z_to_xy_ratio", n_scans, default=0.0)
    over_cond_to_support = _safe_array(data, "overconfidence_cond_to_support", n_scans, default=0.0)
    over_ess_to_exc = _safe_array(data, "overconfidence_ess_to_excitation", n_scans, default=0.0)

    L_pose6 = np.asarray(data.get("L_pose6", np.zeros((n_scans, 6, 6))), dtype=np.float64)
    if L_pose6.ndim == 2:
        L_pose6 = L_pose6[np.newaxis, :, :]

    # Estimated trajectory
    traj_est = None
    results_dir = Path(diagnostics_path).resolve().parent
    est_path = results_dir / "estimated_trajectory.tum"
    if est_path.exists():
        traj_est = load_tum_positions(str(est_path))

    traj_gt = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        traj_gt = load_tum_positions(ground_truth_path)
        if traj_gt is None:
            print("Warning: ground truth path could not be loaded.")

    if traj_gt is not None and gt_y_flip:
        gt_x, gt_y, gt_z, gt_ts = traj_gt
        traj_gt = (gt_x, -gt_y, gt_z, gt_ts)

    # Align trajectories to start at origin for visualization.
    if traj_est is not None:
        ex, ey, ez, ets = traj_est
        ex0, ey0, ez0 = ex[0], ey[0], ez[0]
        traj_est = (ex - ex0, ey - ey0, ez - ez0, ets)

    if traj_gt is not None:
        gx, gy, gz, gts = traj_gt
        gx0, gy0, gz0 = gx[0], gy[0], gz[0]
        traj_gt = (gx - gx0, gy - gy0, gz - gz0, gts)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="gc_slam_dashboard_", suffix=".html")
        os.close(fd)

    data_json = json.dumps(
        numpy_to_json(
            {
                "n_scans": n_scans,
                "scan_idx": scan_idx,
                "timestamps": timestamps,
                "log10_cond_pose6": log10_cond_pose6,
                "log10_eigmin_pose6": log10_eigmin_pose6,
                "support_frac": support_frac,
                "log10_support_ess": log10_support_ess,
                "mismatch_nll_per_ess": mismatch_nll,
                "fusion_alpha": fusion_alpha,
                "trust_alpha": trust_alpha,
                "power_beta": power_beta,
                "dt_scale": dt_scale,
                "ex_scale": ex_scale,
                "total_trigger": total_trigger,
                "n_triggers": n_triggers,
                "frob": frob,
                "over_dt_asym": over_dt_asym,
                "over_z_ratio": over_z_ratio,
                "over_cond_to_support": over_cond_to_support,
                "over_ess_to_exc": over_ess_to_exc,
                "L_pose6": L_pose6,
            }
        )
    )

    traj_json = json.dumps(
        numpy_to_json(
            {
                "est": traj_est,
                "gt": traj_gt,
            }
        )
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>GC SLAM Diagnostics (Minimal Tape)</title>
  <script src=\"https://cdn.plot.ly/plotly-2.30.0.min.js\"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; background: #0f1115; color: #e9eef5; }}
    h1 {{ margin: 0 0 8px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .panel {{ background: #141821; padding: 12px; border-radius: 10px; }}
    .full {{ grid-column: 1 / -1; }}
    #scan-summary {{ white-space: pre; font-family: monospace; font-size: 12px; }}
    input[type=range] {{ width: 100%; }}
  </style>
</head>
<body>
  <h1>GC SLAM Diagnostics (Minimal Tape)</h1>
  <div class=\"panel full\">
    <label for=\"scan-slider\">Scan: <span id=\"scan-label\"></span></label>
    <input id=\"scan-slider\" type=\"range\" min=\"0\" max=\"{n_scans - 1}\" value=\"{selected_scan}\" />
  </div>

  <div class=\"grid\">
    <div class=\"panel\"><div id=\"plot-conditioning\"></div></div>
    <div class=\"panel\"><div id=\"plot-support\"></div></div>
    <div class=\"panel\"><div id=\"plot-influence\"></div></div>
    <div class=\"panel\"><div id=\"plot-approx\"></div></div>
    <div class=\"panel\"><div id=\"plot-heatmap\"></div></div>
    <div class=\"panel\"><div id=\"plot-trajectory\"></div></div>
    <div class=\"panel full\">
      <div id=\"scan-summary\"></div>
    </div>
  </div>

<script>
const diagData = {data_json};
const trajData = {traj_json};
const scanSlider = document.getElementById('scan-slider');
const scanLabel = document.getElementById('scan-label');
const scanSummary = document.getElementById('scan-summary');

function lineTrace(name, y, color) {{
  return {{ x: diagData.scan_idx, y: y, mode: 'lines', name: name, line: {{ color: color }} }};
}}

function highlightTrace(x, y) {{
  return {{ x: [x], y: [y], mode: 'markers', marker: {{ color: '#ff6b6b', size: 8 }}, name: 'scan' }};
}}

function renderPlots(scanIdx) {{
  scanLabel.textContent = scanIdx;

  const condPlot = [
    lineTrace('log10 cond(pose6)', diagData.log10_cond_pose6, '#5ddcff'),
    lineTrace('log10 eigmin(pose6)', diagData.log10_eigmin_pose6, '#f7b731'),
    highlightTrace(scanIdx, diagData.log10_cond_pose6[scanIdx])
  ];
  Plotly.newPlot('plot-conditioning', condPlot, {{ title: 'Conditioning', paper_bgcolor: '#141821', plot_bgcolor: '#141821', font: {{color:'#e9eef5'}} }});

  const supportPlot = [
    lineTrace('support_frac', diagData.support_frac, '#9b59b6'),
    lineTrace('log10 ESS', diagData.log10_support_ess, '#2ecc71'),
    lineTrace('mismatch nll/ess', diagData.mismatch_nll_per_ess, '#ff9f43'),
    highlightTrace(scanIdx, diagData.support_frac[scanIdx])
  ];
  Plotly.newPlot('plot-support', supportPlot, {{ title: 'Support & Mismatch', paper_bgcolor: '#141821', plot_bgcolor: '#141821', font: {{color:'#e9eef5'}} }});

  const inflPlot = [
    lineTrace('fusion_alpha', diagData.fusion_alpha, '#1dd1a1'),
    lineTrace('trust_alpha', diagData.trust_alpha, '#54a0ff'),
    lineTrace('power_beta', diagData.power_beta, '#ff6b6b'),
    lineTrace('dt_scale', diagData.dt_scale, '#feca57'),
    lineTrace('ex_scale', diagData.ex_scale, '#5f27cd'),
    highlightTrace(scanIdx, diagData.fusion_alpha[scanIdx])
  ];
  Plotly.newPlot('plot-influence', inflPlot, {{ title: 'Influence / Tempering', paper_bgcolor: '#141821', plot_bgcolor: '#141821', font: {{color:'#e9eef5'}} }});

  const approxPlot = [
    lineTrace('trigger_mag', diagData.total_trigger, '#ff9f43'),
    lineTrace('n_triggers', diagData.n_triggers, '#48dbfb'),
    lineTrace('frobenius (0/1)', diagData.frob, '#ff6b6b'),
    lineTrace('dt_asym', diagData.over_dt_asym, '#f368e0'),
    lineTrace('z/xy ratio', diagData.over_z_ratio, '#10ac84'),
    lineTrace('cond/support', diagData.over_cond_to_support, '#54a0ff'),
    lineTrace('ess/excitation', diagData.over_ess_to_exc, '#c8d6e5'),
    highlightTrace(scanIdx, diagData.total_trigger[scanIdx])
  ];
  Plotly.newPlot('plot-approx', approxPlot, {{ title: 'Approx / Overconfidence', paper_bgcolor: '#141821', plot_bgcolor: '#141821', font: {{color:'#e9eef5'}} }});

  const heatmap = [{
    z: diagData.L_pose6[scanIdx],
    type: 'heatmap',
    colorscale: 'Viridis'
  }];
  Plotly.newPlot('plot-heatmap', heatmap, {{ title: 'L_pose6 Heatmap', paper_bgcolor: '#141821', plot_bgcolor: '#141821', font: {{color:'#e9eef5'}} }});

  const trajTraces = [];
  if (trajData.est) {{
    const [x, y, z] = trajData.est;
    trajTraces.push({{ x, y, z, mode: 'lines', type: 'scatter3d', name: 'estimate', line: {{ color: '#54a0ff' }} }});
  }}
  if (trajData.gt) {{
    const [x, y, z] = trajData.gt;
    trajTraces.push({{ x, y, z, mode: 'lines', type: 'scatter3d', name: 'ground_truth', line: {{ color: '#feca57' }} }});
  }}
  Plotly.newPlot('plot-trajectory', trajTraces, {{ title: 'Trajectory', scene: {{ bgcolor: '#141821' }}, paper_bgcolor: '#141821', font: {{color:'#e9eef5'}} }});

  const summary = {{
    scan: scanIdx,
    timestamp: diagData.timestamps[scanIdx],
    cond_pose6: Math.pow(10, diagData.log10_cond_pose6[scanIdx]).toFixed(3),
    eigmin_pose6: Math.pow(10, diagData.log10_eigmin_pose6[scanIdx]).toExponential(3),
    support_frac: diagData.support_frac[scanIdx].toFixed(3),
    support_ess: Math.pow(10, diagData.log10_support_ess[scanIdx]).toFixed(2),
    mismatch_nll_per_ess: diagData.mismatch_nll_per_ess[scanIdx].toFixed(3),
    fusion_alpha: diagData.fusion_alpha[scanIdx].toFixed(3),
    trust_alpha: diagData.trust_alpha[scanIdx].toFixed(3),
    power_beta: diagData.power_beta[scanIdx].toFixed(3),
    total_trigger_mag: diagData.total_trigger[scanIdx].toFixed(3),
    n_triggers: diagData.n_triggers[scanIdx],
    frobenius_applied: diagData.frob[scanIdx] > 0.5
  }};
  scanSummary.textContent = JSON.stringify(summary, null, 2);
}}

renderPlots(parseInt(scanSlider.value, 10));
scanSlider.addEventListener('input', (e) => renderPlots(parseInt(e.target.value, 10)));
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="GC SLAM diagnostics dashboard (minimal tape)")
    parser.add_argument("diagnostics_file", help="Path to diagnostics NPZ file")
    parser.add_argument("--output", default=None, help="Path to output HTML file")
    parser.add_argument("--scan", type=int, default=0, help="Initial scan index")
    parser.add_argument("--ground-truth", default=None, help="Path to ground truth TUM file")
    parser.add_argument("--gt-y-flip", action="store_true", help="Flip GT Y axis")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the browser")
    args = parser.parse_args()

    if not os.path.exists(args.diagnostics_file):
        print(f"Error: File not found: {args.diagnostics_file}")
        sys.exit(1)

    print(f"Loading diagnostics from: {args.diagnostics_file}")
    data = load_diagnostics_npz(args.diagnostics_file)
    if int(data.get("n_scans", 0)) == 0:
        print("The diagnostics file exists but contains 0 scans.")
        sys.exit(1)

    output_path = create_dashboard(
        data=data,
        diagnostics_path=args.diagnostics_file,
        selected_scan=args.scan,
        output_path=args.output,
        ground_truth_path=args.ground_truth,
        gt_y_flip=args.gt_y_flip,
    )

    if output_path:
        print(f"Dashboard saved to: {output_path}")
        if not args.no_open:
            open_browser_wayland_compatible(output_path)


if __name__ == "__main__":
    main()
