#!/usr/bin/env python3
"""
Ground-truth sanity checks for evaluation (timestamp overlap + minimum samples).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def _load_timestamps(path: Path) -> np.ndarray:
    ts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 1:
                ts.append(float(parts[0]))
    return np.asarray(ts, dtype=np.float64)


def check_overlap(gt_file: Path, est_file: Path, min_overlap_frac: float, min_points: int) -> Tuple[bool, dict]:
    gt_ts = _load_timestamps(gt_file)
    est_ts = _load_timestamps(est_file)

    result = {
        "gt_file": str(gt_file),
        "est_file": str(est_file),
        "gt_points": int(gt_ts.size),
        "est_points": int(est_ts.size),
        "min_overlap_frac": float(min_overlap_frac),
        "min_points": int(min_points),
    }

    if gt_ts.size < min_points or est_ts.size < min_points:
        result["status"] = "fail"
        result["reason"] = "too_few_points"
        return False, result

    gt_min, gt_max = float(np.min(gt_ts)), float(np.max(gt_ts))
    est_min, est_max = float(np.min(est_ts)), float(np.max(est_ts))
    result.update({
        "gt_min": gt_min,
        "gt_max": gt_max,
        "est_min": est_min,
        "est_max": est_max,
    })

    overlap = max(0.0, min(gt_max, est_max) - max(gt_min, est_min))
    min_range = min(gt_max - gt_min, est_max - est_min)
    overlap_frac = overlap / min_range if min_range > 0 else 0.0

    result["overlap_sec"] = overlap
    result["overlap_frac"] = overlap_frac

    if overlap_frac < min_overlap_frac:
        result["status"] = "fail"
        result["reason"] = "insufficient_overlap"
        return False, result

    result["status"] = "pass"
    return True, result


def main() -> int:
    ap = argparse.ArgumentParser(description="Check GT/EST timestamp overlap for evaluation")
    ap.add_argument("--gt", required=True, help="Ground truth TUM file")
    ap.add_argument("--est", required=True, help="Estimated trajectory TUM file")
    ap.add_argument("--min-overlap", type=float, default=0.5, help="Minimum overlap fraction")
    ap.add_argument("--min-points", type=int, default=10, help="Minimum points in each file")
    ap.add_argument("--out", required=False, help="Output JSON report path")
    args = ap.parse_args()

    gt_path = Path(args.gt)
    est_path = Path(args.est)

    ok, report = check_overlap(gt_path, est_path, args.min_overlap, args.min_points)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
