from pathlib import Path

from tools.gt_checks import check_overlap


def _write_tum(path: Path, timestamps):
    with path.open("w", encoding="utf-8") as f:
        f.write("# timestamp x y z qx qy qz qw\n")
        for t in timestamps:
            f.write(f"{t:.6f} 0 0 0 0 0 0 1\n")


def test_gt_overlap_pass(tmp_path: Path):
    gt = tmp_path / "gt.tum"
    est = tmp_path / "est.tum"
    _write_tum(gt, [0.0, 1.0, 2.0, 3.0])
    _write_tum(est, [1.0, 2.0, 3.0, 4.0])
    ok, report = check_overlap(gt, est, min_overlap_frac=0.5, min_points=2)
    assert ok
    assert report["status"] == "pass"


def test_gt_overlap_fail(tmp_path: Path):
    gt = tmp_path / "gt.tum"
    est = tmp_path / "est.tum"
    _write_tum(gt, [0.0, 1.0, 2.0])
    _write_tum(est, [10.0, 11.0, 12.0])
    ok, report = check_overlap(gt, est, min_overlap_frac=0.5, min_points=2)
    assert not ok
    assert report["status"] == "fail"
