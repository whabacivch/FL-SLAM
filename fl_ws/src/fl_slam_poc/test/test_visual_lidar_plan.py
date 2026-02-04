"""
Validation tests for Visual–LiDAR integration plan (GC v2).

Asserts:
- Manifest: pose_evidence_backend and map_backend present and valid; single path.
- Fixed-cost: N_FEAT, N_SURFEL, K_ASSOC, K_SINKHORN, RINGBUF_LEN are defined and constant.
- CertBundle: approximation_triggers ≠ ∅ ⇒ frobenius_applied.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md Section 14
"""

import os
import pytest
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import CHART_ID_GC_RIGHT_01
from fl_slam_poc.common.certificates import CertBundle
from fl_slam_poc.backend.pipeline import RuntimeManifest


# Valid backend values per constants
VALID_POSE_EVIDENCE_BACKENDS = (constants.GC_POSE_EVIDENCE_BACKEND_PRIMITIVES,)
VALID_MAP_BACKENDS = (constants.GC_MAP_BACKEND_PRIMITIVE_MAP,)


def cert_satisfies_frobenius_policy(cert: CertBundle) -> bool:
    """Invariant: approximation_triggers ≠ ∅ ⇒ frobenius_applied."""
    return not cert.approximation_triggers or cert.frobenius_applied


class TestManifestPoseEvidenceAndMapBackend:
    """Manifest must include pose_evidence_backend and map_backend."""

    def test_manifest_to_dict_has_pose_evidence_backend(self):
        manifest = RuntimeManifest(
            pose_evidence_backend=constants.GC_POSE_EVIDENCE_BACKEND_PRIMITIVES,
            map_backend=constants.GC_MAP_BACKEND_PRIMITIVE_MAP,
        )
        d = manifest.to_dict()
        assert "pose_evidence_backend" in d
        assert d["pose_evidence_backend"] in VALID_POSE_EVIDENCE_BACKENDS

    def test_manifest_to_dict_has_map_backend(self):
        manifest = RuntimeManifest(
            pose_evidence_backend=constants.GC_POSE_EVIDENCE_BACKEND_PRIMITIVES,
            map_backend=constants.GC_MAP_BACKEND_PRIMITIVE_MAP,
        )
        d = manifest.to_dict()
        assert "map_backend" in d
        assert d["map_backend"] in VALID_MAP_BACKENDS

    def test_manifest_primitives_path_sets_lidar_evidence_backend(self):
        """When pose_evidence_backend=primitives, backends[lidar_evidence] must be visual_pose_evidence."""
        manifest = RuntimeManifest(
            pose_evidence_backend=constants.GC_POSE_EVIDENCE_BACKEND_PRIMITIVES,
            map_backend=constants.GC_MAP_BACKEND_PRIMITIVE_MAP,
        )
        d = manifest.to_dict()
        assert "visual_pose_evidence" in d["backends"]["lidar_evidence"]

    def test_manifest_primitive_map_sets_map_update_backend(self):
        """When map_backend=primitive_map, backends[map_update] must be primitive_map."""
        manifest = RuntimeManifest(
            pose_evidence_backend=constants.GC_POSE_EVIDENCE_BACKEND_PRIMITIVES,
            map_backend=constants.GC_MAP_BACKEND_PRIMITIVE_MAP,
        )
        d = manifest.to_dict()
        assert "primitive_map" in d["backends"]["map_update"]


class TestSinglePoseEvidencePath:
    """Only one pose-evidence path is active (primitives only)."""

    def test_pose_evidence_backend_is_primitives(self):
        for backend in VALID_POSE_EVIDENCE_BACKENDS:
            manifest = RuntimeManifest(pose_evidence_backend=backend, map_backend=constants.GC_MAP_BACKEND_PRIMITIVE_MAP)
            d = manifest.to_dict()
            assert d["pose_evidence_backend"] in VALID_POSE_EVIDENCE_BACKENDS


class TestFixedCostBudgets:
    """Fixed budgets N_FEAT, N_SURFEL, K_ASSOC, K_SINKHORN, RINGBUF_LEN are defined and constant."""

    def test_n_feat_defined_and_positive(self):
        assert hasattr(constants, "GC_N_FEAT")
        assert constants.GC_N_FEAT > 0

    def test_n_surfel_defined_and_positive(self):
        assert hasattr(constants, "GC_N_SURFEL")
        assert constants.GC_N_SURFEL > 0

    def test_k_assoc_defined_and_positive(self):
        assert hasattr(constants, "GC_K_ASSOC")
        assert constants.GC_K_ASSOC > 0

    def test_k_sinkhorn_defined_and_positive(self):
        assert hasattr(constants, "GC_K_SINKHORN")
        assert constants.GC_K_SINKHORN > 0

    def test_ringbuf_len_defined_and_positive(self):
        assert hasattr(constants, "GC_RINGBUF_LEN")
        assert constants.GC_RINGBUF_LEN > 0


class TestCertBundleFrobeniusPolicy:
    """approximation_triggers ≠ ∅ ⇒ frobenius_applied."""

    def test_exact_cert_has_no_triggers_satisfies_policy(self):
        cert = CertBundle.create_exact(chart_id=CHART_ID_GC_RIGHT_01, anchor_id="test")
        assert cert_satisfies_frobenius_policy(cert)

    def test_approx_cert_with_triggers_and_frobenius_satisfies_policy(self):
        cert = CertBundle.create_approx(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            triggers=["linearization"],
            frobenius_applied=True,
        )
        assert cert_satisfies_frobenius_policy(cert)

    def test_approx_cert_with_triggers_without_frobenius_violates_policy(self):
        """When triggers are nonempty and frobenius_applied is False, policy is violated."""
        cert = CertBundle.create_approx(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            triggers=["linearization"],
            frobenius_applied=False,
        )
        assert not cert_satisfies_frobenius_policy(cert)


class TestLegacyDiagnosticsRemoval:
    """Legacy diagnostics are removed; minimal tape only."""

    def test_no_scan_diagnostics_symbol(self):
        from fl_slam_poc.backend import diagnostics

        assert not hasattr(diagnostics, "ScanDiagnostics")

    def test_config_has_no_save_full_diagnostics(self):
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "config",
            "gc_unified.yaml",
        )
        with open(config_path, "r") as f:
            cfg = f.read()
        assert "save_full_diagnostics" not in cfg


class TestStatusSchemaNoLegacyFields:
    """Status/auditor schemas do not reference legacy fields."""

    def test_auditor_has_no_map_bins_active(self):
        auditor_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "fl_slam_poc",
            "frontend",
            "audit",
            "wiring_auditor.py",
        )
        with open(auditor_path, "r") as f:
            contents = f.read()
        assert "map_bins_active" not in contents
