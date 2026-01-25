"""
Tests for Golden Child SLAM v2 operators.

Verifies that operators follow spec contracts:
- Return (result, CertBundle, ExpectedEffect) tuple
- Are branch-free (total functions)
- Apply domain projections correctly

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5
"""

import pytest
from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import (
    BeliefGaussianInfo,
    D_Z,
    CHART_ID_GC_RIGHT_01,
    se3_identity,
)
from fl_slam_poc.common.certificates import CertBundle, ExpectedEffect


def random_array(key, shape):
    """Generate random array using JAX PRNG."""
    return jax.random.normal(key, shape)


class TestPointBudgetResample:
    """Tests for PointBudgetResample operator."""

    def test_returns_correct_tuple(self):
        """Operator should return (result, cert, expected_effect)."""
        from fl_slam_poc.backend.operators.point_budget import point_budget_resample
        
        key = jax.random.PRNGKey(42)
        points = random_array(key, (100, 3))
        timestamps = jnp.linspace(0, 1, 100)
        weights = jnp.ones(100)
        
        result, cert, effect = point_budget_resample(points, timestamps, weights)
        
        assert result is not None
        assert isinstance(cert, CertBundle)
        assert isinstance(effect, ExpectedEffect)

    def test_respects_budget(self):
        """Output should not exceed N_POINTS_CAP."""
        from fl_slam_poc.backend.operators.point_budget import point_budget_resample
        
        n_input = 10000
        key = jax.random.PRNGKey(42)
        points = random_array(key, (n_input, 3))
        timestamps = jnp.linspace(0, 1, n_input)
        weights = jnp.ones(n_input)
        
        result, _, _ = point_budget_resample(points, timestamps, weights)
        
        assert result.points.shape[0] <= constants.GC_N_POINTS_CAP

    def test_preserves_mass(self):
        """Total mass should be preserved."""
        from fl_slam_poc.backend.operators.point_budget import point_budget_resample
        
        key = jax.random.PRNGKey(42)
        points = random_array(key, (100, 3))
        timestamps = jnp.linspace(0, 1, 100)
        weights = jnp.ones(100)
        
        result, _, _ = point_budget_resample(points, timestamps, weights)
        
        # Mass should be approximately preserved
        assert abs(result.total_mass_out - result.total_mass_in) < 1e-6


class TestKappaFromResultant:
    """Tests for KappaFromResultant operator."""

    def test_returns_correct_tuple(self):
        """Operator should return (result, cert, expected_effect)."""
        from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_v2
        
        result, cert, effect = kappa_from_resultant_v2(0.5)
        
        assert result is not None
        assert isinstance(cert, CertBundle)
        assert isinstance(effect, ExpectedEffect)

    def test_is_exact_op(self):
        """Kappa computation is a closed-form ExactOp."""
        from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_v2
        
        _, cert, _ = kappa_from_resultant_v2(0.5)
        assert cert.exact is True

    def test_kappa_range(self):
        """Kappa should be non-negative for valid R_bar."""
        from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_v2
        
        for R_bar in [0.0, 0.5, 0.9, 0.99]:
            result, _, _ = kappa_from_resultant_v2(R_bar)
            assert result.kappa >= 0

    def test_kappa_monotonic(self):
        """Kappa should increase with R_bar."""
        from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_v2
        
        kappas = []
        for R_bar in [0.1, 0.5, 0.8]:
            result, _, _ = kappa_from_resultant_v2(R_bar)
            kappas.append(result.kappa)
        
        assert kappas[0] < kappas[1] < kappas[2]


class TestBinSoftAssign:
    """Tests for BinSoftAssign operator."""

    def test_returns_correct_tuple(self):
        """Operator should return (result, cert, expected_effect)."""
        from fl_slam_poc.backend.operators.binning import bin_soft_assign, create_bin_atlas
        
        n_points = 50
        n_bins = 20
        key = jax.random.PRNGKey(42)
        point_dirs = random_array(key, (n_points, 3))
        point_dirs = point_dirs / jnp.linalg.norm(point_dirs, axis=1, keepdims=True)
        bin_dirs = create_bin_atlas(n_bins)
        
        result, cert, effect = bin_soft_assign(point_dirs, bin_dirs)
        
        assert result is not None
        assert isinstance(cert, CertBundle)
        assert isinstance(effect, ExpectedEffect)

    def test_responsibilities_sum_to_one(self):
        """Each point's responsibilities should sum to 1."""
        from fl_slam_poc.backend.operators.binning import bin_soft_assign, create_bin_atlas
        
        n_points = 50
        n_bins = 20
        key = jax.random.PRNGKey(42)
        point_dirs = random_array(key, (n_points, 3))
        point_dirs = point_dirs / jnp.linalg.norm(point_dirs, axis=1, keepdims=True)
        bin_dirs = create_bin_atlas(n_bins)
        
        result, _, _ = bin_soft_assign(point_dirs, bin_dirs)
        
        row_sums = jnp.sum(result.responsibilities, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-6)

    def test_is_exact_op(self):
        """Soft assign with softmax is exact (no approximation)."""
        from fl_slam_poc.backend.operators.binning import bin_soft_assign, create_bin_atlas
        
        n_points = 50
        n_bins = 20
        key = jax.random.PRNGKey(42)
        point_dirs = random_array(key, (n_points, 3))
        point_dirs = point_dirs / jnp.linalg.norm(point_dirs, axis=1, keepdims=True)
        bin_dirs = create_bin_atlas(n_bins)
        
        _, cert, _ = bin_soft_assign(point_dirs, bin_dirs)
        
        assert cert.exact is True


class TestWahbaSVD:
    """Tests for WahbaSVD operator."""

    def test_returns_correct_tuple(self):
        """Operator should return (result, cert, expected_effect)."""
        from fl_slam_poc.backend.operators.wahba import wahba_svd
        
        n_bins = 10
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        mu_map = random_array(key1, (n_bins, 3))
        mu_map = mu_map / jnp.linalg.norm(mu_map, axis=1, keepdims=True)
        mu_scan = random_array(key2, (n_bins, 3))
        mu_scan = mu_scan / jnp.linalg.norm(mu_scan, axis=1, keepdims=True)
        weights = jnp.ones(n_bins)
        
        result, cert, effect = wahba_svd(mu_map, mu_scan, weights)
        
        assert result is not None
        assert isinstance(cert, CertBundle)
        assert isinstance(effect, ExpectedEffect)

    def test_rotation_is_valid(self):
        """Output rotation should be SO(3)."""
        from fl_slam_poc.backend.operators.wahba import wahba_svd
        
        n_bins = 10
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        mu_map = random_array(key1, (n_bins, 3))
        mu_map = mu_map / jnp.linalg.norm(mu_map, axis=1, keepdims=True)
        mu_scan = random_array(key2, (n_bins, 3))
        mu_scan = mu_scan / jnp.linalg.norm(mu_scan, axis=1, keepdims=True)
        weights = jnp.ones(n_bins)
        
        result, _, _ = wahba_svd(mu_map, mu_scan, weights)
        R = result.R_hat
        
        # Check orthogonality: R^T R = I
        assert jnp.allclose(R.T @ R, jnp.eye(3), atol=1e-6)
        # Check det = 1 (not -1)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)

    def test_is_exact_op(self):
        """Wahba SVD is exact (closed-form solution)."""
        from fl_slam_poc.backend.operators.wahba import wahba_svd
        
        n_bins = 10
        key = jax.random.PRNGKey(42)
        mu_map = random_array(key, (n_bins, 3))
        mu_map = mu_map / jnp.linalg.norm(mu_map, axis=1, keepdims=True)
        mu_scan = mu_map  # Same directions
        weights = jnp.ones(n_bins)
        
        _, cert, _ = wahba_svd(mu_map, mu_scan, weights)
        
        assert cert.exact is True


class TestPredictDiffusion:
    """Tests for PredictDiffusion operator."""

    def test_returns_correct_tuple(self):
        """Operator should return (result, cert, expected_effect)."""
        from fl_slam_poc.backend.operators.predict import (
            predict_diffusion,
        )
        
        # Create a simple prior belief
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
        )
        belief = BeliefGaussianInfo(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            X_anchor=se3_identity(),
            stamp_sec=0.0,
            z_lin=jnp.zeros(D_Z),
            L=jnp.eye(D_Z),
            h=jnp.zeros(D_Z),
            cert=cert,
        )
        
        Q = jnp.eye(D_Z, dtype=jnp.float64)
        result, pred_cert, effect = predict_diffusion(belief, Q, dt_sec=0.1)
        
        assert result is not None
        assert isinstance(pred_cert, CertBundle)
        assert isinstance(effect, ExpectedEffect)

    def test_timestamp_updated(self):
        """Predicted belief should have updated timestamp."""
        from fl_slam_poc.backend.operators.predict import (
            predict_diffusion,
        )
        
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
        )
        belief = BeliefGaussianInfo(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            X_anchor=se3_identity(),
            stamp_sec=1.0,
            z_lin=jnp.zeros(D_Z),
            L=jnp.eye(D_Z),
            h=jnp.zeros(D_Z),
            cert=cert,
        )
        
        Q = jnp.eye(D_Z, dtype=jnp.float64)
        result, _, _ = predict_diffusion(belief, Q, dt_sec=0.5)
        
        assert result.stamp_sec == 1.5


class TestInfoFusionAdditive:
    """Tests for InfoFusionAdditive operator."""

    def test_returns_correct_tuple(self):
        """Operator should return (result, cert, expected_effect)."""
        from fl_slam_poc.backend.operators.fusion import info_fusion_additive
        
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
        )
        belief = BeliefGaussianInfo(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            X_anchor=se3_identity(),
            stamp_sec=0.0,
            z_lin=jnp.zeros(D_Z),
            L=jnp.eye(D_Z),
            h=jnp.zeros(D_Z),
            cert=cert,
        )
        
        L_evidence = 0.1 * jnp.eye(D_Z)
        h_evidence = jnp.zeros(D_Z)
        
        result, fuse_cert, effect = info_fusion_additive(
            belief, L_evidence, h_evidence, alpha=0.5
        )
        
        assert result is not None
        assert isinstance(fuse_cert, CertBundle)
        assert isinstance(effect, ExpectedEffect)

    def test_information_increases(self):
        """Information should increase or stay same after fusion."""
        from fl_slam_poc.backend.operators.fusion import info_fusion_additive
        
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
        )
        belief = BeliefGaussianInfo(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            X_anchor=se3_identity(),
            stamp_sec=0.0,
            z_lin=jnp.zeros(D_Z),
            L=jnp.eye(D_Z),
            h=jnp.zeros(D_Z),
            cert=cert,
        )
        
        L_evidence = 0.5 * jnp.eye(D_Z)
        h_evidence = jnp.zeros(D_Z)
        
        result, _, _ = info_fusion_additive(belief, L_evidence, h_evidence, alpha=1.0)
        
        # Trace of L should increase
        assert jnp.trace(result.L) >= jnp.trace(belief.L)


class TestHypothesisBarycenterProjection:
    """Tests for HypothesisBarycenterProjection operator."""

    def test_returns_correct_tuple(self):
        """Operator should return (result, cert, expected_effect)."""
        from fl_slam_poc.backend.operators.hypothesis import hypothesis_barycenter_projection
        
        # Create K_HYP hypotheses
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
        )
        template = BeliefGaussianInfo(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            X_anchor=se3_identity(),
            stamp_sec=0.0,
            z_lin=jnp.zeros(D_Z),
            L=jnp.eye(D_Z),
            h=jnp.zeros(D_Z),
            cert=cert,
        )
        
        hypotheses = [template.copy() for _ in range(constants.GC_K_HYP)]
        weights = jnp.ones(constants.GC_K_HYP) / constants.GC_K_HYP
        
        result, proj_cert, effect = hypothesis_barycenter_projection(
            hypotheses, weights
        )
        
        assert result is not None
        assert isinstance(proj_cert, CertBundle)
        assert isinstance(effect, ExpectedEffect)

    def test_weight_floor_enforced(self):
        """Weights should be at least HYP_WEIGHT_FLOOR (0.0025)."""
        from fl_slam_poc.backend.operators.hypothesis import hypothesis_barycenter_projection
        
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
        )
        template = BeliefGaussianInfo(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            X_anchor=se3_identity(),
            stamp_sec=0.0,
            z_lin=jnp.zeros(D_Z),
            L=jnp.eye(D_Z),
            h=jnp.zeros(D_Z),
            cert=cert,
        )
        
        hypotheses = [template.copy() for _ in range(constants.GC_K_HYP)]
        # Give weights below the floor (0.0025) - these should be lifted
        weights = jnp.array([0.998, 0.001, 0.0005, 0.0005])
        
        result, _, _ = hypothesis_barycenter_projection(hypotheses, weights)
        
        # Result should still be valid (floor adjustment should happen)
        # Floor = 0.0025, so 0.001, 0.0005, 0.0005 need lifting
        assert result.floor_adjustment > 0
