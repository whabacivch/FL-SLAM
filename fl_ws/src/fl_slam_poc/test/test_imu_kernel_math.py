import numpy as np

from fl_slam_poc.backend.math.imu_kernel import imu_batched_projection_kernel
from fl_slam_poc.common.jax_init import jnp


def test_imu_kernel_shapes_and_finiteness():
    anchor_mus = jnp.zeros((1, 15))
    anchor_covs = jnp.tile(jnp.eye(15)[None, :, :], (1, 1, 1))
    current_mu = jnp.zeros(15)
    current_cov = jnp.eye(15)
    routing_weights = jnp.array([1.0])
    imu_stamps = jnp.array([0.0, 0.01])
    imu_accel = jnp.zeros((2, 3))
    imu_gyro = jnp.zeros((2, 3))
    imu_valid = jnp.array([True, True])
    R_imu = jnp.eye(9) * 1e-3
    R_nom = jnp.eye(9) * 1e-3
    gravity = jnp.array([0.0, 0.0, -9.81])

    joint_mean, joint_cov, diagnostics = imu_batched_projection_kernel(
        anchor_mus=anchor_mus,
        anchor_covs=anchor_covs,
        current_mu=current_mu,
        current_cov=current_cov,
        routing_weights=routing_weights,
        imu_stamps=imu_stamps,
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        imu_valid=imu_valid,
        R_imu=R_imu,
        R_nom=R_nom,
        gravity=gravity,
        dt_total=0.01,
    )

    joint_mean_np = np.asarray(joint_mean)
    joint_cov_np = np.asarray(joint_cov)
    h_weights_np = np.asarray(diagnostics['hellinger_weights'])
    routing_np = np.asarray(diagnostics['routing_weights'])
    hellinger_mean = float(diagnostics['hellinger_mean'])

    # Joint state is [anchor_0 (15D), current (15D)] = 30D
    assert joint_mean_np.shape == (30,)
    assert joint_cov_np.shape == (30, 30)
    assert h_weights_np.shape == (1,)
    assert routing_np.shape == (1,)
    assert np.all(np.isfinite(joint_mean_np))
    assert np.all(np.isfinite(joint_cov_np))
    assert np.all(np.isfinite(h_weights_np))
    assert np.all(np.isfinite(routing_np))
    assert 0.0 <= hellinger_mean <= 1.0
