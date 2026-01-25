"""
Golden Child SLAM v2 constants only.

Legacy constants have been moved to:
  archive/legacy_common/constants_legacy.py
"""

# =============================================================================
# GOLDEN CHILD MANIFEST CONSTANTS (RuntimeManifest)
# Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 6
# These are HARD CONSTANTS - do not modify without spec change
# =============================================================================

# Chart convention
GC_CHART_ID = "GC-RIGHT-01"  # Global chart convention for all beliefs/evidence

# State dimensions
GC_D_Z = 22  # Augmented tangent dimension
GC_D_DESKEW = 22  # Deskew tangent dimension (same as D_Z)

# Fixed-cost budgets (compile-time constants)
GC_K_HYP = 4  # Number of hypotheses, always present
GC_HYP_WEIGHT_FLOOR = 0.0025  # 0.01 / K_HYP, minimum hypothesis weight
GC_B_BINS = 48  # Atlas bins (fixed)
GC_N_POINTS_CAP = 8192  # Max LiDAR points per scan per hypothesis (fixed)

# Epsilon constants (domain stabilization)
GC_EPS_PSD = 1e-12  # Minimum eigenvalue for PSD projection
GC_EPS_LIFT = 1e-9  # Lift for SPD solves (configurable via yaml)
GC_EPS_MASS = 1e-12  # Mass regularization for InvMass
GC_EPS_R = 1e-6  # Clamp epsilon for Rbar in kappa
GC_EPS_DEN = 1e-12  # Denominator regularization in kappa
GC_EXC_EPS = 1e-12  # Domain guard for excitation ratios

# World gravity (m/s^2) in the odom/world frame used by evidence extraction.
GC_GRAVITY_W = (0.0, 0.0, -9.81)

# Trust/fusion constants
GC_ALPHA_MIN = 0.1  # Minimum fusion scale alpha
GC_ALPHA_MAX = 1.0  # Maximum fusion scale alpha
GC_KAPPA_SCALE = 1.0  # Scale for trust computation
GC_C0_COND = 1e6  # Conditioning scale for trust

# Excitation coupling constants
GC_C_DT = 1.0  # Time offset coupling constant
GC_C_EX = 1.0  # Extrinsic coupling constant
GC_C_FROB = 1.0  # Frobenius strength blending constant

# Anchor drift parameters (continuous reanchoring)
GC_ANCHOR_DRIFT_M0 = 0.5  # Position drift threshold (meters)
GC_ANCHOR_DRIFT_R0 = 0.2  # Rotation drift threshold (radians)

# State slice indices (0-based, per spec Section 1.1)
GC_SLICE_SO3_START = 0
GC_SLICE_SO3_END = 3
GC_SLICE_TRANS_START = 3
GC_SLICE_TRANS_END = 6
GC_SLICE_VEL_START = 6
GC_SLICE_VEL_END = 9
GC_SLICE_GYRO_BIAS_START = 9
GC_SLICE_GYRO_BIAS_END = 12
GC_SLICE_ACCEL_BIAS_START = 12
GC_SLICE_ACCEL_BIAS_END = 15
GC_SLICE_TIME_OFFSET_START = 15
GC_SLICE_TIME_OFFSET_END = 16
GC_SLICE_EXTRINSIC_START = 16
GC_SLICE_EXTRINSIC_END = 22

# Soft assign temperature (for BinSoftAssign)
GC_TAU_SOFT_ASSIGN = 0.1  # Default temperature (configurable)

# =============================================================================
# END GOLDEN CHILD MANIFEST CONSTANTS
# =============================================================================

# =============================================================================
# ADAPTIVE NOISE (Inverse-Wishart priors) — GC v2
# =============================================================================
#
# These are *priors/hyperpriors* (not fixed-tuned constants). They must not be
# inlined in operator code; always reference via `constants.py`.
#
# Units note:
# - IMU noise densities below are treated as continuous-time PSD values (per Hz)
#   in their natural units. Mapping into process diffusion Q is done by the
#   declared process model (see plan/spec).
#
# IW weak prior configuration:
# We store total ν, but choose ν so that (ν - p - 1) is a small positive
# pseudocount (fast adaptation) rather than making the IW mean undefined.
GC_IW_NU_WEAK_ADD = 0.5  # ν = p + 1 + GC_IW_NU_WEAK_ADD  (so ν - p - 1 = 0.5)

# Datasheet-derived PSD priors (noise density squared, per Hz)
GC_IMU_GYRO_NOISE_DENSITY = 8.7e-7   # (rad^2 / s^2) / Hz
GC_IMU_ACCEL_NOISE_DENSITY = 9.5e-5  # (m^2 / s^4) / Hz

# LiDAR residual noise proxy PSD prior (refined by IW updates later)
GC_LIDAR_NOISE_3D = 1e-3  # (m^2) / Hz (proxy)

# Default LiDAR translation-measurement covariance used by TranslationWLS (prior; adapted by IW updates).
GC_LIDAR_SIGMA_MEAS = 0.01  # isotropic 3x3 covariance scale (legacy default)

# Livox Mid-360 bucketization constants (Phase 3 part 2)
GC_LIDAR_N_LINES = 8
GC_LIDAR_N_TAGS = 3
GC_LIDAR_N_BUCKETS = GC_LIDAR_N_LINES * GC_LIDAR_N_TAGS  # 24

# Process-noise block priors for slow states (diffusion-rate units, per second).
# These are weak priors for bias/time/extrinsic drift and will be adapted by IW updates.
GC_PROCESS_BG_NOISE = 1e-8        # gyro bias diffusion prior
GC_PROCESS_BA_NOISE = 1e-6        # accel bias diffusion prior
GC_PROCESS_DT_NOISE = 1e-6        # time-offset diffusion prior
GC_PROCESS_EXTRINSIC_NOISE = 1e-8 # extrinsic diffusion prior (se(3) 6D)

# IW retention factors (forgetful prior). Applies deterministically every scan.
GC_IW_RHO_ROT = 0.995
GC_IW_RHO_TRANS = 0.99
GC_IW_RHO_VEL = 0.95
GC_IW_RHO_BG = 0.999
GC_IW_RHO_BA = 0.999
GC_IW_RHO_DT = 0.9999
GC_IW_RHO_EX = 0.9999

# Measurement-noise retention (separate from process noise; deterministic per scan)
GC_IW_RHO_MEAS_GYRO = 0.995
GC_IW_RHO_MEAS_ACCEL = 0.995
GC_IW_RHO_MEAS_LIDAR = 0.99

# Test-only invariants still referenced by active test suite.
N_MIN_SE3_DOF = 6  # SE(3) has 6 DOF, need at least 6 constraints
K_SIGMOID = 0.5  # Chosen so w(n=6) ≈ 0.5, w(n=12) ≈ 0.95
