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
GC_T_SLICES = 5  # Deskew time slices per scan (fixed)
GC_SIGMA_POINTS = 45  # 2 * D_DESKEW + 1 (fixed)
GC_N_POINTS_CAP = 8192  # Max LiDAR points per scan per hypothesis (fixed)

# Epsilon constants (domain stabilization)
GC_EPS_PSD = 1e-12  # Minimum eigenvalue for PSD projection
GC_EPS_LIFT = 1e-9  # Lift for SPD solves (configurable via yaml)
GC_EPS_MASS = 1e-12  # Mass regularization for InvMass
GC_EPS_R = 1e-6  # Clamp epsilon for Rbar in kappa
GC_EPS_DEN = 1e-12  # Denominator regularization in kappa

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

# Test-only invariants still referenced by active test suite.
N_MIN_SE3_DOF = 6  # SE(3) has 6 DOF, need at least 6 constraints
K_SIGMOID = 0.5  # Chosen so w(n=6) ≈ 0.5, w(n=12) ≈ 0.95
