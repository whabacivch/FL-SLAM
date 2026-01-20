"""
FL-SLAM Constants and Configuration Values.

All magic numbers are centralized here with clear documentation.
"""

# =============================================================================
# Depth Processing Constants
# =============================================================================

# Valid depth range for depth camera processing (meters)
DEPTH_MIN_VALID = 0.1  # Minimum valid depth (10cm) - closer is sensor noise
DEPTH_MAX_VALID = 10.0  # Maximum valid depth (10m) - farther is unreliable

# Depth image stride for point cloud generation
DEPTH_STRIDE_DEFAULT = 4  # Process every 4th pixel (balance speed vs density)

# =============================================================================
# Numerical Stability Thresholds
# =============================================================================

# Epsilon for weight/probability comparisons
WEIGHT_EPSILON = 1e-12  # Weights below this are considered zero

# Epsilon for covariance regularization
COV_REGULARIZATION_MIN = 1e-9  # Minimum eigenvalue for positive definiteness

# Epsilon for numerical comparisons
NUMERICAL_EPSILON = 1e-6  # General numerical tolerance

# =============================================================================
# Sensor Timeout Constants
# =============================================================================

# How long before a sensor is considered stale (seconds)
SENSOR_TIMEOUT_DEFAULT = 5.0

# Grace period after node startup before warning about missing sensors (seconds)
SENSOR_STARTUP_GRACE_PERIOD = 10.0

# =============================================================================
# Buffer and History Constants
# =============================================================================

# Maximum length for feature/sensor data buffers
FEATURE_BUFFER_MAX_LENGTH = 10

# Maximum length for state history buffer
STATE_BUFFER_MAX_LENGTH = 500

# Maximum trajectory path length for visualization
TRAJECTORY_PATH_MAX_LENGTH = 1000

# =============================================================================
# ICP Constants
# =============================================================================

# Minimum degrees of freedom for SE(3) observability
N_MIN_SE3_DOF = 6  # SE(3) has 6 DOF, need at least 6 constraints

# Sigmoid steepness for DOF weight function
K_SIGMOID = 0.5  # Chosen so w(n=6) ≈ 0.5, w(n=12) ≈ 0.95

# Default ICP convergence tolerance
ICP_TOLERANCE_DEFAULT = 1e-4

# Default ICP maximum iterations
ICP_MAX_ITER_DEFAULT = 15

# ICP reference point count for information weighting
ICP_N_REF_DEFAULT = 100.0

# ICP MSE sigma for quality weighting
ICP_SIGMA_MSE_DEFAULT = 0.01

# =============================================================================
# Descriptor Constants
# =============================================================================

# Number of bins for scan descriptors (histogram)
DESCRIPTOR_BINS_DEFAULT = 60

# =============================================================================
# Timestamp Alignment Constants
# =============================================================================

# Prior sigma for timestamp alignment (seconds)
ALIGNMENT_SIGMA_PRIOR = 0.1

# Prior strength for alignment model
ALIGNMENT_PRIOR_STRENGTH = 5.0

# Floor value for alignment sigma (minimum uncertainty)
ALIGNMENT_SIGMA_FLOOR = 0.001

# =============================================================================
# Process Noise Constants
# =============================================================================

# Prior for translational process noise (meters)
PROCESS_NOISE_TRANS_PRIOR = 0.03

# Prior for rotational process noise (radians)
PROCESS_NOISE_ROT_PRIOR = 0.015

# Prior strength for process noise model
PROCESS_NOISE_PRIOR_STRENGTH = 10.0

# =============================================================================
# NIG (Normal-Inverse-Gamma) Prior Constants
# =============================================================================

# Prior parameters for descriptor models
NIG_PRIOR_KAPPA = 1.0   # Prior belief strength
NIG_PRIOR_ALPHA = 2.0   # Shape parameter (must be > 1 for finite variance)
NIG_PRIOR_BETA = 1.0    # Scale parameter

# =============================================================================
# Birth Model Constants
# =============================================================================

# Poisson intensity for new component birth
BIRTH_INTENSITY_DEFAULT = 10.0

# Expected scan period (seconds)
SCAN_PERIOD_DEFAULT = 0.1

# Base weight for new components
BASE_COMPONENT_WEIGHT_DEFAULT = 1.0

# =============================================================================
# Fisher-Rao Distance Constants
# =============================================================================

# Prior scale for Fisher-Rao distance normalization
FR_DISTANCE_SCALE_PRIOR = 1.0

# Prior strength for Fisher-Rao scale
FR_SCALE_PRIOR_STRENGTH = 5.0

# =============================================================================
# Warning Count Limits
# =============================================================================

# Maximum times to warn about missing data before suppressing
MAX_WARNING_COUNT = 3

# =============================================================================
# Covariance Initial Values
# =============================================================================

# Initial position covariance (meters^2)
INIT_POS_COV = 0.2**2

# Initial rotation covariance (radians^2)
INIT_ROT_COV = 0.1**2

# =============================================================================
# Loop Closure Constants
# =============================================================================

# Minimum responsibility for creating an anchor
ANCHOR_RESPONSIBILITY_MIN = 0.1

# Minimum responsibility for publishing a loop factor
LOOP_RESPONSIBILITY_MIN = 0.1

# =============================================================================
# Debug/Logging Constants
# =============================================================================

# How often to log periodic status (seconds)
STATUS_CHECK_PERIOD = 5.0

# How many initial scans to log for debugging
INITIAL_SCAN_LOG_COUNT = 10

# How often to log scan count after initial period
SCAN_LOG_FREQUENCY = 20
