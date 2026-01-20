# LoopFactor: Relative pose constraint between anchor and current pose
#
# CONVENTION (EXPLICIT):
#   rel_pose = Z = T_anchor^{-1} ∘ T_current
#
# This means:
#   - Z transforms points FROM current frame TO anchor frame
#   - Backend reconstruction: T_current = T_anchor ∘ Z
#   - For static robot (T_anchor == T_current), Z is identity
#
# Covariance Convention:
#   - 6x6 matrix in ROW-MAJOR order (36 elements)
#   - Basis: [δx, δy, δz, δωx, δωy, δωz] (translation first, rotation second)
#   - Expressed in se(3) tangent space at identity, transported via adjoint
#
# Weight Convention:
#   - Probabilistic weight ∈ [0, 1] representing confidence
#   - Combines: observation alignment × association responsibility × information content
#
# Reference: Barfoot (2017), Miyamoto et al. (2024)

std_msgs/Header header

# Anchor identifier this loop closes to
int32 anchor_id

# Probabilistic weight (soft association, not hard gate)
float64 weight

# Relative pose Z = T_anchor^{-1} ∘ T_current
geometry_msgs/Pose rel_pose

# Covariance in se(3) tangent space [translation, rotation], row-major 6x6
float64[36] covariance

# Approximation triggers for audit trail (e.g., "Linearization", "BudgetTruncation")
string[] approximation_triggers

# Solver metadata for proof-of-execution
string solver_name           # e.g., "ICP", "ClosedFormSVD"
float64 solver_objective     # Final objective value (e.g., MSE)
float64 solver_tolerance     # Convergence tolerance used
int32 solver_iterations      # Iterations to convergence
int32 solver_max_iterations  # Maximum iterations allowed
float64 information_weight   # Fisher information-based weight from point count
