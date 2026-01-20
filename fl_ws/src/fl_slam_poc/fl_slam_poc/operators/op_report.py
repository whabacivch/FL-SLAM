"""
Operator Report for audit compliance.

Every operator that performs an approximation MUST emit an OpReport that:
1. Declares what family mapping occurred (family_in → family_out)
2. Lists all approximation triggers (linearization, truncation, etc.)
3. If Frobenius is required and applied, provides stats on the correction
4. If Frobenius is NOT required, explicitly declares why (frobenius_required=False)

Frobenius Policy (from AGENTS.md):
    - Approximations require Frobenius correction UNLESS:
      a) allow_ablation=True (explicit ablation flag for benchmarking)
      b) frobenius_required=False with justification in notes
    
    When Frobenius IS applied, must include:
    - frobenius_operator: name of the operator
    - frobenius_delta_norm: ||delta_corrected - delta_raw||
    - frobenius_input_stats: statistics on input parameters
    - frobenius_output_stats: statistics on corrected output
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpReport:
    """
    Audit-compliant operation report.
    
    Attributes:
        name: Operator name (e.g., "GaussianPredictSE3")
        exact: True if operation is exact (no approximation)
        approximation_triggers: List of what caused approximation
        family_in: Input distribution family
        family_out: Output distribution family
        closed_form: True if no iterative solver was used
        solver_used: Name of solver if iterative (e.g., "ICP", "Newton")
        frobenius_required: Whether Frobenius correction is required
        frobenius_applied: Whether Frobenius correction was applied
        frobenius_operator: Name of Frobenius operator used
        frobenius_delta_norm: ||corrected - raw|| for the update
        frobenius_input_stats: Input statistics for audit
        frobenius_output_stats: Output statistics for audit
        domain_projection: Whether domain constraint was hit
        allow_ablation: Explicit opt-out for benchmarking
        metrics: Additional metrics for debugging
        notes: Human-readable explanation
        timestamp: When the report was generated
    """
    name: str
    exact: bool
    approximation_triggers: list[str] = field(default_factory=list)
    family_in: str = ""
    family_out: str = ""
    closed_form: bool = False
    solver_used: Optional[str] = None
    frobenius_required: Optional[bool] = None
    frobenius_applied: bool = False
    frobenius_operator: Optional[str] = None
    frobenius_delta_norm: Optional[float] = None
    frobenius_input_stats: Optional[dict] = None
    frobenius_output_stats: Optional[dict] = None
    domain_projection: bool = False
    allow_ablation: bool = False
    metrics: dict = field(default_factory=dict)
    notes: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def validate(self, allow_ablation: bool = False) -> None:
        """
        Validate the report satisfies audit requirements.
        
        Raises ValueError if validation fails.
        """
        allow_ablation = bool(allow_ablation or self.allow_ablation)
        
        # Exact operations cannot have approximation triggers
        if self.exact and self.approximation_triggers:
            raise ValueError("Exact op cannot declare approximation triggers.")
        
        # Approximations require Frobenius OR explicit exemption
        if self.approximation_triggers:
            # Determine if Frobenius is required
            frob_required = True if self.frobenius_required is None else self.frobenius_required
            
            if frob_required:
                # Frobenius IS required - must be applied OR ablation allowed
                if not self.frobenius_applied and not allow_ablation:
                    raise ValueError(
                        f"Approximation '{self.approximation_triggers}' requires "
                        "Frobenius correction (set frobenius_required=False with "
                        "justification in notes, or allow_ablation=True)."
                    )
            # else: frobenius_required=False, no correction needed (justification in notes)
        
        # If Frobenius applied, must have complete stats
        if self.frobenius_applied:
            if self.frobenius_operator is None:
                raise ValueError("Frobenius applied must name an operator.")
            if self.frobenius_delta_norm is None:
                raise ValueError("Frobenius applied must include delta norm.")
            if self.frobenius_input_stats is None:
                raise ValueError("Frobenius applied must include input stats.")
            if self.frobenius_output_stats is None:
                raise ValueError("Frobenius applied must include output stats.")
        
        # Closed-form ops should not have iterative solver
        if self.closed_form and self.solver_used is not None:
            raise ValueError("Closed-form op must not list a solver.")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "exact": self.exact,
            "approximation_triggers": list(self.approximation_triggers),
            "family_in": self.family_in,
            "family_out": self.family_out,
            "closed_form": self.closed_form,
            "solver_used": self.solver_used,
            "frobenius_required": self.frobenius_required,
            "frobenius_applied": self.frobenius_applied,
            "frobenius_operator": self.frobenius_operator,
            "frobenius_delta_norm": self.frobenius_delta_norm,
            "frobenius_input_stats": self.frobenius_input_stats,
            "frobenius_output_stats": self.frobenius_output_stats,
            "domain_projection": self.domain_projection,
            "allow_ablation": self.allow_ablation,
            "metrics": dict(self.metrics),
            "notes": self.notes,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
