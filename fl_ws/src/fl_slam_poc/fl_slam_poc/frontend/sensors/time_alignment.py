"""
Time alignment helper for sensor normalizers.

Aligns stream timestamps to a reference timeline using a single offset
computed from the first observed pair (ref, local). Provides drift and
monotonicity checks; no gating or heuristic corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeAlignmentState:
    offset_sec: Optional[float] = None
    last_ref_stamp: Optional[float] = None
    last_out_stamp: Optional[float] = None
    last_local_stamp: Optional[float] = None


class TimeAligner:
    """Simple offset-based time alignment with drift checks (no heuristics)."""

    def __init__(self, max_drift_sec: float) -> None:
        if max_drift_sec <= 0.0:
            raise ValueError("max_drift_sec must be > 0")
        self._max_drift_sec = float(max_drift_sec)
        self._state = TimeAlignmentState()

    @property
    def offset_ready(self) -> bool:
        return self._state.offset_sec is not None

    @property
    def offset_sec(self) -> Optional[float]:
        return self._state.offset_sec

    @property
    def last_ref_stamp(self) -> Optional[float]:
        return self._state.last_ref_stamp

    def update_reference(self, ref_stamp: float) -> None:
        self._state.last_ref_stamp = float(ref_stamp)
        if self._state.offset_sec is None or self._state.last_local_stamp is None:
            return
        aligned = float(self._state.last_local_stamp) + float(self._state.offset_sec)
        drift = abs(aligned - float(self._state.last_ref_stamp))
        if drift > self._max_drift_sec:
            raise RuntimeError(
                f"TimeAligner: drift {drift:.6f}s exceeds max_drift_sec={self._max_drift_sec:.6f}s"
            )

    def try_init_offset(self, local_stamp: float) -> Optional[float]:
        if self._state.offset_sec is not None:
            return self._state.offset_sec
        if self._state.last_ref_stamp is None:
            return None
        self._state.offset_sec = float(self._state.last_ref_stamp - float(local_stamp))
        return self._state.offset_sec

    def note_local_stamp(self, local_stamp: float) -> None:
        self._state.last_local_stamp = float(local_stamp)

    def align(self, local_stamp: float) -> float:
        if self._state.offset_sec is None:
            raise RuntimeError("TimeAligner: offset not initialized")
        aligned = float(local_stamp) + float(self._state.offset_sec)
        # Monotonicity check on output
        if self._state.last_out_stamp is not None and aligned < self._state.last_out_stamp:
            raise RuntimeError(
                f"TimeAligner: non-monotonic output timestamp {aligned:.6f} < "
                f"{self._state.last_out_stamp:.6f}"
            )
        self._state.last_out_stamp = aligned
        return aligned

    def check_drift(self, local_stamp: float) -> None:
        # Deprecated: drift is checked on reference updates using last_local_stamp.
        self.note_local_stamp(local_stamp)
