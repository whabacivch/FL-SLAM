"""
IMU processing subpackage.

Handles:
- IMU message buffering
- IMU segment extraction for keyframes
- Bias tracking

Note: IMU preintegration math is in backend/math/imu_kernel.py (Contract B).
This package handles only I/O and buffering.

Usage:
    from fl_slam_poc.frontend.imu import IMUBuffer
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class IMUMeasurement:
    """Single IMU measurement."""
    stamp: float  # Timestamp in seconds
    accel: np.ndarray  # Accelerometer reading (3,)
    gyro: np.ndarray  # Gyroscope reading (3,)


@dataclass
class IMUSegment:
    """IMU segment between two keyframes."""
    keyframe_i: int  # Reference keyframe ID
    keyframe_j: int  # Target keyframe ID
    t_i: float  # Start timestamp
    t_j: float  # End timestamp
    measurements: List[IMUMeasurement] = field(default_factory=list)
    bias_ref_bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_ref_ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    @property
    def stamps(self) -> np.ndarray:
        """Get all timestamps as array."""
        return np.array([m.stamp for m in self.measurements], dtype=np.float64)
    
    @property
    def accel(self) -> np.ndarray:
        """Get all accelerometer readings as (N, 3) array."""
        return np.array([m.accel for m in self.measurements], dtype=np.float64)
    
    @property
    def gyro(self) -> np.ndarray:
        """Get all gyroscope readings as (N, 3) array."""
        return np.array([m.gyro for m in self.measurements], dtype=np.float64)
    
    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.t_j - self.t_i


class IMUBuffer:
    """
    Circular buffer for IMU measurements.
    
    Stores raw IMU measurements and provides segment extraction
    between keyframe timestamps.
    
    Usage:
        buffer = IMUBuffer(max_duration=10.0)
        buffer.add(stamp, accel, gyro)
        segment = buffer.extract_segment(t_start, t_end, kf_i, kf_j)
    """
    
    def __init__(self, max_duration: float = 10.0, max_size: int = 10000):
        """
        Initialize IMU buffer.
        
        Args:
            max_duration: Maximum time span to keep (seconds)
            max_size: Maximum number of measurements to store
        """
        self.max_duration = max_duration
        self.max_size = max_size
        self._buffer: deque[IMUMeasurement] = deque(maxlen=max_size)
        
        # Bias tracking (updated from backend)
        self.current_bias_gyro = np.zeros(3, dtype=np.float64)
        self.current_bias_accel = np.zeros(3, dtype=np.float64)
    
    def add(self, stamp: float, accel: np.ndarray, gyro: np.ndarray) -> None:
        """
        Add IMU measurement to buffer.
        
        Args:
            stamp: Timestamp in seconds
            accel: Accelerometer reading (3,)
            gyro: Gyroscope reading (3,)
        """
        meas = IMUMeasurement(
            stamp=stamp,
            accel=np.asarray(accel, dtype=np.float64).reshape(3),
            gyro=np.asarray(gyro, dtype=np.float64).reshape(3),
        )
        self._buffer.append(meas)
        
        # Prune old measurements
        self._prune()
    
    def _prune(self) -> None:
        """Remove measurements older than max_duration from latest."""
        if len(self._buffer) == 0:
            return
        
        latest_stamp = self._buffer[-1].stamp
        cutoff = latest_stamp - self.max_duration
        
        while len(self._buffer) > 0 and self._buffer[0].stamp < cutoff:
            self._buffer.popleft()
    
    def extract_segment(
        self,
        t_start: float,
        t_end: float,
        keyframe_i: int,
        keyframe_j: int,
    ) -> Optional[IMUSegment]:
        """
        Extract IMU segment between two timestamps.
        
        Args:
            t_start: Start timestamp
            t_end: End timestamp
            keyframe_i: Reference keyframe ID
            keyframe_j: Target keyframe ID
            
        Returns:
            IMUSegment if measurements found, None otherwise
        """
        if len(self._buffer) == 0:
            return None
        
        # Find measurements in range
        measurements: List[IMUMeasurement] = []
        for meas in self._buffer:
            if t_start <= meas.stamp <= t_end:
                measurements.append(meas)
        
        if len(measurements) < 2:
            return None
        
        return IMUSegment(
            keyframe_i=keyframe_i,
            keyframe_j=keyframe_j,
            t_i=t_start,
            t_j=t_end,
            measurements=measurements,
            bias_ref_bg=self.current_bias_gyro.copy(),
            bias_ref_ba=self.current_bias_accel.copy(),
        )
    
    def update_biases(self, bias_gyro: np.ndarray, bias_accel: np.ndarray) -> None:
        """Update current bias estimates (from backend)."""
        self.current_bias_gyro = np.asarray(bias_gyro, dtype=np.float64).reshape(3)
        self.current_bias_accel = np.asarray(bias_accel, dtype=np.float64).reshape(3)
    
    @property
    def size(self) -> int:
        """Number of measurements in buffer."""
        return len(self._buffer)
    
    @property
    def time_span(self) -> float:
        """Time span covered by buffer (seconds)."""
        if len(self._buffer) < 2:
            return 0.0
        return self._buffer[-1].stamp - self._buffer[0].stamp


__all__ = [
    "IMUMeasurement",
    "IMUSegment",
    "IMUBuffer",
]
