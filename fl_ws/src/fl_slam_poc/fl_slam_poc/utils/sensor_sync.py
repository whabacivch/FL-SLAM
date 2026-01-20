"""
Sensor Synchronization Utility.

Handles timestamp alignment for multi-sensor data fusion with probabilistic weighting.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import numpy as np

from fl_slam_poc.models import TimeAlignmentModel
from fl_slam_poc import constants


@dataclass
class AlignedData:
    """Result of sensor synchronization."""
    data: Any
    timestamp_offset: float  # How far off from query time (seconds)
    weight: float  # Probabilistic weight based on alignment quality
    source: str  # Which buffer/sensor this came from


class SensorSynchronizer:
    """
    Synchronizes multi-sensor data with probabilistic timestamp alignment.
    
    Handles the common pattern of:
    1. Buffering timestamped sensor data
    2. Finding nearest data to a query timestamp
    3. Computing probabilistic alignment weight
    4. Warning when alignment is poor
    
    This replaces duplicated code across scan/image/depth/odom synchronization.
    """
    
    def __init__(
        self,
        name: str,
        buffer_size: int = constants.FEATURE_BUFFER_MAX_LENGTH,
        alignment_sigma: float = constants.ALIGNMENT_SIGMA_PRIOR,
        alignment_strength: float = constants.ALIGNMENT_PRIOR_STRENGTH,
        alignment_floor: float = constants.ALIGNMENT_SIGMA_FLOOR,
        logger = None,
    ):
        """
        Initialize sensor synchronizer.
        
        Args:
            name: Sensor name for logging
            buffer_size: Maximum buffer length
            alignment_sigma: Prior sigma for alignment model
            alignment_strength: Prior strength for alignment model
            alignment_floor: Floor value for alignment sigma
            logger: Optional ROS logger for warnings
        """
        self.name = name
        self.buffer: deque = deque(maxlen=buffer_size)
        self.alignment_model = TimeAlignmentModel(
            alignment_sigma, alignment_strength, alignment_floor
        )
        self.logger = logger
        
        # Tracking for warnings
        self._warning_count = 0
        self._max_warnings = constants.MAX_WARNING_COUNT
    
    def add(self, timestamp: float, data: Any):
        """Add timestamped data to buffer."""
        self.buffer.append((timestamp, data))
    
    def get_nearest(
        self, 
        query_time: float,
        return_tuple: bool = False,
    ) -> Optional[AlignedData]:
        """
        Get nearest data to query timestamp with alignment weight.
        
        Args:
            query_time: Timestamp to align to (seconds)
            return_tuple: If True, return (data, offset) tuple for backward compat
            
        Returns:
            AlignedData with data, offset, and weight (or None if buffer empty)
            If return_tuple=True, returns (data, offset) tuple
        """
        if not self.buffer:
            if self._warning_count < self._max_warnings and self.logger:
                self.logger.warn(
                    f"SensorSync[{self.name}]: Buffer empty at t={query_time:.3f}"
                )
                self._warning_count += 1
            
            if return_tuple:
                return None, None
            return None
        
        # Find nearest
        closest = min(self.buffer, key=lambda x: abs(x[0] - query_time))
        timestamp, data = closest
        offset = query_time - timestamp
        
        # Update alignment model and compute weight
        self.alignment_model.update(offset)
        weight = self.alignment_model.weight(offset)
        
        # Backward compatibility mode
        if return_tuple:
            return data, float(offset)
        
        return AlignedData(
            data=data,
            timestamp_offset=float(offset),
            weight=float(weight),
            source=self.name,
        )
    
    def get_all_aligned(
        self, 
        query_time: float,
        max_offset: Optional[float] = None,
    ) -> List[AlignedData]:
        """
        Get all buffered data with alignment weights.
        
        Useful for multi-hypothesis or when you want to fuse multiple observations.
        
        Args:
            query_time: Timestamp to align to
            max_offset: Optional maximum time offset to include
            
        Returns:
            List of AlignedData sorted by weight (best first)
        """
        if not self.buffer:
            return []
        
        results = []
        for timestamp, data in self.buffer:
            offset = query_time - timestamp
            
            # Skip if outside time window
            if max_offset is not None and abs(offset) > max_offset:
                continue
            
            self.alignment_model.update(offset)
            weight = self.alignment_model.weight(offset)
            
            results.append(AlignedData(
                data=data,
                timestamp_offset=float(offset),
                weight=float(weight),
                source=self.name,
            ))
        
        # Sort by weight (best first)
        results.sort(key=lambda x: x.weight, reverse=True)
        return results
    
    def clear(self):
        """Clear buffer and reset statistics."""
        self.buffer.clear()
        self._warning_count = 0
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0


class MultiSensorSynchronizer:
    """
    Manages synchronization for multiple sensors.
    
    Example usage:
        sync = MultiSensorSynchronizer(logger)
        sync.add_sensor("odom", buffer_size=200)
        sync.add_sensor("scan", buffer_size=100)
        sync.add_sensor("depth", buffer_size=50)
        
        # Add data
        sync.add("odom", timestamp, odom_data)
        
        # Query all sensors
        aligned = sync.get_all_nearest(query_time)
        odom_data = aligned["odom"].data if aligned["odom"] else None
    """
    
    def __init__(self, logger=None):
        self.sensors: dict[str, SensorSynchronizer] = {}
        self.logger = logger
    
    def add_sensor(
        self,
        name: str,
        buffer_size: int = constants.FEATURE_BUFFER_MAX_LENGTH,
        **kwargs
    ):
        """Add a sensor to synchronize."""
        self.sensors[name] = SensorSynchronizer(
            name=name,
            buffer_size=buffer_size,
            logger=self.logger,
            **kwargs
        )
    
    def add(self, sensor_name: str, timestamp: float, data: Any):
        """Add data to a sensor buffer."""
        if sensor_name in self.sensors:
            self.sensors[sensor_name].add(timestamp, data)
    
    def get_nearest(self, sensor_name: str, query_time: float) -> Optional[AlignedData]:
        """Get nearest data for a specific sensor."""
        if sensor_name in self.sensors:
            return self.sensors[sensor_name].get_nearest(query_time)
        return None
    
    def get_all_nearest(self, query_time: float) -> dict[str, Optional[AlignedData]]:
        """Get nearest data for all sensors at query time."""
        return {
            name: sync.get_nearest(query_time)
            for name, sync in self.sensors.items()
        }
    
    def compute_joint_weight(self, aligned_data: dict[str, Optional[AlignedData]]) -> float:
        """
        Compute joint probability weight from multiple aligned sensors.
        
        Assumes independence: P(all aligned) = ∏ P(sensor_i aligned)
        """
        weight = 1.0
        for data in aligned_data.values():
            if data is not None:
                weight *= data.weight
        return weight
