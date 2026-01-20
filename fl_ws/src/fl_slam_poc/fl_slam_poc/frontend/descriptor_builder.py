"""
Descriptor Builder - Extraction and Composition.

Handles:
- Scan range histogram descriptors
- Image feature descriptors (placeholder)
- Depth feature descriptors (placeholder)
- Multi-modal descriptor composition
- Global NIG model maintenance

ALL mathematical operations call models.nig - NO math duplication.
"""

from typing import Optional
import numpy as np
from sensor_msgs.msg import LaserScan

from fl_slam_poc.models import NIGModel, NIG_PRIOR_KAPPA, NIG_PRIOR_ALPHA, NIG_PRIOR_BETA


class DescriptorBuilder:
    """
    Builds multi-modal descriptors from sensor data.
    
    Uses models.nig for all probabilistic descriptor operations (exact).
    """
    
    def __init__(self, descriptor_bins: int):
        """
        Args:
            descriptor_bins: Number of bins for scan descriptor histogram
        """
        self.descriptor_bins = descriptor_bins
        self.global_desc_model = None  # Global NIG model (uses models.nig)
    
    def scan_descriptor(self, msg: LaserScan) -> np.ndarray:
        """
        Extract scan range histogram descriptor.
        
        Returns:
            descriptor: np.ndarray of shape (descriptor_bins,)
        """
        ranges = np.asarray(msg.ranges, dtype=float).reshape(-1)
        valid = np.isfinite(ranges)
        valid &= (ranges >= float(msg.range_min))
        valid &= (ranges <= float(msg.range_max))
        
        if not np.any(valid):
            # Empty descriptor (all zeros)
            return np.zeros(self.descriptor_bins, dtype=float)
        
        r = ranges[valid]
        
        # Histogram of ranges
        hist, _ = np.histogram(r, bins=self.descriptor_bins, 
                               range=(float(msg.range_min), float(msg.range_max)))
        
        desc = np.asarray(hist, dtype=float)
        
        # Normalize to unit sum (probability distribution)
        desc_sum = float(np.sum(desc))
        if desc_sum > 1e-12:
            desc = desc / desc_sum
        
        return desc
    
    def image_descriptor(self, image_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Image descriptor is NOT used for loop closure (RGB is in dense evidence).
        
        Return None to keep descriptor dimension fixed at descriptor_bins.
        """
        # RGB contributes via dense evidence pipeline, not descriptor channels
        return None
    
    def depth_descriptor(self, points: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Depth descriptor is NOT used for loop closure (just for ICP points).
        
        Return None to keep descriptor dimension fixed at descriptor_bins.
        """
        # Depth contributes via 3D points for ICP, not descriptor channels
        return None
    
    def compose_descriptor(self, 
                          scan_desc: np.ndarray,
                          image_feat: Optional[np.ndarray],
                          depth_feat: Optional[np.ndarray]) -> np.ndarray:
        """
        Compose multi-modal descriptor by concatenation.
        
        Returns:
            descriptor: Concatenated descriptor vector
        """
        # CRITICAL: Descriptor dimensionality must be FIXED over time.
        # If a modality is missing at a given timestep (e.g., depth TF not ready),
        # we insert a zero placeholder to keep the NIG model shapes consistent.
        parts = [np.asarray(scan_desc, dtype=float).reshape(-1)]

        # Image feature slot (currently 1D placeholder)
        if image_feat is None:
            parts.append(np.zeros(1, dtype=float))
        else:
            parts.append(np.asarray(image_feat, dtype=float).reshape(-1))

        # Depth feature slot (currently 1D placeholder)
        if depth_feat is None:
            parts.append(np.zeros(1, dtype=float))
        else:
            parts.append(np.asarray(depth_feat, dtype=float).reshape(-1))

        return np.concatenate(parts, axis=0)
    
    def init_global_model(self, descriptor: np.ndarray):
        """
        Initialize global NIG descriptor model.
        
        Uses models.nig (exact generative model).
        """
        if self.global_desc_model is None:
            self.global_desc_model = NIGModel.from_prior(
                mu=descriptor,
                kappa=NIG_PRIOR_KAPPA,
                alpha=NIG_PRIOR_ALPHA,
                beta=NIG_PRIOR_BETA
            )
    
    def update_global_model(self, descriptor: np.ndarray, weight: float):
        """
        Update global NIG model with new observation.
        
        Uses models.nig.update (exact Bayesian update).
        """
        if self.global_desc_model is not None:
            self.global_desc_model.update(descriptor, weight=weight)
    
    def get_global_model(self) -> Optional[NIGModel]:
        """Get current global NIG model."""
        return self.global_desc_model
    
    def copy_global_model(self) -> Optional[NIGModel]:
        """Get copy of global NIG model for new anchor."""
        if self.global_desc_model is None:
            return None
        return self.global_desc_model.copy()
