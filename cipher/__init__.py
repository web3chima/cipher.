"""
Cipher: Privacy-Preserving Robotics Navigation System

A system that applies Zero Knowledge Proofs to robot sensor data,
enabling verifiable location assertions without transmitting sensitive
spatial data.
"""

__version__ = "0.1.0"

from cipher.models import (
    PointCloud,
    CameraFrame,
    SpatialFeatures,
    VisualFeatures,
    FusedFeatures,
    LocationClassification,
    LocationHash,
    Proof,
    VerificationResult,
    CipherConfig,
    LocationResult,
    ProcessingResult,
    TransmissionResult,
)
from cipher.lidar_processor import LiDARProcessor, CorruptedDataError
from cipher.vslam_processor import VSLAMProcessor
from cipher.feature_fusion import FeatureFusion

__all__ = [
    "PointCloud",
    "CameraFrame",
    "SpatialFeatures",
    "VisualFeatures",
    "FusedFeatures",
    "LocationClassification",
    "LocationHash",
    "Proof",
    "VerificationResult",
    "CipherConfig",
    "LocationResult",
    "ProcessingResult",
    "TransmissionResult",
    "LiDARProcessor",
    "VSLAMProcessor",
    "FeatureFusion",
    "CorruptedDataError",
]
