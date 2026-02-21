"""
Core data models for the Cipher system.

All models support validation and serialization for property-based testing.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Any, Dict
import numpy as np
import json
from enum import Enum


class HashAlgorithm(Enum):
    """Supported cryptographic hash algorithms."""
    SHA256 = "SHA-256"
    SHA3 = "SHA-3"
    BLAKE3 = "BLAKE3"


class ZKPProtocol(Enum):
    """Supported ZKP protocols."""
    GROTH16 = "Groth16"
    PLONK = "PLONK"


class RetentionPolicy(Enum):
    """Data retention policies."""
    IMMEDIATE_PURGE = "immediate_purge"
    SESSION_BASED = "session_based"
    TIME_BASED = "time_based"


@dataclass
class PointIPFS:
    """LiDAR point IPFS data."""
    points: np.ndarray  # Nx3 array of (x, y, z) coordinates
    intensities: Optional[np.ndarray]  # N-length array of intensity values
    timestamp: float  # Unix timestamp
    sensor_id: str  # Identifier for LiDAR sensor
    
    def __post_init__(self):
        """Validate point IPFS data."""
        if not isinstance(self.points, np.ndarray):
            raise ValueError("points must be a numpy array")
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must be Nx3 array")
        if self.intensities is not None:
            if not isinstance(self.intensities, np.ndarray):
                raise ValueError("intensities must be a numpy array")
            if len(self.intensities) != len(self.points):
                raise ValueError("intensities length must match points length")
        if self.timestamp <= 0:
            raise ValueError("timestamp must be positive")
        if not self.sensor_id:
            raise ValueError("sensor_id cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "points": self.points.tolist(),
            "intensities": self.intensities.tolist() if self.intensities is not None else None,
            "timestamp": self.timestamp,
            "sensor_id": self.sensor_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointCloud":
        """Deserialize from dictionary."""
        return cls(
            points=np.array(data["points"]),
            intensities=np.array(data["intensities"]) if data["intensities"] is not None else None,
            timestamp=data["timestamp"],
            sensor_id=data["sensor_id"],
        )


@dataclass
class CameraFrame:
    """Camera frame data."""
    image: np.ndarray  # HxWx3 RGB image array
    timestamp: float  # Unix timestamp
    camera_id: str  # Identifier for camera
    resolution: Tuple[int, int]  # (width, height)
    
    def __post_init__(self):
        """Validate camera frame data."""
        if not isinstance(self.image, np.ndarray):
            raise ValueError("image must be a numpy array")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("image must be HxWx3 array")
        if self.timestamp <= 0:
            raise ValueError("timestamp must be positive")
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if len(self.resolution) != 2 or self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise ValueError("resolution must be (width, height) with positive values")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "image": self.image.tolist(),
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "resolution": list(self.resolution),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraFrame":
        """Deserialize from dictionary."""
        return cls(
            image=np.array(data["image"]),
            timestamp=data["timestamp"],
            camera_id=data["camera_id"],
            resolution=tuple(data["resolution"]),
        )


@dataclass
class SpatialFeatures:
    """Spatial features extracted from LiDAR data."""
    room_dimensions: Tuple[float, float, float]  # (length, width, height)
    wall_positions: List[np.ndarray]  # List of wall plane equations
    object_positions: List[Tuple[float, float, float]]  # (x, y, z) positions
    geometric_signature: np.ndarray  # Normalized feature vector
    confidence: float  # 0.0 to 1.0
    
    def __post_init__(self):
        """Validate spatial features."""
        if len(self.room_dimensions) != 3:
            raise ValueError("room_dimensions must be (length, width, height)")
        if any(d <= 0 for d in self.room_dimensions):
            raise ValueError("room_dimensions must be positive")
        if not isinstance(self.geometric_signature, np.ndarray):
            raise ValueError("geometric_signature must be a numpy array")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "room_dimensions": list(self.room_dimensions),
            "wall_positions": [w.tolist() for w in self.wall_positions],
            "object_positions": [list(pos) for pos in self.object_positions],
            "geometric_signature": self.geometric_signature.tolist(),
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialFeatures":
        """Deserialize from dictionary."""
        return cls(
            room_dimensions=tuple(data["room_dimensions"]),
            wall_positions=[np.array(w) for w in data["wall_positions"]],
            object_positions=[tuple(pos) for pos in data["object_positions"]],
            geometric_signature=np.array(data["geometric_signature"]),
            confidence=data["confidence"],
        )


@dataclass
class VisualFeatures:
    """Visual features extracted from camera data."""
    keypoints: List[Tuple[float, float]]  # (x, y) pixel coordinates
    descriptors: np.ndarray  # NxD descriptor matrix
    visual_signature: np.ndarray  # Normalized feature vector
    confidence: float  # 0.0 to 1.0
    
    def __post_init__(self):
        """Validate visual features."""
        if not isinstance(self.descriptors, np.ndarray):
            raise ValueError("descriptors must be a numpy array")
        if not isinstance(self.visual_signature, np.ndarray):
            raise ValueError("visual_signature must be a numpy array")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "keypoints": [list(kp) for kp in self.keypoints],
            "descriptors": self.descriptors.tolist(),
            "visual_signature": self.visual_signature.tolist(),
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualFeatures":
        """Deserialize from dictionary."""
        return cls(
            keypoints=[tuple(kp) for kp in data["keypoints"]],
            descriptors=np.array(data["descriptors"]),
            visual_signature=np.array(data["visual_signature"]),
            confidence=data["confidence"],
        )


@dataclass
class FusedFeatures:
    """Combined spatial and visual features."""
    spatial: SpatialFeatures
    visual: VisualFeatures
    combined_signature: np.ndarray  # Unified feature vector for hashing
    fusion_confidence: float  # 0.0 to 1.0
    
    def __post_init__(self):
        """Validate fused features."""
        if not isinstance(self.combined_signature, np.ndarray):
            raise ValueError("combined_signature must be a numpy array")
        if not 0.0 <= self.fusion_confidence <= 1.0:
            raise ValueError("fusion_confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "spatial": self.spatial.to_dict(),
            "visual": self.visual.to_dict(),
            "combined_signature": self.combined_signature.tolist(),
            "fusion_confidence": self.fusion_confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusedFeatures":
        """Deserialize from dictionary."""
        return cls(
            spatial=SpatialFeatures.from_dict(data["spatial"]),
            visual=VisualFeatures.from_dict(data["visual"]),
            combined_signature=np.array(data["combined_signature"]),
            fusion_confidence=data["fusion_confidence"],
        )


@dataclass
class LocationClassification:
    """Location classification result."""
    label: str  # e.g., "living_room", "kitchen", "hallway"
    confidence: float  # 0.0 to 1.0
    alternative_labels: List[Tuple[str, float]]  # Alternative classifications with scores
    
    def __post_init__(self):
        """Validate location classification."""
        if not self.label:
            raise ValueError("label cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "alternative_labels": [[label, score] for label, score in self.alternative_labels],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocationClassification":
        """Deserialize from dictionary."""
        return cls(
            label=data["label"],
            confidence=data["confidence"],
            alternative_labels=[tuple(alt) for alt in data["alternative_labels"]],
        )


@dataclass
class LocationHash:
    """Cryptographic hash representing a location."""
    hash_value: bytes  # 32 bytes for SHA-256
    algorithm: str  # "SHA-256", "SHA-3", "BLAKE3"
    timestamp: float  # When hash was generated
    
    def __post_init__(self):
        """Validate location hash."""
        if not isinstance(self.hash_value, bytes):
            raise ValueError("hash_value must be bytes")
        if len(self.hash_value) == 0:
            raise ValueError("hash_value cannot be empty")
        if self.algorithm not in [a.value for a in HashAlgorithm]:
            raise ValueError(f"algorithm must be one of {[a.value for a in HashAlgorithm]}")
        if self.timestamp <= 0:
            raise ValueError("timestamp must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hash_value": self.hash_value.hex(),
            "algorithm": self.algorithm,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocationHash":
        """Deserialize from dictionary."""
        return cls(
            hash_value=bytes.fromhex(data["hash_value"]),
            algorithm=data["algorithm"],
            timestamp=data["timestamp"],
        )


@dataclass
class Proof:
    """Zero-knowledge proof of location."""
    proof_data: bytes  # Serialized zk-SNARK proof (≤2kb)
    location_hash: LocationHash  # Associated location hash
    protocol: str  # "Groth16" or "PLONK"
    timestamp: float  # When proof was generated
    device_id: str  # Robot identifier
    
    def __post_init__(self):
        """Validate proof."""
        if not isinstance(self.proof_data, bytes):
            raise ValueError("proof_data must be bytes")
        if len(self.proof_data) == 0:
            raise ValueError("proof_data cannot be empty")
        if len(self.proof_data) > 2048:
            raise ValueError("proof_data must not exceed 2kb")
        if self.protocol not in [p.value for p in ZKPProtocol]:
            raise ValueError(f"protocol must be one of {[p.value for p in ZKPProtocol]}")
        if self.timestamp <= 0:
            raise ValueError("timestamp must be positive")
        if not self.device_id:
            raise ValueError("device_id cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "proof_data": self.proof_data.hex(),
            "location_hash": self.location_hash.to_dict(),
            "protocol": self.protocol,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Proof":
        """Deserialize from dictionary."""
        return cls(
            proof_data=bytes.fromhex(data["proof_data"]),
            location_hash=LocationHash.from_dict(data["location_hash"]),
            protocol=data["protocol"],
            timestamp=data["timestamp"],
            device_id=data["device_id"],
        )


@dataclass
class VerificationResult:
    """Result of proof verification."""
    is_valid: bool  # Proof validity
    location_hash: LocationHash  # Verified location hash
    timestamp: float  # Verification timestamp
    verifier_id: str  # Cloud verifier identifier
    error_message: Optional[str] = None  # Error details if invalid
    
    def __post_init__(self):
        """Validate verification result."""
        if self.timestamp <= 0:
            raise ValueError("timestamp must be positive")
        if not self.verifier_id:
            raise ValueError("verifier_id cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_valid": self.is_valid,
            "location_hash": self.location_hash.to_dict(),
            "timestamp": self.timestamp,
            "verifier_id": self.verifier_id,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        """Deserialize from dictionary."""
        return cls(
            is_valid=data["is_valid"],
            location_hash=LocationHash.from_dict(data["location_hash"]),
            timestamp=data["timestamp"],
            verifier_id=data["verifier_id"],
            error_message=data.get("error_message"),
        )


@dataclass
class CipherConfig:
    """Configuration for Cipher system."""
    hash_algorithm: str = "SHA-256"  # Hash algorithm choice
    zkp_protocol: str = "Groth16"  # ZKP protocol choice
    retention_policy: str = "session_based"  # Data retention policy
    location_granularity: str = "medium"  # Classification granularity
    processing_timeout_ms: int = 500  # Max processing time per sensor
    proof_generation_timeout_ms: int = 2000  # Max proof generation time
    
    def __post_init__(self):
        """Validate configuration."""
        if self.hash_algorithm not in [a.value for a in HashAlgorithm]:
            raise ValueError(f"hash_algorithm must be one of {[a.value for a in HashAlgorithm]}")
        if self.zkp_protocol not in [p.value for p in ZKPProtocol]:
            raise ValueError(f"zkp_protocol must be one of {[p.value for p in ZKPProtocol]}")
        if self.retention_policy not in [r.value for r in RetentionPolicy]:
            raise ValueError(f"retention_policy must be one of {[r.value for r in RetentionPolicy]}")
        if self.location_granularity not in ["coarse", "medium", "fine"]:
            raise ValueError("location_granularity must be one of ['coarse', 'medium', 'fine']")
        if self.processing_timeout_ms <= 0:
            raise ValueError("processing_timeout_ms must be positive")
        if self.proof_generation_timeout_ms <= 0:
            raise ValueError("proof_generation_timeout_ms must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CipherConfig":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class LocationResult:
    """Result of location determination."""
    classification: LocationClassification
    location_hash: LocationHash
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "classification": self.classification.to_dict(),
            "location_hash": self.location_hash.to_dict(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocationResult":
        """Deserialize from dictionary."""
        return cls(
            classification=LocationClassification.from_dict(data["classification"]),
            location_hash=LocationHash.from_dict(data["location_hash"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProcessingResult:
    """Result of sensor data processing."""
    success: bool
    classification: Optional[LocationClassification] = None
    proof: Optional[Proof] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "classification": self.classification.to_dict() if self.classification else None,
            "proof": self.proof.to_dict() if self.proof else None,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingResult":
        """Deserialize from dictionary."""
        return cls(
            success=data["success"],
            classification=LocationClassification.from_dict(data["classification"]) if data["classification"] else None,
            proof=Proof.from_dict(data["proof"]) if data["proof"] else None,
            error_message=data.get("error_message"),
        )


@dataclass
class TransmissionResult:
    """Result of proof transmission."""
    success: bool
    timestamp: float
    server_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate transmission result."""
        if self.timestamp <= 0:
            raise ValueError("timestamp must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "timestamp": self.timestamp,
            "server_response": self.server_response,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransmissionResult":
        """Deserialize from dictionary."""
        return cls(
            success=data["success"],
            timestamp=data["timestamp"],
            server_response=data.get("server_response"),
            error_message=data.get("error_message"),
        )
