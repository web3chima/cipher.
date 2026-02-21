"""
Hypothesis strategies for generating test data for Cipher models.

These strategies generate valid instances of all data models for property-based testing.
"""

from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from cipher.models import (
    PointIPFS,
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
    HashAlgorithm,
    ZKPProtocol,
    RetentionPolicy,
)


# Basic strategies
@st.composite
def timestamps(draw):
    """Generate valid timestamps."""
    return draw(st.floats(min_value=1.0, max_value=2e9))


@st.composite
def confidences(draw):
    """Generate valid confidence scores."""
    return draw(st.floats(min_value=0.0, max_value=1.0))


@st.composite
def sensor_ids(draw):
    """Generate valid sensor IDs."""
    return draw(st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=('Cs',))))


# PointIPFS strategy
@st.composite
def point_IPFS(draw):
    """Generate valid PointIPFS instances."""
    n_points = draw(st.integers(min_value=1, max_value=1000))
    points = draw(arrays(
        dtype=np.float64,
        shape=(n_points, 3),
        elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    ))
    
    has_intensities = draw(st.booleans())
    intensities = None
    if has_intensities:
        intensities = draw(arrays(
            dtype=np.float64,
            shape=(n_points,),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ))
    
    timestamp = draw(timestamps())
    sensor_id = draw(sensor_ids())
    
    return PointIPFS(
        points=points,
        intensities=intensities,
        timestamp=timestamp,
        sensor_id=sensor_id
    )


# CameraFrame strategy
@st.composite
def camera_frames(draw):
    """Generate valid CameraFrame instances."""
    width = draw(st.integers(min_value=64, max_value=640))
    height = draw(st.integers(min_value=64, max_value=480))
    
    image = draw(arrays(
        dtype=np.uint8,
        shape=(height, width, 3),
        elements=st.integers(min_value=0, max_value=255)
    ))
    
    timestamp = draw(timestamps())
    camera_id = draw(sensor_ids())
    
    return CameraFrame(
        image=image,
        timestamp=timestamp,
        camera_id=camera_id,
        resolution=(width, height)
    )


# SpatialFeatures strategy
@st.composite
def spatial_features(draw):
    """Generate valid SpatialFeatures instances."""
    room_dimensions = (
        draw(st.floats(min_value=1.0, max_value=20.0)),
        draw(st.floats(min_value=1.0, max_value=20.0)),
        draw(st.floats(min_value=2.0, max_value=5.0))
    )
    
    n_walls = draw(st.integers(min_value=0, max_value=10))
    wall_positions = [
        draw(arrays(
            dtype=np.float64,
            shape=(4,),
            elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ))
        for _ in range(n_walls)
    ]
    
    n_objects = draw(st.integers(min_value=0, max_value=20))
    object_positions = [
        (
            draw(st.floats(min_value=-10.0, max_value=10.0)),
            draw(st.floats(min_value=-10.0, max_value=10.0)),
            draw(st.floats(min_value=0.0, max_value=3.0))
        )
        for _ in range(n_objects)
    ]
    
    signature_size = draw(st.integers(min_value=10, max_value=100))
    geometric_signature = draw(arrays(
        dtype=np.float64,
        shape=(signature_size,),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    confidence = draw(confidences())
    
    return SpatialFeatures(
        room_dimensions=room_dimensions,
        wall_positions=wall_positions,
        object_positions=object_positions,
        geometric_signature=geometric_signature,
        confidence=confidence
    )


# VisualFeatures strategy
@st.composite
def visual_features(draw):
    """Generate valid VisualFeatures instances."""
    n_keypoints = draw(st.integers(min_value=0, max_value=100))  # Reduced from 500
    keypoints = [
        (
            draw(st.floats(min_value=0.0, max_value=640.0)),
            draw(st.floats(min_value=0.0, max_value=480.0))
        )
        for _ in range(n_keypoints)
    ]
    
    descriptor_dim = draw(st.integers(min_value=32, max_value=128))  # Reduced from 256
    descriptors = draw(arrays(
        dtype=np.float64,
        shape=(n_keypoints, descriptor_dim),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    signature_size = draw(st.integers(min_value=10, max_value=50))  # Reduced from 100
    visual_signature = draw(arrays(
        dtype=np.float64,
        shape=(signature_size,),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    confidence = draw(confidences())
    
    return VisualFeatures(
        keypoints=keypoints,
        descriptors=descriptors,
        visual_signature=visual_signature,
        confidence=confidence
    )


# FusedFeatures strategy
@st.composite
def fused_features(draw):
    """Generate valid FusedFeatures instances."""
    spatial = draw(spatial_features())
    visual = draw(visual_features())
    
    signature_size = draw(st.integers(min_value=20, max_value=200))
    combined_signature = draw(arrays(
        dtype=np.float64,
        shape=(signature_size,),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    fusion_confidence = draw(confidences())
    
    return FusedFeatures(
        spatial=spatial,
        visual=visual,
        combined_signature=combined_signature,
        fusion_confidence=fusion_confidence
    )


# LocationClassification strategy
@st.composite
def location_classifications(draw):
    """Generate valid LocationClassification instances."""
    labels = ["living_room", "kitchen", "hallway", "bedroom", "bathroom", "office", "garage"]
    label = draw(st.sampled_from(labels))
    confidence = draw(confidences())
    
    n_alternatives = draw(st.integers(min_value=0, max_value=3))
    alternative_labels = [
        (draw(st.sampled_from([l for l in labels if l != label])), draw(confidences()))
        for _ in range(n_alternatives)
    ]
    
    return LocationClassification(
        label=label,
        confidence=confidence,
        alternative_labels=alternative_labels
    )


# LocationHash strategy
@st.composite
def location_hashes(draw):
    """Generate valid LocationHash instances."""
    hash_size = draw(st.integers(min_value=16, max_value=64))
    hash_value = draw(st.binary(min_size=hash_size, max_size=hash_size))
    algorithm = draw(st.sampled_from([a.value for a in HashAlgorithm]))
    timestamp = draw(timestamps())
    
    return LocationHash(
        hash_value=hash_value,
        algorithm=algorithm,
        timestamp=timestamp
    )


# Proof strategy
@st.composite
def proofs(draw):
    """Generate valid Proof instances."""
    proof_size = draw(st.integers(min_value=100, max_value=2048))
    proof_data = draw(st.binary(min_size=proof_size, max_size=proof_size))
    location_hash = draw(location_hashes())
    protocol = draw(st.sampled_from([p.value for p in ZKPProtocol]))
    timestamp = draw(timestamps())
    device_id = draw(sensor_ids())
    
    return Proof(
        proof_data=proof_data,
        location_hash=location_hash,
        protocol=protocol,
        timestamp=timestamp,
        device_id=device_id
    )


# VerificationResult strategy
@st.composite
def verification_results(draw):
    """Generate valid VerificationResult instances."""
    is_valid = draw(st.booleans())
    location_hash = draw(location_hashes())
    timestamp = draw(timestamps())
    verifier_id = draw(sensor_ids())
    error_message = None
    if not is_valid:
        error_message = draw(st.text(min_size=1, max_size=200))
    
    return VerificationResult(
        is_valid=is_valid,
        location_hash=location_hash,
        timestamp=timestamp,
        verifier_id=verifier_id,
        error_message=error_message
    )


# CipherConfig strategy
@st.composite
def cipher_configs(draw):
    """Generate valid CipherConfig instances."""
    hash_algorithm = draw(st.sampled_from([a.value for a in HashAlgorithm]))
    zkp_protocol = draw(st.sampled_from([p.value for p in ZKPProtocol]))
    retention_policy = draw(st.sampled_from([r.value for r in RetentionPolicy]))
    location_granularity = draw(st.sampled_from(["coarse", "medium", "fine"]))
    processing_timeout_ms = draw(st.integers(min_value=100, max_value=2000))
    proof_generation_timeout_ms = draw(st.integers(min_value=500, max_value=5000))
    
    return CipherConfig(
        hash_algorithm=hash_algorithm,
        zkp_protocol=zkp_protocol,
        retention_policy=retention_policy,
        location_granularity=location_granularity,
        processing_timeout_ms=processing_timeout_ms,
        proof_generation_timeout_ms=proof_generation_timeout_ms
    )


# LocationResult strategy
@st.composite
def location_results(draw):
    """Generate valid LocationResult instances."""
    classification = draw(location_classifications())
    location_hash = draw(location_hashes())
    metadata = draw(st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(st.text(), st.integers(), st.floats(allow_nan=False, allow_infinity=False))
    ))
    
    return LocationResult(
        classification=classification,
        location_hash=location_hash,
        metadata=metadata
    )


# ProcessingResult strategy
@st.composite
def processing_results(draw):
    """Generate valid ProcessingResult instances."""
    success = draw(st.booleans())
    classification = None
    proof = None
    error_message = None
    
    if success:
        classification = draw(location_classifications())
        has_proof = draw(st.booleans())
        if has_proof:
            proof = draw(proofs())
    else:
        error_message = draw(st.text(min_size=1, max_size=200))
    
    return ProcessingResult(
        success=success,
        classification=classification,
        proof=proof,
        error_message=error_message
    )


# TransmissionResult strategy
@st.composite
def transmission_results(draw):
    """Generate valid TransmissionResult instances."""
    success = draw(st.booleans())
    timestamp = draw(timestamps())
    server_response = None
    error_message = None
    
    if success:
        server_response = draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.text(), st.integers(), st.booleans())
        ))
    else:
        error_message = draw(st.text(min_size=1, max_size=200))
    
    return TransmissionResult(
        success=success,
        timestamp=timestamp,
        server_response=server_response,
        error_message=error_message
    )
