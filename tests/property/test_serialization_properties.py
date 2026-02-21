"""
Property-based tests for data model serialization.

Feature: cipher-zkp-robotics
Property: Serialization round trip - For any valid data model instance, 
serializing then deserializing should produce an equivalent object.
"""

import pytest
from hypothesis import given, settings, HealthCheck
import numpy as np

from tests.strategies import (
    point_IPFS,
    camera_frames,
    spatial_features,
    visual_features,
    fused_features,
    location_classifications,
    location_hashes,
    proofs,
    verification_results,
    cipher_configs,
    location_results,
    processing_results,
    transmission_results,
)


def arrays_equal(arr1, arr2):
    """Check if two numpy arrays are equal, handling None values and empty arrays."""
    if arr1 is None and arr2 is None:
        return True
    if arr1 is None or arr2 is None:
        return False
    # Handle empty arrays - if both are empty, consider them equal regardless of shape
    if arr1.size == 0 and arr2.size == 0:
        return True
    return np.array_equal(arr1, arr2)


def lists_of_arrays_equal(list1, list2):
    """Check if two lists of numpy arrays are equal."""
    if len(list1) != len(list2):
        return False
    return all(arrays_equal(a1, a2) for a1, a2 in zip(list1, list2))


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(point_IPFS())
@settings(max_examples=100)
def test_point_IPFS_serialization_round_trip(point_IPFS):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid PointIPFS instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = point_IPFS.to_dict()
    
    # Deserialize
    deserialized = point_IPFS.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert arrays_equal(deserialized.points, point_IPFS.points)
    assert arrays_equal(deserialized.intensities, point_IPFS.intensities)
    assert deserialized.timestamp == point_IPFS.timestamp
    assert deserialized.sensor_id == point_IPFS.sensor_id


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(camera_frames())
@settings(max_examples=100)
def test_camera_frame_serialization_round_trip(camera_frame):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid CameraFrame instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = camera_frame.to_dict()
    
    # Deserialize
    deserialized = camera_frame.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert arrays_equal(deserialized.image, camera_frame.image)
    assert deserialized.timestamp == camera_frame.timestamp
    assert deserialized.camera_id == camera_frame.camera_id
    assert deserialized.resolution == camera_frame.resolution


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(spatial_features())
@settings(max_examples=100)
def test_spatial_features_serialization_round_trip(spatial_features_instance):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid SpatialFeatures instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = spatial_features_instance.to_dict()
    
    # Deserialize
    deserialized = spatial_features_instance.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert deserialized.room_dimensions == spatial_features_instance.room_dimensions
    assert lists_of_arrays_equal(deserialized.wall_positions, spatial_features_instance.wall_positions)
    assert deserialized.object_positions == spatial_features_instance.object_positions
    assert arrays_equal(deserialized.geometric_signature, spatial_features_instance.geometric_signature)
    assert deserialized.confidence == spatial_features_instance.confidence


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(visual_features())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_visual_features_serialization_round_trip(visual_features_instance):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid VisualFeatures instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = visual_features_instance.to_dict()
    
    # Deserialize
    deserialized = visual_features_instance.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert deserialized.keypoints == visual_features_instance.keypoints
    assert arrays_equal(deserialized.descriptors, visual_features_instance.descriptors)
    assert arrays_equal(deserialized.visual_signature, visual_features_instance.visual_signature)
    assert deserialized.confidence == visual_features_instance.confidence


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(fused_features())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_fused_features_serialization_round_trip(fused_features_instance):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid FusedFeatures instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = fused_features_instance.to_dict()
    
    # Deserialize
    deserialized = fused_features_instance.__class__.from_dict(serialized)
    
    # Verify equivalence - check spatial features
    assert deserialized.spatial.room_dimensions == fused_features_instance.spatial.room_dimensions
    assert lists_of_arrays_equal(deserialized.spatial.wall_positions, fused_features_instance.spatial.wall_positions)
    assert deserialized.spatial.object_positions == fused_features_instance.spatial.object_positions
    assert arrays_equal(deserialized.spatial.geometric_signature, fused_features_instance.spatial.geometric_signature)
    assert deserialized.spatial.confidence == fused_features_instance.spatial.confidence
    
    # Verify equivalence - check visual features
    assert deserialized.visual.keypoints == fused_features_instance.visual.keypoints
    assert arrays_equal(deserialized.visual.descriptors, fused_features_instance.visual.descriptors)
    assert arrays_equal(deserialized.visual.visual_signature, fused_features_instance.visual.visual_signature)
    assert deserialized.visual.confidence == fused_features_instance.visual.confidence
    
    # Verify equivalence - check combined features
    assert arrays_equal(deserialized.combined_signature, fused_features_instance.combined_signature)
    assert deserialized.fusion_confidence == fused_features_instance.fusion_confidence


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(location_classifications())
@settings(max_examples=100)
def test_location_classification_serialization_round_trip(location_classification):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid LocationClassification instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = location_classification.to_dict()
    
    # Deserialize
    deserialized = location_classification.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert deserialized.label == location_classification.label
    assert deserialized.confidence == location_classification.confidence
    assert deserialized.alternative_labels == location_classification.alternative_labels


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(location_hashes())
@settings(max_examples=100)
def test_location_hash_serialization_round_trip(location_hash):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid LocationHash instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = location_hash.to_dict()
    
    # Deserialize
    deserialized = location_hash.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert deserialized.hash_value == location_hash.hash_value
    assert deserialized.algorithm == location_hash.algorithm
    assert deserialized.timestamp == location_hash.timestamp


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(proofs())
@settings(max_examples=100)
def test_proof_serialization_round_trip(proof):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid Proof instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = proof.to_dict()
    
    # Deserialize
    deserialized = proof.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert deserialized.proof_data == proof.proof_data
    assert deserialized.location_hash.hash_value == proof.location_hash.hash_value
    assert deserialized.location_hash.algorithm == proof.location_hash.algorithm
    assert deserialized.location_hash.timestamp == proof.location_hash.timestamp
    assert deserialized.protocol == proof.protocol
    assert deserialized.timestamp == proof.timestamp
    assert deserialized.device_id == proof.device_id


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(verification_results())
@settings(max_examples=100)
def test_verification_result_serialization_round_trip(verification_result):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid VerificationResult instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = verification_result.to_dict()
    
    # Deserialize
    deserialized = verification_result.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert deserialized.is_valid == verification_result.is_valid
    assert deserialized.location_hash.hash_value == verification_result.location_hash.hash_value
    assert deserialized.location_hash.algorithm == verification_result.location_hash.algorithm
    assert deserialized.location_hash.timestamp == verification_result.location_hash.timestamp
    assert deserialized.timestamp == verification_result.timestamp
    assert deserialized.verifier_id == verification_result.verifier_id
    assert deserialized.error_message == verification_result.error_message


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(cipher_configs())
@settings(max_examples=100)
def test_cipher_config_serialization_round_trip(cipher_config):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid CipherConfig instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = cipher_config.to_dict()
    
    # Deserialize
    deserialized = cipher_config.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert deserialized.hash_algorithm == cipher_config.hash_algorithm
    assert deserialized.zkp_protocol == cipher_config.zkp_protocol
    assert deserialized.retention_policy == cipher_config.retention_policy
    assert deserialized.location_granularity == cipher_config.location_granularity
    assert deserialized.processing_timeout_ms == cipher_config.processing_timeout_ms
    assert deserialized.proof_generation_timeout_ms == cipher_config.proof_generation_timeout_ms


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(location_results())
@settings(max_examples=100)
def test_location_result_serialization_round_trip(location_result):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid LocationResult instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = location_result.to_dict()
    
    # Deserialize
    deserialized = location_result.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert deserialized.classification.label == location_result.classification.label
    assert deserialized.classification.confidence == location_result.classification.confidence
    assert deserialized.classification.alternative_labels == location_result.classification.alternative_labels
    assert deserialized.location_hash.hash_value == location_result.location_hash.hash_value
    assert deserialized.location_hash.algorithm == location_result.location_hash.algorithm
    assert deserialized.location_hash.timestamp == location_result.location_hash.timestamp
    assert deserialized.metadata == location_result.metadata


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(processing_results())
@settings(max_examples=100)
def test_processing_result_serialization_round_trip(processing_result):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid ProcessingResult instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = processing_result.to_dict()
    
    # Deserialize
    deserialized = processing_result.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert deserialized.success == processing_result.success
    assert deserialized.error_message == processing_result.error_message
    
    if processing_result.classification is not None:
        assert deserialized.classification.label == processing_result.classification.label
        assert deserialized.classification.confidence == processing_result.classification.confidence
    else:
        assert deserialized.classification is None
    
    if processing_result.proof is not None:
        assert deserialized.proof.proof_data == processing_result.proof.proof_data
        assert deserialized.proof.protocol == processing_result.proof.protocol
        assert deserialized.proof.device_id == processing_result.proof.device_id
    else:
        assert deserialized.proof is None


# Feature: cipher-zkp-robotics, Property: Serialization round trip
@given(transmission_results())
@settings(max_examples=100)
def test_transmission_result_serialization_round_trip(transmission_result):
    """
    **Validates: Requirements All (foundational)**
    
    For any valid TransmissionResult instance, serializing then deserializing 
    should produce an equivalent object.
    """
    # Serialize
    serialized = transmission_result.to_dict()
    
    # Deserialize
    deserialized = transmission_result.__class__.from_dict(serialized)
    
    # Verify equivalence
    assert deserialized.success == transmission_result.success
    assert deserialized.timestamp == transmission_result.timestamp
    assert deserialized.server_response == transmission_result.server_response
    assert deserialized.error_message == transmission_result.error_message
