"""
Feature fusion module for combining spatial and visual features.

This module combines LiDAR spatial features and VSLAM visual features to
produce robust location recognition with single-sensor fallback capability.
"""

import numpy as np
import time
from typing import Optional
from cipher.models import (
    SpatialFeatures,
    VisualFeatures,
    FusedFeatures,
    LocationClassification,
    LocationResult,
    LocationHash,
)


class FeatureFusion:
    """
    Combine spatial and visual features for robust location recognition.
    
    This class fuses LiDAR spatial features and VSLAM visual features using
    weighted fusion based on sensor confidence scores. It supports single-sensor
    fallback when one sensor fails, ensuring continued operation.
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize the feature fusion module.
        
        Args:
            confidence_threshold: Minimum confidence threshold for location
                                 classification (default: 0.3)
        """
        self.confidence_threshold = confidence_threshold
        self._location_labels = [
            "living_room", "kitchen", "hallway", "bedroom", 
            "bathroom", "office", "garage", "dining_room",
            "entryway", "closet", "laundry_room", "basement",
            "attic", "balcony", "patio", "workshop",
            "storage_room", "utility_room", "pantry", "library"
        ]
    
    def fuse_features(
        self, 
        spatial: Optional[SpatialFeatures], 
        visual: Optional[VisualFeatures]
    ) -> FusedFeatures:
        """
        Combine spatial and visual features.
        
        This method performs weighted fusion based on sensor confidence scores.
        If one sensor fails (None), it falls back to the available sensor.
        
        Args:
            spatial: Features from LiDAR processor (can be None)
            visual: Features from VSLAM processor (can be None)
            
        Returns:
            FusedFeatures: Combined feature representation
            
        Raises:
            ValueError: If both spatial and visual features are None
        """
        # Handle single-sensor fallback
        if spatial is None and visual is None:
            raise ValueError("At least one sensor must provide features")
        
        if spatial is None:
            # Visual-only fallback
            return self._create_visual_only_fusion(visual)
        
        if visual is None:
            # Spatial-only fallback
            return self._create_spatial_only_fusion(spatial)
        
        # Both sensors available - perform weighted fusion
        spatial_weight = spatial.confidence
        visual_weight = visual.confidence
        total_weight = spatial_weight + visual_weight
        
        # Normalize weights
        if total_weight > 0:
            spatial_weight /= total_weight
            visual_weight /= total_weight
        else:
            # Equal weights if both confidences are zero
            spatial_weight = 0.5
            visual_weight = 0.5
        
        # Combine signatures using weighted average
        spatial_sig = spatial.geometric_signature
        visual_sig = visual.visual_signature
        
        # Ensure signatures are same length by padding/truncating
        target_length = max(len(spatial_sig), len(visual_sig))
        
        # Pad spatial signature if needed
        if len(spatial_sig) < target_length:
            spatial_sig = np.pad(
                spatial_sig, 
                (0, target_length - len(spatial_sig)), 
                mode='constant'
            )
        else:
            spatial_sig = spatial_sig[:target_length]
        
        # Pad visual signature if needed
        if len(visual_sig) < target_length:
            visual_sig = np.pad(
                visual_sig,
                (0, target_length - len(visual_sig)),
                mode='constant'
            )
        else:
            visual_sig = visual_sig[:target_length]
        
        # Weighted combination
        combined_signature = (
            spatial_weight * spatial_sig + 
            visual_weight * visual_sig
        )
        
        # Normalize combined signature
        norm = np.linalg.norm(combined_signature)
        if norm > 0:
            combined_signature = combined_signature / norm
        
        # Calculate fusion confidence (weighted average of sensor confidences)
        fusion_confidence = (
            spatial_weight * spatial.confidence + 
            visual_weight * visual.confidence
        )
        
        return FusedFeatures(
            spatial=spatial,
            visual=visual,
            combined_signature=combined_signature,
            fusion_confidence=fusion_confidence
        )
    
    def determine_location(self, fused: FusedFeatures) -> LocationResult:
        """
        Determine final location classification with confidence.
        
        This method analyzes the fused features to produce a semantic location
        label with confidence score. It combines information from both spatial
        and visual features when available.
        
        Args:
            fused: Combined features
            
        Returns:
            LocationResult: Location label, confidence, and metadata
        """
        # Extract features for classification
        spatial = fused.spatial
        visual = fused.visual
        
        # Determine which sensors are actually available (not dummy features)
        spatial_available = spatial is not None and spatial.confidence > 0.0
        visual_available = visual is not None and visual.confidence > 0.0
        
        # Collect classification scores from both sensors
        scores = {}
        
        # Spatial-based classification
        if spatial_available:
            spatial_scores = self._classify_from_spatial(spatial)
            for label, score in spatial_scores.items():
                scores[label] = scores.get(label, 0.0) + score * spatial.confidence
        
        # Visual-based classification
        if visual_available:
            visual_scores = self._classify_from_visual(visual)
            for label, score in visual_scores.items():
                scores[label] = scores.get(label, 0.0) + score * visual.confidence
        
        # Normalize scores by total confidence
        total_confidence = 0.0
        if spatial_available:
            total_confidence += spatial.confidence
        if visual_available:
            total_confidence += visual.confidence
        
        if total_confidence > 0:
            scores = {label: score / total_confidence for label, score in scores.items()}
        
        # Default classification if no scores
        if not scores:
            scores = {"unknown": 0.3}
        
        # Sort by score and select top classification
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        label = sorted_scores[0][0]
        confidence = sorted_scores[0][1]
        
        # Alternative labels (top 3)
        alternative_labels = [(l, s) for l, s in sorted_scores[1:4]]
        
        # Create location classification
        classification = LocationClassification(
            label=label,
            confidence=confidence,
            alternative_labels=alternative_labels
        )
        
        # Generate location hash
        location_hash = self._generate_location_hash(fused)
        
        # Create metadata
        metadata = {
            "fusion_confidence": fused.fusion_confidence,
            "low_confidence": confidence < self.confidence_threshold,
            "spatial_available": spatial_available,
            "visual_available": visual_available,
        }
        
        # Add uncertainty indication if confidence is low
        if confidence < self.confidence_threshold:
            metadata["uncertainty_warning"] = (
                f"Location classification confidence ({confidence:.2f}) "
                f"is below threshold ({self.confidence_threshold:.2f})"
            )
        
        return LocationResult(
            classification=classification,
            location_hash=location_hash,
            metadata=metadata
        )
    
    def _create_spatial_only_fusion(self, spatial: SpatialFeatures) -> FusedFeatures:
        """Create fused features from spatial features only."""
        # Use spatial signature as combined signature
        combined_signature = spatial.geometric_signature.copy()
        
        # Normalize
        norm = np.linalg.norm(combined_signature)
        if norm > 0:
            combined_signature = combined_signature / norm
        
        # Create dummy visual features for compatibility
        dummy_visual = VisualFeatures(
            keypoints=[],
            descriptors=np.array([]).reshape(0, 32),
            visual_signature=np.zeros_like(spatial.geometric_signature),
            confidence=0.0
        )
        
        return FusedFeatures(
            spatial=spatial,
            visual=dummy_visual,
            combined_signature=combined_signature,
            fusion_confidence=spatial.confidence
        )
    
    def _create_visual_only_fusion(self, visual: VisualFeatures) -> FusedFeatures:
        """Create fused features from visual features only."""
        # Use visual signature as combined signature
        combined_signature = visual.visual_signature.copy()
        
        # Normalize
        norm = np.linalg.norm(combined_signature)
        if norm > 0:
            combined_signature = combined_signature / norm
        
        # Create dummy spatial features for compatibility
        dummy_spatial = SpatialFeatures(
            room_dimensions=(1.0, 1.0, 1.0),
            wall_positions=[],
            object_positions=[],
            geometric_signature=np.zeros_like(visual.visual_signature),
            confidence=0.0
        )
        
        return FusedFeatures(
            spatial=dummy_spatial,
            visual=visual,
            combined_signature=combined_signature,
            fusion_confidence=visual.confidence
        )
    
    def _classify_from_spatial(self, spatial: SpatialFeatures) -> dict:
        """
        Classify location based on spatial features.
        
        Returns:
            Dictionary mapping location labels to scores
        """
        scores = {}
        
        # Extract spatial characteristics
        length, width, height = spatial.room_dimensions
        area = length * width
        volume = area * height
        aspect_ratio = max(length, width) / min(length, width) if min(length, width) > 0 else 1.0
        n_objects = len(spatial.object_positions)
        
        # Living room: large area, moderate height, many objects
        if area > 20.0 and height > 2.3 and n_objects > 5:
            scores["living_room"] = 0.7
        
        # Kitchen: medium area, many objects (appliances)
        if 10.0 < area < 25.0 and n_objects > 3:
            scores["kitchen"] = 0.65
        
        # Hallway: long and narrow (high aspect ratio), small area
        if aspect_ratio > 3.0 and area < 15.0:
            scores["hallway"] = 0.75
        
        # Bedroom: medium area, moderate height
        if 12.0 < area < 30.0 and 2.2 < height < 2.8:
            scores["bedroom"] = 0.6
        
        # Bathroom: small area, few objects
        if area < 10.0 and n_objects < 3:
            scores["bathroom"] = 0.65
        
        # Office: medium area, many objects (furniture, equipment)
        if 10.0 < area < 25.0 and n_objects > 4:
            scores["office"] = 0.6
        
        # Garage: large area, high ceiling
        if area > 25.0 and height > 2.5:
            scores["garage"] = 0.55
        
        # Closet: very small area
        if area < 5.0:
            scores["closet"] = 0.7
        
        # Default based on area if no strong match
        if not scores:
            if area < 8.0:
                scores["bathroom"] = 0.4
            elif area < 20.0:
                scores["bedroom"] = 0.4
            else:
                scores["living_room"] = 0.4
        
        return scores
    
    def _classify_from_visual(self, visual: VisualFeatures) -> dict:
        """
        Classify location based on visual features.
        
        Returns:
            Dictionary mapping location labels to scores
        """
        scores = {}
        
        # Extract visual characteristics
        n_keypoints = len(visual.keypoints)
        descriptor_variance = np.var(visual.descriptors) if visual.descriptors.size > 0 else 0.0
        
        # Living room: many features, high variance (diverse objects)
        if n_keypoints > 50 and descriptor_variance > 0.1:
            scores["living_room"] = 0.7
        
        # Kitchen: many features, moderate variance
        if 30 < n_keypoints < 100 and 0.05 < descriptor_variance < 0.15:
            scores["kitchen"] = 0.65
        
        # Hallway: few features, low variance (plain walls)
        if n_keypoints < 30 and descriptor_variance < 0.08:
            scores["hallway"] = 0.75
        
        # Bedroom: moderate features
        if 20 < n_keypoints < 80:
            scores["bedroom"] = 0.6
        
        # Bathroom: few features
        if n_keypoints < 40:
            scores["bathroom"] = 0.65
        
        # Office: many features, high variance
        if n_keypoints > 40 and descriptor_variance > 0.08:
            scores["office"] = 0.6
        
        # Default based on feature count if no strong match
        if not scores:
            if n_keypoints < 20:
                scores["closet"] = 0.4
            elif n_keypoints < 50:
                scores["bedroom"] = 0.4
            else:
                scores["living_room"] = 0.4
        
        return scores
    
    def _generate_location_hash(self, fused: FusedFeatures) -> LocationHash:
        """
        Generate a location hash from fused features.
        
        This is a placeholder implementation. The actual hash generation
        will be implemented by the HashGenerator class in task 6.1.
        
        Args:
            fused: Fused features
            
        Returns:
            LocationHash: Cryptographic hash representing the location
        """
        import hashlib
        
        # Serialize combined signature for hashing
        signature_bytes = fused.combined_signature.tobytes()
        
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(signature_bytes)
        hash_value = hash_obj.digest()
        
        return LocationHash(
            hash_value=hash_value,
            algorithm="SHA-256",
            timestamp=time.time()
        )
