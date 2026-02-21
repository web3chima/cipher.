"""
Unit tests for the FeatureFusion class.

Tests cover fusion scenarios, single-sensor fallback, and location determination.
"""

import pytest
import numpy as np
from cipher.feature_fusion import FeatureFusion
from cipher.models import (
    SpatialFeatures,
    VisualFeatures,
    FusedFeatures,
    LocationClassification,
    LocationResult,
)


class TestFeatureFusion:
    """Test suite for FeatureFusion class."""
    
    def test_initialization(self):
        """Test FeatureFusion initialization."""
        fusion = FeatureFusion()
        assert fusion.confidence_threshold == 0.3
        
        fusion_custom = FeatureFusion(confidence_threshold=0.5)
        assert fusion_custom.confidence_threshold == 0.5
    
    def test_fuse_features_both_sensors(self):
        """Test feature fusion with both spatial and visual features."""
        fusion = FeatureFusion()
        
        # Create spatial features
        spatial = SpatialFeatures(
            room_dimensions=(5.0, 4.0, 2.5),
            wall_positions=[],
            object_positions=[(1.0, 1.0, 0.5), (2.0, 2.0, 0.5)],
            geometric_signature=np.random.rand(128),
            confidence=0.8
        )
        
        # Create visual features
        visual = VisualFeatures(
            keypoints=[(10.0, 20.0), (30.0, 40.0)],
            descriptors=np.random.rand(2, 32),
            visual_signature=np.random.rand(128),
            confidence=0.7
        )
        
        # Fuse features
        fused = fusion.fuse_features(spatial, visual)
        
        # Verify result
        assert isinstance(fused, FusedFeatures)
        assert fused.spatial == spatial
        assert fused.visual == visual
        assert len(fused.combined_signature) > 0
        assert 0.0 <= fused.fusion_confidence <= 1.0
        
        # Fusion confidence should be weighted average
        expected_confidence = (0.8 * 0.8 + 0.7 * 0.7) / (0.8 + 0.7)
        assert abs(fused.fusion_confidence - expected_confidence) < 0.01
    
    def test_fuse_features_spatial_only(self):
        """Test feature fusion with spatial features only (visual sensor failed)."""
        fusion = FeatureFusion()
        
        # Create spatial features
        spatial = SpatialFeatures(
            room_dimensions=(5.0, 4.0, 2.5),
            wall_positions=[],
            object_positions=[(1.0, 1.0, 0.5)],
            geometric_signature=np.random.rand(128),
            confidence=0.8
        )
        
        # Fuse with None visual
        fused = fusion.fuse_features(spatial, None)
        
        # Verify result
        assert isinstance(fused, FusedFeatures)
        assert fused.spatial == spatial
        assert fused.visual.confidence == 0.0  # Dummy visual features
        assert len(fused.combined_signature) > 0
        assert fused.fusion_confidence == 0.8  # Should match spatial confidence
    
    def test_fuse_features_visual_only(self):
        """Test feature fusion with visual features only (spatial sensor failed)."""
        fusion = FeatureFusion()
        
        # Create visual features
        visual = VisualFeatures(
            keypoints=[(10.0, 20.0), (30.0, 40.0)],
            descriptors=np.random.rand(2, 32),
            visual_signature=np.random.rand(128),
            confidence=0.7
        )
        
        # Fuse with None spatial
        fused = fusion.fuse_features(None, visual)
        
        # Verify result
        assert isinstance(fused, FusedFeatures)
        assert fused.spatial.confidence == 0.0  # Dummy spatial features
        assert fused.visual == visual
        assert len(fused.combined_signature) > 0
        assert fused.fusion_confidence == 0.7  # Should match visual confidence
    
    def test_fuse_features_both_none_raises_error(self):
        """Test that fusion raises error when both sensors fail."""
        fusion = FeatureFusion()
        
        with pytest.raises(ValueError, match="At least one sensor must provide features"):
            fusion.fuse_features(None, None)
    
    def test_determine_location_both_sensors(self):
        """Test location determination with both sensors."""
        fusion = FeatureFusion()
        
        # Create features for a living room (large area, many objects/keypoints)
        spatial = SpatialFeatures(
            room_dimensions=(6.0, 5.0, 2.5),  # Large area (30 sq m)
            wall_positions=[],
            object_positions=[(i, i, 0.5) for i in range(8)],  # Many objects
            geometric_signature=np.random.rand(128),
            confidence=0.8
        )
        
        visual = VisualFeatures(
            keypoints=[(i*10.0, i*10.0) for i in range(60)],  # Many keypoints
            descriptors=np.random.rand(60, 32),
            visual_signature=np.random.rand(128),
            confidence=0.7
        )
        
        fused = fusion.fuse_features(spatial, visual)
        result = fusion.determine_location(fused)
        
        # Verify result
        assert isinstance(result, LocationResult)
        assert isinstance(result.classification, LocationClassification)
        assert result.classification.label in fusion._location_labels
        assert 0.0 <= result.classification.confidence <= 1.0
        assert len(result.classification.alternative_labels) <= 3
        
        # Verify metadata
        assert "fusion_confidence" in result.metadata
        assert "low_confidence" in result.metadata
        assert result.metadata["spatial_available"] is True
        assert result.metadata["visual_available"] is True
    
    def test_determine_location_spatial_only(self):
        """Test location determination with spatial features only."""
        fusion = FeatureFusion()
        
        # Create spatial features for a hallway (long and narrow)
        spatial = SpatialFeatures(
            room_dimensions=(10.0, 2.0, 2.5),  # High aspect ratio
            wall_positions=[],
            object_positions=[],
            geometric_signature=np.random.rand(128),
            confidence=0.8
        )
        
        fused = fusion.fuse_features(spatial, None)
        result = fusion.determine_location(fused)
        
        # Verify result
        assert isinstance(result, LocationResult)
        assert result.metadata["spatial_available"] is True
        assert result.metadata["visual_available"] is False
    
    def test_determine_location_visual_only(self):
        """Test location determination with visual features only."""
        fusion = FeatureFusion()
        
        # Create visual features
        visual = VisualFeatures(
            keypoints=[(i*10.0, i*10.0) for i in range(25)],
            descriptors=np.random.rand(25, 32),
            visual_signature=np.random.rand(128),
            confidence=0.7
        )
        
        fused = fusion.fuse_features(None, visual)
        result = fusion.determine_location(fused)
        
        # Verify result
        assert isinstance(result, LocationResult)
        assert result.metadata["spatial_available"] is False
        assert result.metadata["visual_available"] is True
    
    def test_low_confidence_indication(self):
        """Test that low confidence is indicated in metadata."""
        fusion = FeatureFusion(confidence_threshold=0.8)
        
        # Create features with low confidence
        spatial = SpatialFeatures(
            room_dimensions=(3.0, 3.0, 2.5),
            wall_positions=[],
            object_positions=[],
            geometric_signature=np.random.rand(128),
            confidence=0.3
        )
        
        visual = VisualFeatures(
            keypoints=[(10.0, 20.0)],
            descriptors=np.random.rand(1, 32),
            visual_signature=np.random.rand(128),
            confidence=0.2
        )
        
        fused = fusion.fuse_features(spatial, visual)
        result = fusion.determine_location(fused)
        
        # Verify low confidence indication
        assert result.metadata["low_confidence"] is True
        assert "uncertainty_warning" in result.metadata
    
    def test_location_hash_generation(self):
        """Test that location hash is generated."""
        fusion = FeatureFusion()
        
        spatial = SpatialFeatures(
            room_dimensions=(5.0, 4.0, 2.5),
            wall_positions=[],
            object_positions=[],
            geometric_signature=np.random.rand(128),
            confidence=0.8
        )
        
        visual = VisualFeatures(
            keypoints=[(10.0, 20.0)],
            descriptors=np.random.rand(1, 32),
            visual_signature=np.random.rand(128),
            confidence=0.7
        )
        
        fused = fusion.fuse_features(spatial, visual)
        result = fusion.determine_location(fused)
        
        # Verify location hash
        assert result.location_hash is not None
        assert len(result.location_hash.hash_value) == 32  # SHA-256 produces 32 bytes
        assert result.location_hash.algorithm == "SHA-256"
        assert result.location_hash.timestamp > 0
    
    def test_different_signature_lengths(self):
        """Test fusion with different signature lengths."""
        fusion = FeatureFusion()
        
        # Create features with different signature lengths
        spatial = SpatialFeatures(
            room_dimensions=(5.0, 4.0, 2.5),
            wall_positions=[],
            object_positions=[],
            geometric_signature=np.random.rand(64),  # Shorter signature
            confidence=0.8
        )
        
        visual = VisualFeatures(
            keypoints=[(10.0, 20.0)],
            descriptors=np.random.rand(1, 32),
            visual_signature=np.random.rand(128),  # Longer signature
            confidence=0.7
        )
        
        # Should handle different lengths gracefully
        fused = fusion.fuse_features(spatial, visual)
        
        assert isinstance(fused, FusedFeatures)
        assert len(fused.combined_signature) > 0
    
    def test_zero_confidence_handling(self):
        """Test fusion with zero confidence from both sensors."""
        fusion = FeatureFusion()
        
        spatial = SpatialFeatures(
            room_dimensions=(5.0, 4.0, 2.5),
            wall_positions=[],
            object_positions=[],
            geometric_signature=np.random.rand(128),
            confidence=0.0
        )
        
        visual = VisualFeatures(
            keypoints=[],
            descriptors=np.array([]).reshape(0, 32),
            visual_signature=np.random.rand(128),
            confidence=0.0
        )
        
        # Should handle zero confidence gracefully (equal weights)
        fused = fusion.fuse_features(spatial, visual)
        
        assert isinstance(fused, FusedFeatures)
        assert fused.fusion_confidence == 0.0
    
    def test_classification_consistency(self):
        """Test that same features produce consistent classification."""
        fusion = FeatureFusion()
        
        spatial = SpatialFeatures(
            room_dimensions=(5.0, 4.0, 2.5),
            wall_positions=[],
            object_positions=[(1.0, 1.0, 0.5)],
            geometric_signature=np.random.rand(128),
            confidence=0.8
        )
        
        visual = VisualFeatures(
            keypoints=[(10.0, 20.0)],
            descriptors=np.random.rand(1, 32),
            visual_signature=np.random.rand(128),
            confidence=0.7
        )
        
        fused = fusion.fuse_features(spatial, visual)
        
        # Determine location multiple times
        result1 = fusion.determine_location(fused)
        result2 = fusion.determine_location(fused)
        
        # Should produce consistent results
        assert result1.classification.label == result2.classification.label
        assert result1.classification.confidence == result2.classification.confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
