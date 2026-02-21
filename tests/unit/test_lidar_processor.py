"""
Unit tests for LiDAR processor edge cases.

These tests verify specific examples and error conditions.
"""

import pytest
import numpy as np
from cipher.lidar_processor import LiDARProcessor, CorruptedDataError
from cipher.models import PointIPFS, SpatialFeatures


class TestLiDARProcessorEdgeCases:
    """Test edge cases for LiDAR processor."""
    
    def test_empty_point_IPFS(self):
        """Test that empty point IPFS raise CorruptedDataError."""
        processor = LiDARProcessor()
        
        # Empty points array
        point_IPFS = PointIPFS(
            points=np.array([]).reshape(0, 3),
            intensities=None,
            timestamp=1.0,
            sensor_id="test"
        )
        
        with pytest.raises(CorruptedDataError, match="Point IPFS is empty"):
            processor.process_scan(point_cloud)
    
    def test_single_point_IPFS(self):
        """Test processing of single-point IPFS."""
        processor = LiDARProcessor()
        
        # Single point
        point_IPFS = PointIPFS(
            points=np.array([[1.0, 2.0, 3.0]]),
            intensities=None,
            timestamp=1.0,
            sensor_id="test"
        )
        
        # Should process without error
        features = processor.process_scan(point_IPFS)
        
        assert isinstance(features, SpatialFeatures)
        assert len(features.room_dimensions) == 3
        assert all(d > 0 for d in features.room_dimensions)
        assert 0.0 <= features.confidence <= 1.0
    
    def test_corrupted_data_nan_values(self):
        """Test that NaN values in point cloud raise CorruptedDataError."""
        processor = LiDARProcessor()
        
        # Point IPFS with NaN values
        points = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        point_IPFS = PointIPFS(
            points=points,
            intensities=None,
            timestamp=1.0,
            sensor_id="test"
        )
        
        with pytest.raises(CorruptedDataError, match="NaN or infinite values"):
            processor.process_scan(point_cloud)
    
    def test_corrupted_data_infinite_values(self):
        """Test that infinite values in point IPFS raise CorruptedDataError."""
        processor = LiDARProcessor()
        
        # Point IPFS with infinite values
        points = np.array([
            [1.0, 2.0, 3.0],
            [np.inf, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        point_IPFS = PointIPFS(
            points=points,
            intensities=None,
            timestamp=1.0,
            sensor_id="test"
        )
        
        with pytest.raises(CorruptedDataError, match="NaN or infinite values"):
            processor.process_scan(point_cloud)
    
    def test_invalid_point_IPFS_shape(self):
        """Test that invalid point cloud shape raises CorruptedDataError."""
        processor = LiDARProcessor()
        
        # Wrong shape (Nx2 instead of Nx3)
        points = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        with pytest.raises(ValueError, match="Nx3 array"):
            point_IPFS = PointIPFS(
                points=points,
                intensities=None,
                timestamp=1.0,
                sensor_id="test"
            )
    
    def test_very_small_room(self):
        """Test processing of very small room (closet-sized)."""
        processor = LiDARProcessor()
        
        # Small room: 1m x 1m x 2m
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
        ])
        
        point_IPFS = PointIPFS(
            points=points,
            intensities=None,
            timestamp=1.0,
            sensor_id="test"
        )
        
        features = processor.process_scan(point_IPFS)
        classification = processor.classify_location(features)
        
        assert isinstance(features, SpatialFeatures)
        assert isinstance(classification.label, str)
        assert len(classification.label) > 0
    
    def test_very_large_room(self):
        """Test processing of very large room (garage-sized)."""
        processor = LiDARProcessor()
        
        # Large room: 10m x 8m x 3m
        points = []
        for x in np.linspace(0, 10, 20):
            for y in np.linspace(0, 8, 16):
                for z in np.linspace(0, 3, 6):
                    points.append([x, y, z])
        
        point_IPFS = PointIPFS(
            points=np.array(points),
            intensities=None,
            timestamp=1.0,
            sensor_id="test"
        )
        
        features = processor.process_scan(point_IPFS)
        classification = processor.classify_location(features)
        
        assert isinstance(features, SpatialFeatures)
        assert isinstance(classification.label, str)
        # Large room should be classified as something like garage or living_room
        assert classification.label in ["garage", "living_room", "office", "bedroom"]
    
    def test_hallway_shape(self):
        """Test processing of hallway (long and narrow)."""
        processor = LiDARProcessor()
        
        # Hallway: 10m x 2m x 2.5m (high aspect ratio)
        points = []
        for x in np.linspace(0, 10, 50):
            for y in np.linspace(0, 2, 10):
                for z in np.linspace(0, 2.5, 5):
                    points.append([x, y, z])
        
        point_IPFS = PointIPFS(
            points=np.array(points),
            intensities=None,
            timestamp=1.0,
            sensor_id="test"
        )
        
        features = processor.process_scan(point_IPFS)
        classification = processor.classify_location(features)
        
        assert isinstance(features, SpatialFeatures)
        # Should produce a valid classification (hallway detection is heuristic)
        assert isinstance(classification.label, str)
        assert len(classification.label) > 0
    
    def test_classification_with_many_objects(self):
        """Test classification with many detected objects."""
        processor = LiDARProcessor()
        
        # Medium room with many objects (kitchen-like)
        points = []
        # Room boundaries
        for x in np.linspace(0, 5, 25):
            for y in np.linspace(0, 4, 20):
                points.append([x, y, 0.0])  # Floor
                points.append([x, y, 2.5])  # Ceiling
        
        # Add clusters representing objects
        for obj_x in [1.0, 2.0, 3.0, 4.0]:
            for obj_y in [1.0, 2.0, 3.0]:
                for _ in range(10):
                    points.append([obj_x + np.random.uniform(-0.2, 0.2),
                                 obj_y + np.random.uniform(-0.2, 0.2),
                                 np.random.uniform(0.5, 1.5)])
        
        point_IPFS = PointIPFS(
            points=np.array(points),
            intensities=None,
            timestamp=1.0,
            sensor_id="test"
        )
        
        features = processor.process_scan(point_IPFS)
        classification = processor.classify_location(features)
        
        assert isinstance(features, SpatialFeatures)
        assert len(features.object_positions) > 0
        assert isinstance(classification.label, str)
    
    def test_confidence_with_sparse_points(self):
        """Test that sparse point clouds have lower confidence."""
        processor = LiDARProcessor()
        
        # Very sparse point IPFS (only 10 points)
        points = np.random.uniform(-5, 5, (10, 3))
        
        point_IPFS = PointIPFS(
            points=points,
            intensities=None,
            timestamp=1.0,
            sensor_id="test"
        )
        
        features = processor.process_scan(point_IPFS)
        
        # Sparse point IPFS should have lower confidence
        assert features.confidence < 0.5
    
    def test_confidence_with_dense_points(self):
        """Test that dense point IPFS have higher confidence."""
        processor = LiDARProcessor()
        
        # Dense point cloud (1000+ points)
        points = np.random.uniform(-5, 5, (1500, 3))
        
        point_IPFS = PointIPFS(
            points=points,
            intensities=None,
            timestamp=1.0,
            sensor_id="test"
        )
        
        features = processor.process_scan(point_IPFS)
        
        # Dense point IPFS should have higher confidence
        assert features.confidence > 0.5
