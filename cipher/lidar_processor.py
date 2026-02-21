"""
LiDAR processor for local point cloud processing and feature extraction.

This module processes LiDAR sensor data entirely on-device without network calls,
extracting spatial features and classifying locations for privacy-preserving navigation.
"""

import numpy as np
from typing import Optional
from cipher.models import PointCloud, SpatialFeatures, LocationClassification


class CorruptedDataError(Exception):
    """Raised when sensor data is corrupted or incomplete."""
    pass


class LiDARProcessor:
    """
    Process LiDAR point IPFS data locally to extract spatial features.
    
    All processing occurs on-device with no network calls, ensuring raw
    sensor data never leaves the robot.
    """
    
    def __init__(self):
        """Initialize the LiDAR processor."""
        self._location_labels = [
            "living_room", "kitchen", "hallway", "bedroom", 
            "bathroom", "office", "garage", "dining_room",
            "entryway", "closet", "laundry_room", "basement",
            "attic", "balcony", "patio", "workshop",
            "storage_room", "utility_room", "pantry", "library"
        ]
    
    def process_scan(self, point_IPFS: PointIPFS) -> SpatialFeatures:
        """
        Process LiDAR point IPFS and extract spatial features.
        
        Args:
            point_IPFS: Raw 3D point IPFS from LiDAR sensor
            
        Returns:
            SpatialFeatures: Extracted features (geometry, dimensions, object positions)
            
        Raises:
            CorruptedDataError: If point IPFS is invalid or incomplete
        """
        # Validate input
        if point_IPFS.points is None or len(point_IPFS.points) == 0:
            raise CorruptedDataError("Point IPFS is empty")
        
        if not isinstance(point_IPFS.points, np.ndarray):
            raise CorruptedDataError("Point IPFS points must be numpy array")
        
        if point_cloud.points.ndim != 2 or point_cloud.points.shape[1] != 3:
            raise CorruptedDataError("Point cloud must be Nx3 array")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(point_IPFS.points)) or np.any(np.isinf(point_IPFS.points)):
            raise CorruptedDataError("Point IPFS contains NaN or infinite values")
        
        # Extract room dimensions from point cloud bounds
        room_dimensions = self._extract_room_dimensions(point_IPFS.points)
        
        # Detect wall positions
        wall_positions = self._detect_walls(point_IPFS.points)
        
        # Detect object positions
        object_positions = self._detect_objects(point_IPFS.points)
        
        # Generate geometric signature
        geometric_signature = self._generate_geometric_signature(
            point_IPFS.points, room_dimensions, wall_positions, object_positions
        )
        
        # Calculate confidence based on point density and coverage
        confidence = self._calculate_confidence(point_IPFS.points)
        
        return SpatialFeatures(
            room_dimensions=room_dimensions,
            wall_positions=wall_positions,
            object_positions=object_positions,
            geometric_signature=geometric_signature,
            confidence=confidence
        )
    
    def classify_location(self, features: SpatialFeatures) -> LocationClassification:
        """
        Classify location based on spatial features.
        
        Args:
            features: Extracted spatial features
            
        Returns:
            LocationClassification: Semantic label and confidence score
        """
        # Simple classification based on room dimensions and features
        length, width, height = features.room_dimensions
        area = length * width
        volume = area * height
        n_objects = len(features.object_positions)
        n_walls = len(features.wall_positions)
        
        # Classification logic based on spatial characteristics
        scores = {}
        
        # Living room: large area, multiple objects
        if area > 20.0 and n_objects > 5:
            scores["living_room"] = 0.7 + min(0.2, n_objects / 50.0)
        
        # Kitchen: medium area, many objects (appliances)
        if 10.0 < area < 25.0 and n_objects > 3:
            scores["kitchen"] = 0.6 + min(0.3, n_objects / 30.0)
        
        # Hallway: long and narrow
        aspect_ratio = max(length, width) / min(length, width)
        if aspect_ratio > 3.0 and area < 15.0:
            scores["hallway"] = 0.8
        
        # Bedroom: medium to large area, moderate objects
        if 12.0 < area < 30.0 and 2 < n_objects < 10:
            scores["bedroom"] = 0.65
        
        # Bathroom: small area, few objects
        if area < 10.0 and n_objects < 5:
            scores["bathroom"] = 0.7
        
        # Office: medium area, moderate objects
        if 10.0 < area < 20.0 and 3 < n_objects < 8:
            scores["office"] = 0.6
        
        # Garage: large area, high ceiling
        if area > 25.0 and height > 2.5:
            scores["garage"] = 0.65
        
        # Default to most likely based on area if no strong match
        if not scores:
            if area < 8.0:
                scores["closet"] = 0.4
            elif area < 15.0:
                scores["bedroom"] = 0.4
            elif area < 25.0:
                scores["living_room"] = 0.4
            else:
                scores["garage"] = 0.4
        
        # Sort by score and select top classification
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        label = sorted_scores[0][0]
        confidence = sorted_scores[0][1]
        
        # Alternative labels (top 3)
        alternative_labels = [(l, s) for l, s in sorted_scores[1:4]]
        
        return LocationClassification(
            label=label,
            confidence=confidence,
            alternative_labels=alternative_labels
        )
    
    def _extract_room_dimensions(self, points: np.ndarray) -> tuple:
        """Extract room dimensions from point cloud bounds."""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords
        
        # Ensure positive dimensions
        length = max(abs(dimensions[0]), 0.1)
        width = max(abs(dimensions[1]), 0.1)
        height = max(abs(dimensions[2]), 0.1)
        
        return (float(length), float(width), float(height))
    
    def _detect_walls(self, points: np.ndarray) -> list:
        """Detect wall positions using simple plane fitting."""
        walls = []
        
        # Simple wall detection: find points at extremes in x and y
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        
        # Create plane equations for 4 walls (ax + by + cz + d = 0)
        # Wall 1: x = min_x (normal: [1, 0, 0])
        walls.append(np.array([1.0, 0.0, 0.0, -min_x]))
        
        # Wall 2: x = max_x (normal: [-1, 0, 0])
        walls.append(np.array([-1.0, 0.0, 0.0, max_x]))
        
        # Wall 3: y = min_y (normal: [0, 1, 0])
        walls.append(np.array([0.0, 1.0, 0.0, -min_y]))
        
        # Wall 4: y = max_y (normal: [0, -1, 0])
        walls.append(np.array([0.0, -1.0, 0.0, max_y]))
        
        return walls
    
    def _detect_objects(self, points: np.ndarray) -> list:
        """Detect object positions using simple clustering."""
        objects = []
        
        # Simple object detection: divide space into grid and find clusters
        if len(points) < 10:
            return objects
        
        # Use a simple grid-based approach
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Create a coarse grid
        grid_size = 0.5  # 50cm cells
        x_bins = int((max_coords[0] - min_coords[0]) / grid_size) + 1
        y_bins = int((max_coords[1] - min_coords[1]) / grid_size) + 1
        
        # Limit grid size to avoid excessive computation
        x_bins = min(x_bins, 20)
        y_bins = min(y_bins, 20)
        
        # Count points in each grid cell
        grid = np.zeros((x_bins, y_bins))
        for point in points:
            x_idx = int((point[0] - min_coords[0]) / grid_size)
            y_idx = int((point[1] - min_coords[1]) / grid_size)
            x_idx = min(x_idx, x_bins - 1)
            y_idx = min(y_idx, y_bins - 1)
            grid[x_idx, y_idx] += 1
        
        # Find cells with significant point density (potential objects)
        threshold = np.mean(grid) + np.std(grid)
        for i in range(x_bins):
            for j in range(y_bins):
                if grid[i, j] > threshold:
                    x = min_coords[0] + (i + 0.5) * grid_size
                    y = min_coords[1] + (j + 0.5) * grid_size
                    z = np.mean(points[:, 2])  # Average height
                    objects.append((float(x), float(y), float(z)))
        
        return objects
    
    def _generate_geometric_signature(
        self, 
        points: np.ndarray, 
        dimensions: tuple,
        walls: list,
        objects: list
    ) -> np.ndarray:
        """Generate normalized geometric signature for hashing."""
        # Create a feature vector combining various geometric properties
        features = []
        
        # Room dimension features (normalized)
        max_dim = max(dimensions)
        features.extend([d / max_dim for d in dimensions])
        
        # Volume and area features
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        area = dimensions[0] * dimensions[1]
        features.append(np.log1p(volume) / 10.0)  # Log-scaled volume
        features.append(np.log1p(area) / 10.0)    # Log-scaled area
        
        # Aspect ratios
        features.append(dimensions[0] / dimensions[1])
        features.append(dimensions[1] / dimensions[2])
        
        # Object count (normalized)
        features.append(min(len(objects) / 20.0, 1.0))
        
        # Point cloud statistics
        features.append(np.mean(points[:, 2]) / dimensions[2])  # Relative mean height
        features.append(np.std(points[:, 2]) / dimensions[2])   # Relative height variance
        
        # Pad or truncate to fixed size (128 dimensions)
        signature_size = 128
        if len(features) < signature_size:
            # Pad with point cloud histogram features
            hist_x, _ = np.histogram(points[:, 0], bins=20, density=True)
            hist_y, _ = np.histogram(points[:, 1], bins=20, density=True)
            hist_z, _ = np.histogram(points[:, 2], bins=20, density=True)
            
            features.extend(hist_x.tolist())
            features.extend(hist_y.tolist())
            features.extend(hist_z.tolist())
        
        # Ensure exactly signature_size dimensions
        features = features[:signature_size]
        if len(features) < signature_size:
            features.extend([0.0] * (signature_size - len(features)))
        
        return np.array(features, dtype=np.float64)
    
    def _calculate_confidence(self, points: np.ndarray) -> float:
        """Calculate confidence score based on point cloud quality."""
        # Base confidence on point density
        n_points = len(points)
        
        # More points = higher confidence (up to a limit)
        density_score = min(n_points / 1000.0, 1.0)
        
        # Check spatial coverage
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        ranges = max_coords - min_coords
        
        # Penalize if range is too small (incomplete scan)
        coverage_score = 1.0
        if np.any(ranges < 0.5):  # Less than 50cm in any dimension
            coverage_score = 0.5
        
        # Combine scores
        confidence = 0.6 * density_score + 0.4 * coverage_score
        
        return float(np.clip(confidence, 0.0, 1.0))
