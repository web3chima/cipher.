"""
VSLAM processor for local camera frame processing and visual feature extraction.

This module processes camera frames entirely on-device without network calls or
persistent image storage, extracting visual features and classifying locations
for privacy-preserving navigation.
"""

import numpy as np
from typing import Optional
from cipher.models import CameraFrame, VisualFeatures, LocationClassification


class CorruptedDataError(Exception):
    """Raised when sensor data is corrupted or incomplete."""
    pass


class VSLAMProcessor:
    """
    Process camera frames locally to extract visual features.
    
    All processing occurs on-device with no network calls and no persistent
    image storage, ensuring raw sensor data never leaves the robot.
    """
    
    def __init__(self):
        """Initialize the VSLAM processor."""
        self._location_labels = [
            "living_room", "kitchen", "hallway", "bedroom", 
            "bathroom", "office", "garage", "dining_room",
            "entryway", "closet", "laundry_room", "basement",
            "attic", "balcony", "patio", "workshop",
            "storage_room", "utility_room", "pantry", "library"
        ]
    
    def process_frame(self, image: CameraFrame) -> VisualFeatures:
        """
        Process camera frame and extract visual features.
        
        Args:
            image: Raw camera frame
            
        Returns:
            VisualFeatures: Extracted features (keypoints, descriptors)
            
        Raises:
            CorruptedDataError: If image is invalid or incomplete
        """
        # Validate input
        if image.image is None or image.image.size == 0:
            raise CorruptedDataError("Camera frame is empty")
        
        if not isinstance(image.image, np.ndarray):
            raise CorruptedDataError("Camera frame image must be numpy array")
        
        if image.image.ndim != 3 or image.image.shape[2] != 3:
            raise CorruptedDataError("Camera frame must be HxWx3 array")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(image.image)) or np.any(np.isinf(image.image)):
            raise CorruptedDataError("Camera frame contains NaN or infinite values")
        
        # Extract keypoints using simplified ORB-like feature detection
        keypoints = self._extract_keypoints(image.image)
        
        # Generate descriptors for keypoints
        descriptors = self._generate_descriptors(image.image, keypoints)
        
        # Generate visual signature for location recognition
        visual_signature = self._generate_visual_signature(image.image, keypoints, descriptors)
        
        # Calculate confidence based on feature quality
        confidence = self._calculate_confidence(keypoints, descriptors)
        
        # Note: Raw image data is discarded after feature extraction
        # No persistent image files are created
        
        return VisualFeatures(
            keypoints=keypoints,
            descriptors=descriptors,
            visual_signature=visual_signature,
            confidence=confidence
        )
    
    def classify_location(self, features: VisualFeatures) -> LocationClassification:
        """
        Classify location based on visual features.
        
        Args:
            features: Extracted visual features
            
        Returns:
            LocationClassification: Semantic label and confidence score
        """
        # Simple classification based on visual characteristics
        n_keypoints = len(features.keypoints)
        descriptor_variance = np.var(features.descriptors) if features.descriptors.size > 0 else 0.0
        signature_mean = np.mean(features.visual_signature)
        signature_std = np.std(features.visual_signature)
        
        # Classification logic based on visual characteristics
        scores = {}
        
        # Living room: many features, high variance (diverse objects)
        if n_keypoints > 50 and descriptor_variance > 0.1:
            scores["living_room"] = 0.7 + min(0.2, n_keypoints / 500.0)
        
        # Kitchen: many features, moderate variance (appliances, cabinets)
        if 30 < n_keypoints < 100 and 0.05 < descriptor_variance < 0.15:
            scores["kitchen"] = 0.65 + min(0.25, n_keypoints / 400.0)
        
        # Hallway: few features, low variance (plain walls)
        if n_keypoints < 30 and descriptor_variance < 0.08:
            scores["hallway"] = 0.75
        
        # Bedroom: moderate features, moderate variance
        if 20 < n_keypoints < 80 and 0.05 < descriptor_variance < 0.12:
            scores["bedroom"] = 0.6
        
        # Bathroom: few features, low to moderate variance (tiles, fixtures)
        if n_keypoints < 40 and descriptor_variance < 0.1:
            scores["bathroom"] = 0.65
        
        # Office: many features, high variance (desk, shelves, equipment)
        if n_keypoints > 40 and descriptor_variance > 0.08:
            scores["office"] = 0.6
        
        # Garage: moderate features, variable lighting
        if 20 < n_keypoints < 60:
            scores["garage"] = 0.55
        
        # Default to most likely based on feature count if no strong match
        if not scores:
            if n_keypoints < 20:
                scores["closet"] = 0.4
            elif n_keypoints < 50:
                scores["bedroom"] = 0.4
            elif n_keypoints < 100:
                scores["living_room"] = 0.4
            else:
                scores["office"] = 0.4
        
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
    
    def _extract_keypoints(self, image: np.ndarray) -> list:
        """
        Extract keypoints from image using simplified corner detection.
        
        This is a lightweight implementation that doesn't require OpenCV.
        In production, this would use ORB or SIFT feature detection.
        """
        # Convert to grayscale
        gray = self._to_grayscale(image)
        
        # Simple corner detection using gradient-based approach
        # Compute gradients
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        
        # Compute corner response (simplified Harris corner detector)
        grad_xx = grad_x * grad_x
        grad_yy = grad_y * grad_y
        grad_xy = grad_x * grad_y
        
        # Apply Gaussian-like smoothing using simple averaging
        window_size = 5
        grad_xx_smooth = self._smooth(grad_xx, window_size)
        grad_yy_smooth = self._smooth(grad_yy, window_size)
        grad_xy_smooth = self._smooth(grad_xy, window_size)
        
        # Compute corner response
        det = grad_xx_smooth * grad_yy_smooth - grad_xy_smooth * grad_xy_smooth
        trace = grad_xx_smooth + grad_yy_smooth
        k = 0.04
        corner_response = det - k * trace * trace
        
        # Find local maxima as keypoints
        threshold = np.percentile(corner_response, 95)  # Top 5% responses
        keypoints = []
        
        h, w = corner_response.shape
        for y in range(10, h - 10, 5):  # Sample every 5 pixels
            for x in range(10, w - 10, 5):
                if corner_response[y, x] > threshold:
                    # Check if local maximum
                    local_region = corner_response[max(0, y-5):min(h, y+6), max(0, x-5):min(w, x+6)]
                    if corner_response[y, x] == np.max(local_region):
                        keypoints.append((float(x), float(y)))
        
        # Limit number of keypoints to avoid excessive computation
        if len(keypoints) > 200:
            # Keep strongest keypoints
            keypoint_strengths = [corner_response[int(y), int(x)] for x, y in keypoints]
            sorted_indices = np.argsort(keypoint_strengths)[::-1]
            keypoints = [keypoints[i] for i in sorted_indices[:200]]
        
        return keypoints
    
    def _generate_descriptors(self, image: np.ndarray, keypoints: list) -> np.ndarray:
        """
        Generate descriptors for keypoints.
        
        This is a simplified descriptor generation. In production, this would
        use ORB or SIFT descriptors.
        """
        if len(keypoints) == 0:
            return np.array([]).reshape(0, 32)
        
        gray = self._to_grayscale(image)
        h, w = gray.shape
        
        descriptors = []
        patch_size = 16
        descriptor_dim = 32
        
        for x, y in keypoints:
            x_int, y_int = int(x), int(y)
            
            # Extract patch around keypoint
            y_start = max(0, y_int - patch_size // 2)
            y_end = min(h, y_int + patch_size // 2)
            x_start = max(0, x_int - patch_size // 2)
            x_end = min(w, x_int + patch_size // 2)
            
            patch = gray[y_start:y_end, x_start:x_end]
            
            if patch.size == 0:
                # Use zero descriptor if patch is invalid
                descriptor = np.zeros(descriptor_dim)
            else:
                # Compute simple descriptor from patch statistics
                # Divide patch into 4x4 grid and compute statistics for each cell
                grid_size = 4
                cell_h = max(1, patch.shape[0] // grid_size)
                cell_w = max(1, patch.shape[1] // grid_size)
                
                descriptor_parts = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        cell = patch[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                        if cell.size > 0:
                            descriptor_parts.append(np.mean(cell))
                            descriptor_parts.append(np.std(cell))
                        else:
                            descriptor_parts.append(0.0)
                            descriptor_parts.append(0.0)
                
                # Pad or truncate to descriptor_dim
                descriptor_parts = descriptor_parts[:descriptor_dim]
                while len(descriptor_parts) < descriptor_dim:
                    descriptor_parts.append(0.0)
                
                descriptor = np.array(descriptor_parts)
                
                # Normalize descriptor
                norm = np.linalg.norm(descriptor)
                if norm > 0:
                    descriptor = descriptor / norm
            
            descriptors.append(descriptor)
        
        return np.array(descriptors, dtype=np.float64)
    
    def _generate_visual_signature(
        self, 
        image: np.ndarray, 
        keypoints: list,
        descriptors: np.ndarray
    ) -> np.ndarray:
        """Generate normalized visual signature for location recognition."""
        features = []
        
        # Image statistics
        gray = self._to_grayscale(image)
        features.append(np.mean(gray) / 255.0)
        features.append(np.std(gray) / 255.0)
        
        # Color statistics (RGB channels)
        for channel in range(3):
            features.append(np.mean(image[:, :, channel]) / 255.0)
            features.append(np.std(image[:, :, channel]) / 255.0)
        
        # Keypoint statistics
        features.append(min(len(keypoints) / 200.0, 1.0))  # Normalized keypoint count
        
        if len(keypoints) > 0:
            # Keypoint spatial distribution
            kp_x = [x for x, y in keypoints]
            kp_y = [y for x, y in keypoints]
            features.append(np.std(kp_x) / image.shape[1])
            features.append(np.std(kp_y) / image.shape[0])
        else:
            features.append(0.0)
            features.append(0.0)
        
        # Descriptor statistics
        if descriptors.size > 0:
            features.append(np.mean(descriptors))
            features.append(np.std(descriptors))
            features.append(np.min(descriptors))
            features.append(np.max(descriptors))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Image gradient statistics
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.append(np.mean(grad_magnitude) / 255.0)
        features.append(np.std(grad_magnitude) / 255.0)
        
        # Pad to fixed size (128 dimensions)
        signature_size = 128
        if len(features) < signature_size:
            # Add histogram features
            hist, _ = np.histogram(gray, bins=50, range=(0, 255), density=True)
            features.extend(hist.tolist())
        
        # Ensure exactly signature_size dimensions
        features = features[:signature_size]
        if len(features) < signature_size:
            features.extend([0.0] * (signature_size - len(features)))
        
        return np.array(features, dtype=np.float64)
    
    def _calculate_confidence(self, keypoints: list, descriptors: np.ndarray) -> float:
        """Calculate confidence score based on feature quality."""
        # Base confidence on number of keypoints
        n_keypoints = len(keypoints)
        keypoint_score = min(n_keypoints / 100.0, 1.0)
        
        # Descriptor quality score
        descriptor_score = 1.0
        if descriptors.size > 0:
            # Check descriptor variance (higher variance = more distinctive features)
            descriptor_variance = np.var(descriptors)
            descriptor_score = min(descriptor_variance / 0.1, 1.0)
        else:
            descriptor_score = 0.0
        
        # Combine scores
        confidence = 0.6 * keypoint_score + 0.4 * descriptor_score
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale."""
        # Standard RGB to grayscale conversion
        return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    def _smooth(self, array: np.ndarray, window_size: int) -> np.ndarray:
        """Apply simple box filter smoothing."""
        from scipy.ndimage import uniform_filter
        return uniform_filter(array, size=window_size, mode='constant')
