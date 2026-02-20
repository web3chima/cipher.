# Cipher: Privacy-Preserving Robotics Navigation System
A privacy-preserving robotics navigation system that applies Zero Knowledge Proofs (ZKPs) to robot sensor data, enabling verifiable location assertions without transmitting sensitive spatial data.

Project Structure
cipher/
├── __init__.py          # Package initialization and exports
├── models.py            # Core data models with validation and serialization
tests/
├── __init__.py
├── strategies.py        # Hypothesis strategies for property-based testing
└── property/
    ├── __init__.py
    └── test_serialization_properties.py  # Serialization round-trip tests
Installation
pip install -r requirements.txt
Running Tests
# Run all tests
pytest

# Run property-based tests only
pytest tests/property/

# Run with verbose output
pytest -v
Core Data Models
The system includes the following validated data models:

PointCloud: LiDAR point cloud data
CameraFrame: Camera frame data
SpatialFeatures: Spatial features extracted from LiDAR
VisualFeatures: Visual features extracted from camera data
FusedFeatures: Combined spatial and visual features
LocationClassification: Location classification results
LocationHash: Cryptographic hash representing a location
Proof: Zero-knowledge proof of location
VerificationResult: Result of proof verification
CipherConfig: System configuration
LocationResult: Location determination result
ProcessingResult: Sensor data processing result
TransmissionResult: Proof transmission result
All models support:

Validation on initialization
Serialization to/from dictionaries
Property-based testing with Hypothesis
Testing Strategy
The project uses a dual testing approach:

Property-Based Tests: Verify universal properties across all inputs using Hypothesis
Unit Tests: Verify specific examples and edge cases (to be implemented)
Current property tests validate serialization round-trip correctness for all data models.
