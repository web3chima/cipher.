# Cipher: Privacy-Preserving Robotics Navigation System
A privacy-preserving robotics navigation system that applies Zero Knowledge Proofs (ZKPs) to robot sensor data, enabling verifiable location assertions without transmitting sensitive spatial data.

## Project Structure

```
cipher/
├── __init__.py          # Package initialization and exports
├── models.py            # Core data models with validation and serialization
tests/
├── __init__.py
├── strategies.py        # Hypothesis strategies for property-based testing
└── property/
    ├── __init__.py
    └── test_serialization_properties.py  # Serialization round-trip tests                              
circuits/
│   ├── location_proof.lurk              ← single-sensor proof (LiDAR or VSLAM)
│   ├── fused_location_proof.lurk        ← weighted LiDAR + VSLAM fusion proof
│   └── lidar_scan_proof.lurk            ← incremental Nova IVC over point cloud
├── contracts/
│   ├── Groth16Verifier.sol              ← BN254 pairing verifier (fill VK after setup)
│   └── CipherLocationRegistry.sol       ← FVM registry — stores CID + verified hash
├── cipher/
│   └── zkp_lurk_prover.py              ← Python prover: quantise → witness → Nova → Groth16
├── scripts/
│   └── deploy.js                        ← Hardhat deploy to FVM Calibration/Mainnet
└── hardhat.config.js                    ← FVM network config (chainId 314159 / 314)

cipher./server/services/
└── ipfsService.js                       ← Helia IPFS node: storeProof() / fetchProof()

```
```
Robot/
  └─ LiDARProcessor / VSLAMProcessor  
  └─ FeatureFusion                   
  └─ LurkProver.generate_proof()     
       │  prove-fused-location.lurk
       │  Nova IVC folds each scan step
       │
       ▼
  proof.lurk.json  +  public_outputs: [locationHash, timestamp, deviceId]
       │
       └─ lurk compress → proof.groth16.json   (for FVM submission)
       │
       ▼
IPFS (Helia)
  └─ storeProof({ proof, locationHash }) → CID
       │
       ▼
FVM CipherLocationRegistry.submitProof(a,b,c, publicSignals, CID)

```
```
Trusted Setup (one-time)
  └─ Lit Action: multi-party snarkjs ceremony inside TEE
  └─ Encrypt *.zkey with TPKE → store on IPFS
 
Robot Startup
  └─ getSessionSigs() against robot's PKP
  └─ decryptAndCombine(encrypted_zkey) — ACC: entriesForDevice > 0
  └─ zkey available in memory (TEE), never written to disk

Per Sensor Reading
  └─ LiDARProcessor / VSLAMProcessor → 128-dim feature vectors (on-device)
  └─ FeatureFusion → combined_signature
  └─ LurkProver: Nova IVC → lurk compress (using in-memory zkey) → Groth16 proof
  └─ Upload proof bundle to IPFS → CID

FVM Submission (via Lit Action + PKP)
  └─ executeJs(CipherSubmitAction, { a, b, c, publicSignals, ipfsCID })
  └─ Lit Action validates: non-zero hash, valid CID, authorized deviceId
  └─ PKP signs submitProof calldata via threshold ECDSA
  └─ CipherLocationRegistry.submitProof() verified on-chain
```

## Installation

```bash
Prerequisites → Clone → Python env → pip install → npm install → .env setup → start server → start frontend

```

## Running Tests

```bash
# Run all tests
pytest

# Run property-based tests only
pytest tests/property/

# Run with verbose output
pytest -v
```

## Core Data Models

The system includes the following validated data models:

- **PointIPFS**: LiDAR point IPFS data
- **CameraFrame**: Camera frame data
- **SpatialFeatures**: Spatial features extracted from LiDAR
- **VisualFeatures**: Visual features extracted from camera data
- **FusedFeatures**: Combined spatial and visual features
- **LocationClassification**: Location classification results
- **LocationHash**: Cryptographic hash representing a location
- **Proof**: Zero-knowledge proof of location
- **VerificationResult**: Result of proof verification
- **CipherConfig**: System configuration
- **LocationResult**: Location determination result
- **ProcessingResult**: Sensor data processing result
- **TransmissionResult**: Proof transmission result

All models support:
- Validation on initialization
- Serialization to/from dictionaries
- Property-based testing with Hypothesis

## Testing Strategy

The project uses a dual testing approach:

1. **Property-Based Tests**: Verify universal properties across all inputs using Hypothesis
2. **Unit Tests**: Verify specific examples and edge cases (to be implemented)

Current property tests validate serialization round-trip correctness for all data models.
