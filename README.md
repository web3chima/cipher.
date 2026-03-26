# Cipher: Privacy-Preserving Robotics Navigation System
A privacy-preserving robotics navigation system that applies Zero Knowledge Proofs (ZKPs) to robot sensor data, enabling verifiable location assertions without transmitting sensitive spatial data.

## Project Structure

```
cipher/
├── cipher/                         # Python ZK prover + sensor processing
│   ├── models.py                   # Core data models (PointIPFS, Proof, LocationHash, ...)
│   ├── lidar_processor.py          # LiDAR PointCloud → 128-dim SpatialFeatures
│   ├── vslam_processor.py          # CameraFrame → 128-dim VisualFeatures
│   ├── feature_fusion.py           # Weighted fusion → FusedFeatures
│   └── __init__.py
│
├── contracts/
│   └── CipherLicensePayment.sol    # FVM payment contract — license purchase + commitment
│
├── keys/                           # ZK proving keys (gitignored in production)
│   ├── location_proof.zkey
│   ├── fused_location_proof.zkey
│   └── lidar_scan_proof.zkey
│
├── server/                         # Node.js API server (port 3001)
│   ├── index.js                    # Express entry point
│   ├── routes/
│   │   ├── keys.js                 # Session sigs + zkey decrypt
│   │   ├── proof.js                # IPFS store + FVM submit
│   │   ├── license.js              # License key validation + purchase
│   │   ├── registry.js             # FVM registry queries
│   │   └── contact.js              # Telegram group creation
│   ├── services/
│   │   ├── litService.js           # Lit Protocol — session sigs, zkey decrypt
│   │   ├── fvmService.js           # FVM contract calls (ethers v6)
│   │   ├── ipfsService.js          # Helia IPFS — storeProof / fetchProof
│   │   ├── licenseKeyService.js    # Key generation + validation
│   │   └── telegramService.js      # MTProto supergroup creation
│   ├── lit-actions/
│   │   ├── cipher_auth_action.js   # PKP auth — validates device via FVM registry
│   │   ├── cipher_submit_action.js # PKP signing — submits proof to FVM
│   │   └── cipher_ceremony_action.js # Trusted setup ceremony inside TEE
│   ├── middleware/
│   │   └── requireLicense.js       # Gates proof routes via x-cipher-license header
│   ├── litAuthServer.mjs           # Lit session sig server (port 6380)
│   └── litAuthWorker.mjs           # Background worker — async PKP minting (BullMQ)
│
├── src/                            # React frontend (Vite, port 5173)
│   ├── App.jsx                     # Hash router
│   ├── prototypes/                 # Page compositions
│   └── components/
│       ├── sections/               # Header, Hero, SDKShowcase, Footer, ...
│       └── ui/                     # Button, ...
│
└── tests/                          # Python test suite
    ├── strategies.py               # Hypothesis strategies for all models
    ├── property/
    │   ├── test_serialization_properties.py   # Round-trip tests for all models
    │   ├── test_lidar_properties.py           # LiDAR processor behavioral tests
    │   └── test_vslam_properties.py           # VSLAM processor behavioral tests
    └── unit/                       # Unit test structure (tests not yet written)
        ├── test_lidar_processor.py
        ├── test_vslam_processor.py
        └── test_feature_fusion.py

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

Current Property-based tests cover two areas: serialization round-trip correctness for all data models, and behavioral properties of the LiDAR and VSLAM processors.
Unit test structure is in place — test files exist for LiDAR processor, VSLAM processor, and feature fusion. Tests are not yet written.

