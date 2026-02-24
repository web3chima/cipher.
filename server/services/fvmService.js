"use strict";
/**
 * fvmService.js
 *
 * Submit verified Groth16 proofs to the CipherLocationRegistry contract
 * deployed on the Filecoin Virtual Machine (FVM).
 *
 * Called after ipfsService.storeProof() returns a CID:
 *   1. Parse Groth16 proof JSON into (a, b, c, publicSignals)
 *   2. Call CipherLocationRegistry.submitProof() on-chain
 *   3. Return the transaction hash and on-chain entry index
 *
 * Environment variables required in server/.env:
 *   CIPHER_REGISTRY_ADDRESS  — deployed CipherLocationRegistry address
 *   DEPLOYER_PRIVATE_KEY     — wallet that pays gas (robot's wallet)
 *   FVM_CALIBRATION_RPC      — RPC endpoint (default: Glif Calibration)
 *
 * Dependencies (add to server/package.json):
 *   "ethers": "^6.13.0"
 */

const { ethers } = require("ethers");

// ─── CipherLocationRegistry ABI (submitProof + events) ───────────────────────
const REGISTRY_ABI = [
  "function submitProof(uint256[2] a, uint256[2][2] b, uint256[2] c, uint256[3] publicSignals, string ipfsCID) external",
  "function getEntry(uint256 index) external view returns (tuple(bytes32 locationHash, uint256 timestamp, string ipfsCID, address device, bool verified))",
  "function totalEntries() external view returns (uint256)",
  "event LocationVerified(address indexed device, bytes32 indexed locationHash, string ipfsCID, uint256 entryIndex)",
];

/** @type {ethers.Provider | null} */
let _provider = null;
/** @type {ethers.Wallet | null} */
let _wallet   = null;
/** @type {ethers.Contract | null} */
let _registry = null;

/**
 * Lazily initialise provider, wallet, and contract instances.
 * @returns {{ provider: ethers.Provider, wallet: ethers.Wallet, registry: ethers.Contract }}
 */
function getContracts() {
  if (_registry) return { provider: _provider, wallet: _wallet, registry: _registry };

  const rpc = process.env.FVM_CALIBRATION_RPC
    || "https://api.calibration.node.glif.io/rpc/v1";

  const registryAddress = process.env.CIPHER_REGISTRY_ADDRESS;
  if (!registryAddress) {
    throw new Error("fvmService: CIPHER_REGISTRY_ADDRESS is not set in .env");
  }

  const privateKey = process.env.DEPLOYER_PRIVATE_KEY;
  if (!privateKey) {
    throw new Error("fvmService: DEPLOYER_PRIVATE_KEY is not set in .env");
  }

  _provider = new ethers.JsonRpcProvider(rpc);
  _wallet   = new ethers.Wallet(privateKey, _provider);
  _registry = new ethers.Contract(registryAddress, REGISTRY_ABI, _wallet);

  return { provider: _provider, wallet: _wallet, registry: _registry };
}

/**
 * Parse a raw Groth16 proof object (snarkjs / lurk compress output) into
 * the (a, b, c, publicSignals) tuple expected by the Solidity verifier.
 *
 * @param {object} proofJson   Proof JSON from lurk compress
 * @param {object} publicJson  Public signals JSON (or embedded in proofJson)
 * @returns {{ a, b, c, publicSignals }}
 */
function parseProof(proofJson, publicJson) {
  const proof   = proofJson.proof   ?? proofJson;
  const signals = publicJson        ?? proofJson.publicSignals;

  // snarkjs proof shape: { pi_a: [x,y,1], pi_b: [[x0,x1],[y0,y1],[1,0]], pi_c: [x,y,1] }
  const a = [proof.pi_a[0], proof.pi_a[1]];
  const b = [
    [proof.pi_b[0][0], proof.pi_b[0][1]],
    [proof.pi_b[1][0], proof.pi_b[1][1]],
  ];
  const c = [proof.pi_c[0], proof.pi_c[1]];

  // publicSignals: [locationHash, timestamp, deviceId] as decimal strings
  const publicSignals = signals.slice(0, 3).map((s) => BigInt(s));

  return { a, b, c, publicSignals };
}

/**
 * Submit a Groth16 proof to CipherLocationRegistry on FVM.
 *
 * @param {object} opts
 * @param {object}   opts.proofJson     Groth16 proof (from lurk compress)
 * @param {object}   [opts.publicJson]  Public signals (if separate from proof)
 * @param {string}   opts.ipfsCID       CID returned by ipfsService.storeProof()
 *
 * @returns {Promise<{ txHash: string, entryIndex: number, locationHash: string }>}
 */
async function submitProofToFVM({ proofJson, publicJson, ipfsCID }) {
  if (!ipfsCID) throw new Error("fvmService.submitProofToFVM: ipfsCID is required");

  const { registry } = getContracts();
  const { a, b, c, publicSignals } = parseProof(proofJson, publicJson);

  console.log("FVM: submitting proof to CipherLocationRegistry...");
  console.log(`  locationHash: 0x${publicSignals[0].toString(16)}`);
  console.log(`  ipfsCID:      ${ipfsCID}`);

  const tx = await registry.submitProof(a, b, c, publicSignals, ipfsCID);
  console.log(`FVM: tx sent → ${tx.hash}`);

  const receipt = await tx.wait();
  console.log(`FVM: confirmed in block ${receipt.blockNumber}`);

  // Parse LocationVerified event to get entryIndex
  const iface = new ethers.Interface(REGISTRY_ABI);
  let entryIndex = null;
  for (const log of receipt.logs) {
    try {
      const parsed = iface.parseLog(log);
      if (parsed.name === "LocationVerified") {
        entryIndex = Number(parsed.args.entryIndex);
        break;
      }
    } catch (_) { /* not our event */ }
  }

  return {
    txHash:       tx.hash,
    entryIndex,
    locationHash: "0x" + publicSignals[0].toString(16).padStart(64, "0"),
  };
}

/**
 * Retrieve an on-chain entry by index.
 * @param {number} index
 * @returns {Promise<object>}
 */
async function getEntry(index) {
  const { registry } = getContracts();
  const entry = await registry.getEntry(index);
  return {
    locationHash: entry.locationHash,
    timestamp:    Number(entry.timestamp),
    ipfsCID:      entry.ipfsCID,
    device:       entry.device,
    verified:     entry.verified,
  };
}

module.exports = { submitProofToFVM, getEntry, parseProof };
