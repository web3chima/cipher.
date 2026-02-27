"use strict";
/**
 * ipfsService.js
 *
 * Stores Cipher ZK proof bundles on IPFS via a local Helia node.
 *
 * What is stored (proof bundle):
 *   { deviceId, locationHash, proof, publicSignals, timestamp, protocol }
 *
 * What is NEVER stored:
 *   - Raw LiDAR point clouds
 *   - Raw camera frames / images
 *   - Visual or spatial feature vectors
 *
 * The returned CID is submitted to CipherLocationRegistry.submitProof()
 * on the Filecoin Virtual Machine alongside the Groth16 proof.
 *
 * Helia is ESM-only — loaded via dynamic import() so this CJS file works.
 *
 * Dependencies:
 *   "helia": "^4.0.0"
 *   "@helia/json": "^4.0.0"
 *   "multiformats": "^13.0.0"
 */

/** @type {import("helia").Helia | null} */
let _node = null;
let _json = null;

/**
 * Lazily initialise and return the shared Helia + json codec.
 * Uses dynamic import() to load ESM-only Helia from a CJS module.
 */
async function getClient() {
  if (!_node) {
    const { createHelia }     = await import("helia");
    const { json: heliaJson } = await import("@helia/json");

    _node = await createHelia();
    _json = heliaJson(_node);
    console.log("[ipfs] Helia node started");
  }
  return _json;
}

/**
 * Store a ZK proof bundle on IPFS and return its CID string.
 *
 * @param {object} opts
 * @param {string}   opts.deviceId       Robot wallet address or identifier
 * @param {string}   opts.locationHash   Hex-encoded Poseidon hash (public signal)
 * @param {object}   opts.proof          Groth16 proof object { pi_a, pi_b, pi_c }
 * @param {string[]} opts.publicSignals  [locationHash, timestamp, deviceId] as decimal strings
 * @param {number}   opts.timestamp      Unix epoch (seconds)
 * @param {string}   [opts.protocol]     Proof protocol — defaults to "Groth16"
 *
 * @returns {Promise<string>} IPFS CID (e.g. "bafyrei...")
 */
async function storeProof({
  deviceId,
  locationHash,
  proof,
  publicSignals,
  timestamp,
  protocol = "Groth16",
}) {
  if (!deviceId)      throw new Error("ipfsService.storeProof: deviceId is required");
  if (!locationHash)  throw new Error("ipfsService.storeProof: locationHash is required");
  if (!proof)         throw new Error("ipfsService.storeProof: proof is required");
  if (!publicSignals) throw new Error("ipfsService.storeProof: publicSignals is required");

  const j = await getClient();

  const bundle = {
    deviceId,
    locationHash,
    proof,
    publicSignals,
    timestamp: timestamp ?? Math.floor(Date.now() / 1000),
    protocol,
    _cipher: "v1",
  };

  const cid    = await j.add(bundle);
  const cidStr = cid.toString();

  console.log(`[ipfs] stored proof bundle → ${cidStr}`);
  return cidStr;
}

/**
 * Retrieve a proof bundle from IPFS by CID string.
 *
 * @param {string} cidStr CID returned by storeProof()
 * @returns {Promise<object>} The proof bundle
 */
async function fetchProof(cidStr) {
  const { CID }  = await import("multiformats/cid");
  const j        = await getClient();
  return j.get(CID.parse(cidStr));
}

/**
 * Gracefully shut down the Helia node.
 * Call on server shutdown (SIGTERM / SIGINT).
 */
async function stopNode() {
  if (_node) {
    await _node.stop();
    _node = null;
    _json = null;
    console.log("[ipfs] Helia node stopped");
  }
}

module.exports = { storeProof, fetchProof, stopNode };
