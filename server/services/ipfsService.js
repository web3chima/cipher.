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
 * Dependencies (add to server/package.json):
 *   "helia": "^4.0.0"
 *   "@helia/json": "^4.0.0"
 */

const { createHelia }  = require("helia");
const { json: heliaJson } = require("@helia/json");

/** @type {import("helia").Helia | null} */
let _node = null;

/**
 * Lazily initialise and return the shared Helia IPFS node.
 * @returns {Promise<import("helia").Helia>}
 */
async function getNode() {
  if (!_node) {
    _node = await createHelia();
    console.log("Cipher IPFS: Helia node started");
  }
  return _node;
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

  const node = await getNode();
  const j    = heliaJson(node);

  const bundle = {
    deviceId,
    locationHash,
    proof,
    publicSignals,
    timestamp: timestamp ?? Math.floor(Date.now() / 1000),
    protocol,
    // Content-hash fingerprint so the on-chain verifier can match CID → proof
    _cipher: "v1",
  };

  const cid = await j.add(bundle);
  const cidStr = cid.toString();

  console.log(`Cipher IPFS: stored proof bundle → ${cidStr}`);
  return cidStr;
}

/**
 * Retrieve a proof bundle from IPFS by CID.
 *
 * @param {string} cidStr CID returned by storeProof()
 * @returns {Promise<object>} The proof bundle
 */
async function fetchProof(cidStr) {
  const { CID } = require("multiformats/cid");
  const node = await getNode();
  const j    = heliaJson(node);
  const cid  = CID.parse(cidStr);
  return j.get(cid);
}

/**
 * Gracefully shut down the Helia node.
 * Call on server shutdown (SIGTERM / SIGINT).
 */
async function stopNode() {
  if (_node) {
    await _node.stop();
    _node = null;
    console.log("Cipher IPFS: Helia node stopped");
  }
}

module.exports = { storeProof, fetchProof, stopNode };
