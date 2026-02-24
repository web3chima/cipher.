"use strict";
/**
 * routes/proof.js
 *
 * Express route: receive a compressed Groth16 proof from the robot,
 * store the proof bundle on IPFS, then submit it to the FVM
 * CipherLocationRegistry contract.
 *
 * POST /api/proof/submit
 *   Body: {
 *     deviceId      string   Robot identifier
 *     locationHash  string   Poseidon hash as decimal string (public signal)
 *     proof         object   Groth16 proof { pi_a, pi_b, pi_c }
 *     publicSignals string[] [locationHash, timestamp, deviceId]
 *     timestamp     number   Unix epoch (seconds)
 *   }
 *
 *   Response (success):
 *     { ipfsCID, txHash, entryIndex, locationHash }
 *
 *   Response (error):
 *     { error: string, details?: string }
 *
 * Mounted in server/index.js as:
 *   app.use("/api/proof", require("./routes/proof"));
 */

const express = require("express");
const router  = express.Router();

const { storeProof }       = require("../services/ipfsService");
const { submitProofToFVM } = require("../services/fvmService");

// ─── Input validation ────────────────────────────────────────────────────────

function validateProofBody(req, res, next) {
  const { deviceId, locationHash, proof, publicSignals, timestamp } = req.body;
  const errors = [];

  if (!deviceId || typeof deviceId !== "string")
    errors.push("deviceId is required (string)");

  if (!locationHash || typeof locationHash !== "string")
    errors.push("locationHash is required (decimal string)");

  if (!proof || typeof proof !== "object")
    errors.push("proof is required (Groth16 proof object)");
  else {
    if (!Array.isArray(proof.pi_a) || proof.pi_a.length < 2)
      errors.push("proof.pi_a must be an array [x, y, ...]");
    if (!Array.isArray(proof.pi_b) || proof.pi_b.length < 2)
      errors.push("proof.pi_b must be an array [[x0,x1],[y0,y1],...]");
    if (!Array.isArray(proof.pi_c) || proof.pi_c.length < 2)
      errors.push("proof.pi_c must be an array [x, y, ...]");
  }

  if (!Array.isArray(publicSignals) || publicSignals.length < 3)
    errors.push("publicSignals must be an array of at least 3 decimal strings");

  if (timestamp !== undefined && (typeof timestamp !== "number" || timestamp <= 0))
    errors.push("timestamp must be a positive number");

  if (errors.length > 0) {
    return res.status(400).json({ error: "Validation failed", details: errors });
  }

  next();
}

// ─── POST /api/proof/submit ──────────────────────────────────────────────────

router.post("/submit", validateProofBody, async (req, res) => {
  const { deviceId, locationHash, proof, publicSignals, timestamp } = req.body;

  let ipfsCID;

  // 1. Store proof bundle on IPFS
  try {
    ipfsCID = await storeProof({
      deviceId,
      locationHash,
      proof,
      publicSignals,
      timestamp: timestamp ?? Math.floor(Date.now() / 1000),
      protocol: "Groth16",
    });
  } catch (ipfsError) {
    console.error("IPFS store failed:", ipfsError.message);
    return res.status(502).json({
      error:   "IPFS upload failed",
      details: ipfsError.message,
    });
  }

  // 2. Submit proof to FVM CipherLocationRegistry
  try {
    const { txHash, entryIndex, locationHash: onChainHash } =
      await submitProofToFVM({
        proofJson: { proof, publicSignals },
        ipfsCID,
      });

    console.log(
      `Proof submitted — device: ${deviceId}, entryIndex: ${entryIndex}, ` +
      `CID: ${ipfsCID}, tx: ${txHash}`
    );

    return res.json({
      ipfsCID,
      txHash,
      entryIndex,
      locationHash: onChainHash,
    });

  } catch (fvmError) {
    console.error("FVM submission failed:", fvmError.message);
    // Proof is already on IPFS — return CID even if FVM fails
    return res.status(502).json({
      error:   "FVM submission failed",
      details: fvmError.message,
      ipfsCID, // client can retry submitProofToFVM with this CID
    });
  }
});

// ─── GET /api/proof/entry/:index ─────────────────────────────────────────────

const { getEntry } = require("../services/fvmService");

router.get("/entry/:index", async (req, res) => {
  const index = parseInt(req.params.index, 10);
  if (isNaN(index) || index < 0) {
    return res.status(400).json({ error: "index must be a non-negative integer" });
  }

  try {
    const entry = await getEntry(index);
    return res.json(entry);
  } catch (err) {
    return res.status(404).json({ error: err.message });
  }
});

module.exports = router;
