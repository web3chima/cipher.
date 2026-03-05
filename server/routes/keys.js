"use strict";
/**
 * routes/keys.js
 *
 * Lit Protocol key management endpoints.
 * All POST routes require an active Lit node connection (litService.isConnected()).
 * Returns 503 when Lit is not connected (e.g. running in stub mode).
 *
 *   GET  /api/keys/status          — connection state + available circuits
 *   POST /api/keys/session         — get Lit session sigs
 *   POST /api/keys/zkey/:circuit   — decrypt proving key via Lit TPKE
 */

const express    = require("express");
const router     = express.Router();
const config     = require("../config");
const litService = require("../services/litService");

// ─── GET /api/keys/status ─────────────────────────────────────────────────────

router.get("/status", (_req, res) => {
  res.json({
    litEnabled:   config.LIT_ENABLED,
    litConnected: litService.isConnected(),
    pkpAddress:   config.PKP_ETH_ADDRESS || null,
    circuits:     [],   // populated once full Lit integration is in place
  });
});

// ─── Lit-required guard ───────────────────────────────────────────────────────

function requireLit(_req, res, next) {
  if (!litService.isConnected()) {
    return res.status(503).json({
      error: "Lit Protocol not connected",
      hint:  "Configure Lit credentials and restart the server",
    });
  }
  next();
}

// ─── POST /api/keys/session ───────────────────────────────────────────────────

router.post("/session", requireLit, async (_req, res) => {
  // Full implementation: call litService.getSessionSigs({ capacityTokenId, pkpPublicKey })
  // Stubbed — requireLit guard ensures this is never reached in stub mode
  res.status(501).json({ error: "Not implemented" });
});

// ─── POST /api/keys/zkey/:circuit ─────────────────────────────────────────────

router.post("/zkey/:circuit", requireLit, async (req, res) => {
  const { circuit } = req.params;
  if (!circuit || !/^[\w-]+$/.test(circuit)) {
    return res.status(400).json({ error: "Invalid circuit name" });
  }
  // Full implementation: call litService.decryptZkey(circuit, sessionSigs)
  // Stubbed — requireLit guard ensures this is never reached in stub mode
  res.status(501).json({ error: "Not implemented" });
});

module.exports = router;
