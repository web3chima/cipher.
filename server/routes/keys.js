"use strict";
/**
 * server/routes/keys.js
 *
 * Lit Protocol key management endpoints.
 *
 *   GET  /api/keys/status              — connection state + available circuits
 *   GET  /api/keys/pkp-info/:deviceId  — PKP public key + eth address for a device
 *   POST /api/keys/session             — generate session sigs via custom PKP auth
 *   POST /api/keys/zkey/:circuit       — decrypt proving key via Lit TPKE
 */

const express          = require("express");
const router           = express.Router();
const fs               = require("fs");
const path             = require("path");
const config           = require("../config");
const authMethodConfig = require("../config/authMethodConfig");
const litService       = require("../services/litService");

const KEYS_DIR   = path.resolve(__dirname, "../../keys");
const CIRCUITS   = ["location_proof", "fused_location_proof", "lidar_scan_proof"];

// ── Helpers ───────────────────────────────────────────────────────────────────

function requireLit(_req, res, next) {
  if (!litService.isConnected()) {
    return res.status(503).json({
      error: "Lit Protocol not connected",
      hint:  "Set LIT_KEY_MANAGER_ENABLED=true and configure Lit credentials, then restart",
    });
  }
  next();
}

function availableCircuits() {
  return CIRCUITS.filter((c) => {
    const enc = path.join(KEYS_DIR, `${c}.zkey.enc.json`);
    return fs.existsSync(enc);
  });
}

// ── GET /api/keys/status ──────────────────────────────────────────────────────

router.get("/status", (_req, res) => {
  res.json({
    litEnabled:        config.LIT_ENABLED,
    litConnected:      litService.isConnected(),
    litNetwork:        config.LIT_NETWORK,
    pkpAddress:        config.PKP_ETH_ADDRESS  || null,
    authActionCid:     config.LIT_AUTH_ACTION_CID   || null,
    submitActionCid:   config.LIT_SUBMIT_ACTION_CID || null,
    circuits:          availableCircuits(),
  });
});

// ── GET /api/keys/pkp-info/:deviceId ─────────────────────────────────────────
// Robot user walkthrough step 1:
//   Robot calls this to discover its pre-minted PKP credentials.
//   In this single-PKP deployment all devices share one PKP.
//   Extend with a device→PKP registry for multi-device fleets.

router.get("/pkp-info/:deviceId", (req, res) => {
  const { deviceId } = req.params;

  if (!deviceId || !/^0x[0-9a-fA-F]{40}$/.test(deviceId)) {
    return res.status(400).json({ error: "deviceId must be a 0x-prefixed ETH address" });
  }

  if (!config.PKP_PUBLIC_KEY || !config.PKP_ETH_ADDRESS) {
    return res.status(503).json({
      error: "PKP not configured",
      hint:  "Run npm run lit:mint-pkp and set PKP_PUBLIC_KEY + PKP_ETH_ADDRESS in .env",
    });
  }

  // authContext is everything the robot needs to reproduce generateAuthData()
  // client-side and call litClient.getSessionSigs({ authMethods: [authData] })
  res.json({
    deviceId,
    pkpPublicKey:    config.PKP_PUBLIC_KEY,
    pkpEthAddress:   config.PKP_ETH_ADDRESS,
    authActionCid:   config.LIT_AUTH_ACTION_CID || null,
    litNetwork:      config.LIT_NETWORK,
    authContext: {
      uniqueDappName:       authMethodConfig.uniqueDappName,
      uniqueAuthMethodType: authMethodConfig.bigint.toString(),
      userId:               deviceId,
    },
  });
});

// ── POST /api/keys/session ────────────────────────────────────────────────────
// Robot user walkthrough step 2:
//   Robot submits its deviceId → server runs cipher_auth_action.js via Lit →
//   FVM guard checked → PKP threshold-signs SIWE → session sigs returned.
//   No private key required on the robot or this server.
//
// Body: { deviceId: "0x...", ttlMs?: number }

router.post("/session", requireLit, async (req, res) => {
  const { deviceId, ttlMs } = req.body;

  if (!deviceId || !/^0x[0-9a-fA-F]{40}$/.test(deviceId)) {
    return res.status(400).json({ error: "deviceId must be a 0x-prefixed ETH address" });
  }

  try {
    const sessionSigs = await litService.getSessionSigs(
      deviceId,
      ttlMs ? parseInt(ttlMs, 10) : undefined
    );
    res.json({ sessionSigs });
  } catch (err) {
    const isAuthFail = err.message?.includes("CipherAuth:");
    res.status(isAuthFail ? 403 : 502).json({ error: err.message });
  }
});

// ── POST /api/keys/zkey/:circuit ──────────────────────────────────────────────
// Body: { sessionSigs: object }
// Returns: { circuit, zkeyBase64 }

router.post("/zkey/:circuit", requireLit, async (req, res) => {
  const { circuit }    = req.params;
  const { sessionSigs } = req.body;

  if (!circuit || !/^[\w-]+$/.test(circuit)) {
    return res.status(400).json({ error: "Invalid circuit name" });
  }
  if (!sessionSigs || typeof sessionSigs !== "object") {
    return res.status(400).json({ error: "sessionSigs object is required in request body" });
  }

  const bundlePath = path.join(KEYS_DIR, `${circuit}.zkey.enc.json`);
  if (!fs.existsSync(bundlePath)) {
    return res.status(404).json({
      error:     `Encrypted bundle not found for circuit: ${circuit}`,
      available: availableCircuits(),
    });
  }

  let encBundle;
  try {
    encBundle = JSON.parse(fs.readFileSync(bundlePath, "utf8"));
  } catch (err) {
    return res.status(500).json({ error: `Failed to read enc bundle: ${err.message}` });
  }

  try {
    const zkeyBytes  = await litService.decryptZkey(encBundle, sessionSigs);
    const zkeyBase64 = Buffer.from(zkeyBytes).toString("base64");
    res.json({ circuit, zkeyBase64 });
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

module.exports = router;
