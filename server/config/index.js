"use strict";

require("dotenv").config({ path: require("path").resolve(__dirname, "../.env") });

module.exports = Object.freeze({
  PORT:                    parseInt(process.env.PORT || "3001"),
  NETWORK:                 process.env.NETWORK   || "calibration",
  CHAIN_ID:                parseInt(process.env.CHAIN_ID || "314159"),
  FVM_RPC:                 process.env.FVM_CALIBRATION_RPC || "https://api.calibration.node.glif.io/rpc/v1",

  // Lit Protocol — mode selection
  //   STUB: LIT_KEY_MANAGER_ENABLED=false  (default, no Lit)
  //   API:  LIT_USAGE_API_KEY set          (Lit Dev API via Phala TEE, no gas)
  //   NODE: LIT_KEY_MANAGER_ENABLED=true   (full threshold node network)
  LIT_ENABLED:             process.env.LIT_KEY_MANAGER_ENABLED !== "false",
  LIT_NETWORK:             process.env.LIT_NETWORK  || "naga-dev",

  // Lit Dev API (api.dev.litprotocol.com) — API mode, no tstLPX required
  LIT_API_BASE:            process.env.LIT_API_BASE       || "https://api.dev.litprotocol.com",
  LIT_ACCOUNT_API_KEY:     process.env.LIT_ACCOUNT_API_KEY || null,
  LIT_USAGE_API_KEY:       process.env.LIT_USAGE_API_KEY  || null,

  // PKP — set by mint_pkp.js (node mode) or createWallet (API mode)
  PKP_PUBLIC_KEY:          process.env.LIT_PKP_PUBLIC_KEY  || process.env.PKP_PUBLIC_KEY   || null,
  PKP_ETH_ADDRESS:         process.env.LIT_PKP_WALLET      || process.env.PKP_ETH_ADDRESS  || null,

  // Lit Action CIDs — set after: node scripts/deploy_lit_actions.js
  LIT_AUTH_ACTION_CID:     process.env.LIT_AUTH_ACTION_CID    || null,
  LIT_SUBMIT_ACTION_CID:   process.env.LIT_SUBMIT_ACTION_CID  || null,

  // Registry + device
  CIPHER_REGISTRY_ADDRESS: process.env.CIPHER_REGISTRY_ADDRESS || null,
  CIPHER_DEVICE_ID:        process.env.CIPHER_DEVICE_ID        || "unknown",
});
