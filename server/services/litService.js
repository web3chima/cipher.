"use strict";
/**
 * services/litService.js
 *
 * Stub implementation of the Lit Protocol key manager service.
 * Satisfies the interface required by server/index.js:
 *   litService.isConnected() → boolean
 *   litService.connect()     → Promise<void>
 *   litService.disconnect()  → Promise<void>
 *
 * When LIT_KEY_MANAGER_ENABLED=true and Lit credentials are configured,
 * replace this stub with a full @lit-protocol/lit-node-client implementation.
 * Until then, /api/keys endpoints return 503.
 */

let _connected = false;

async function connect() {
  console.log("[lit] Lit Protocol not configured — running in stub mode");
  console.log("[lit] Set LIT_KEY_MANAGER_ENABLED=false to suppress this message");
}

async function disconnect() {
  _connected = false;
}

function isConnected() {
  return _connected;
}

module.exports = { isConnected, connect, disconnect };
