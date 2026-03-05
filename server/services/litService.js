"use strict";
/**
 * server/services/litService.js
 *
 * Full Lit Protocol service — custom PKP authentication via cipher_auth_action.js.
 *
 * Auth flow (no private key on robot server):
 *   1. client.getSessionSigs() calls authNeededCallback
 *   2. callback builds SIWE message, then calls client.executeJs(cipher_auth_action.js)
 *   3. Lit nodes run the auth action — FVM guard checked, PKP threshold-signs SIWE
 *   4. Combined signature becomes the authSig for session sig issuance
 *
 * Exports:
 *   connect()              — call once at server startup
 *   disconnect()           — call on shutdown
 *   isConnected()          — boolean
 *   getSessionSigs(deviceId, ttlMs)   — custom PKP auth, returns sessionSigs dict
 *   decryptZkey(encBundle, sessionSigs) — Lit TPKE decrypt for a circuit bundle
 *   executeSubmitAction(params, sessionSigs) — run cipher_submit_action.js via executeJs
 */

// Lit Protocol deps are lazy-required inside connect() so the server boots
// cleanly when LIT_KEY_MANAGER_ENABLED=false (deps may not be installed yet).
const config = require("../config");

let _client    = null;
let _connected = false;

// ── Lifecycle ──────────────────────────────────────────────────────────────────

async function connect() {
  if (_connected) return;
  if (!config.LIT_ENABLED) {
    console.log("[litService] LIT_KEY_MANAGER_ENABLED=false — running in stub mode");
    return;
  }

  // Lazy-require so the server boots without Lit deps when disabled
  const { LitNodeClient } = require("@lit-protocol/lit-node-client");
  const { LitNetwork }    = require("@lit-protocol/constants");

  const network = config.LIT_NETWORK === "naga"
    ? LitNetwork.Datil
    : config.LIT_NETWORK === "naga-test"
      ? LitNetwork.DatilTest
      : LitNetwork.DatilDev;

  _client = new LitNodeClient({ litNetwork: network, debug: false });
  await _client.connect();
  _connected = true;
  console.log(`[litService] Connected to Lit ${network} network`);
}

async function disconnect() {
  if (!_connected || !_client) return;
  await _client.disconnect();
  _connected = false;
  console.log("[litService] Disconnected from Lit network");
}

function isConnected() {
  return _connected;
}

function _requireClient() {
  if (!_connected || !_client) {
    throw new Error("Lit client not connected — call litService.connect() first");
  }
  return _client;
}

// ── Session signatures — custom PKP auth via cipher_auth_action.js ─────────────

/**
 * Generate Lit session signatures using cipher_auth_action.js as the auth method.
 *
 * No private key is required on the server. The auth action validates the
 * device's FVM registry entries and threshold-signs the SIWE challenge.
 *
 * @param {string} deviceId  Robot ETH address (used as CipherLocationRegistry key)
 * @param {number} [ttlMs]   Session TTL in ms (default 1 hour)
 * @returns {Promise<object>} sessionSigs dict
 */
async function getSessionSigs(deviceId, ttlMs = 60 * 60 * 1000) {
  const client = _requireClient();

  if (!config.LIT_AUTH_ACTION_CID) {
    throw new Error(
      "LIT_AUTH_ACTION_CID not set in .env\n" +
      "Run: node scripts/deploy_lit_actions.js  to upload cipher_auth_action.js"
    );
  }
  if (!config.PKP_PUBLIC_KEY) {
    throw new Error("PKP_PUBLIC_KEY not set in .env — run npm run lit:mint-pkp");
  }
  if (!config.PKP_ETH_ADDRESS) {
    throw new Error("PKP_ETH_ADDRESS not set in .env — run npm run lit:mint-pkp");
  }

  const { LitAbility, LitActionResource, LitEncryptionResource } =
    require("@lit-protocol/auth-helpers");
  const { utils: litUtils } = require("@lit-protocol/lit-client");
  const authMethodConfig    = require("../config/authMethodConfig");

  // Generate the same authData that was used when the PKP was minted.
  // Lit nodes verify this against the PKP's permitted auth methods on-chain,
  // then run cipher_auth_action.js (validationIpfsCid) to check the FVM guard.
  const authData = litUtils.generateAuthData({
    uniqueDappName:       authMethodConfig.uniqueDappName,
    uniqueAuthMethodType: authMethodConfig.bigint,
    userId:               deviceId,
  });

  const expiration = new Date(Date.now() + ttlMs).toISOString();

  const sessionSigs = await client.getSessionSigs({
    pkpPublicKey: config.PKP_PUBLIC_KEY,
    chain:        "filecoin",
    expiration,
    authMethods:  [authData],
    resourceAbilityRequests: [
      {
        resource: new LitEncryptionResource("*"),
        ability:  LitAbility.DecryptionShare,
      },
      {
        resource: new LitActionResource("*"),
        ability:  LitAbility.LitActionExecution,
      },
    ],
  });

  return sessionSigs;
}

// ── Zkey decryption ────────────────────────────────────────────────────────────

/**
 * Decrypt an encrypted zkey bundle via Lit TPKE.
 *
 * ACC (baked into the bundle at lit_setup.js time):
 *   CipherLocationRegistry.entriesForDevice(address) > 0
 *
 * @param {object} encBundle  Contents of keys/<circuit>.zkey.enc.json
 * @param {object} sessionSigs
 * @returns {Promise<Uint8Array>} Decrypted zkey bytes
 */
async function decryptZkey(encBundle, sessionSigs) {
  const client = _requireClient();
  const { ciphertext, dataToEncryptHash, accessControlConditions } = encBundle;

  if (!ciphertext || !dataToEncryptHash || !accessControlConditions) {
    throw new Error(
      "encBundle must contain: { ciphertext, dataToEncryptHash, accessControlConditions }"
    );
  }

  const { decryptedData } = await client.decrypt({
    accessControlConditions,
    ciphertext,
    dataToEncryptHash,
    chain:       "filecoin",
    sessionSigs,
  });

  return decryptedData;
}

// ── Submit action ──────────────────────────────────────────────────────────────

/**
 * Execute cipher_submit_action.js via Lit executeJs.
 *
 * @param {object} params     jsParams for cipher_submit_action.js
 * @param {object} sessionSigs
 * @returns {Promise<{sig, unsignedTx, txHash}>}
 */
async function executeSubmitAction(params, sessionSigs) {
  const client = _requireClient();

  if (!config.LIT_SUBMIT_ACTION_CID) {
    throw new Error(
      "LIT_SUBMIT_ACTION_CID not set in .env — run npm run lit:deploy-actions"
    );
  }

  const result = await client.executeJs({
    ipfsId:      config.LIT_SUBMIT_ACTION_CID,
    sessionSigs,
    jsParams:    params,
  });

  if (!result.signatures?.fvmSubmitTx) {
    throw new Error(
      `Lit Action did not return fvmSubmitTx signature.\n` +
      `Logs: ${JSON.stringify(result.logs)}`
    );
  }

  const { sig }              = result.signatures.fvmSubmitTx;
  const { unsignedTx, txHash } = JSON.parse(result.response);

  return { sig, unsignedTx, txHash };
}

module.exports = {
  connect,
  disconnect,
  isConnected,
  getSessionSigs,
  decryptZkey,
  executeSubmitAction,
};
