"use strict";
/**
 * server/services/litService.js
 *
 * Lit Protocol service — three operating modes:
 *
 *   STUB mode  (LIT_KEY_MANAGER_ENABLED=false)
 *     All methods throw / return 503. Server boots without any Lit deps.
 *
 *   API mode  (LIT_USAGE_API_KEY is set)
 *     Uses https://api.dev.litprotocol.com — no tstLPX gas required.
 *     connect()              → validates usage key via accountExists()
 *     getSessionSigs()       → runs cipher_auth_action.js inside Phala TEE
 *     decryptZkey()          → reads raw .zkey (auth already verified by action)
 *     executeSubmitAction()  → runs cipher_submit_action.js inside Phala TEE
 *
 *   NODE mode  (LIT_KEY_MANAGER_ENABLED=true, no LIT_USAGE_API_KEY)
 *     Full Lit threshold network: session sigs, TPKE decrypt, executeJs.
 *     Requires PKP_PUBLIC_KEY + tstLPX gas for PKP minting.
 */

const fs     = require("fs");
const path   = require("path");
const config = require("../config");

// ── State ──────────────────────────────────────────────────────────────────────
let _client    = null;   // LitNodeClient (node mode only)
let _apiClient = null;   // core_sdk client (API mode only)
let _connected = false;
let _apiMode   = false;

// Absolute path to server/lit-actions/
const LIT_ACTIONS_DIR = path.resolve(__dirname, "../lit-actions");
// Absolute path to cipher./keys/ (raw + enc zkeys)
const KEYS_DIR        = path.resolve(__dirname, "../../keys");

// ── Helpers ────────────────────────────────────────────────────────────────────

function _readAction(filename) {
  const p = path.join(LIT_ACTIONS_DIR, filename);
  if (!fs.existsSync(p)) {
    throw new Error(`Lit action not found: ${p}`);
  }
  return fs.readFileSync(p, "utf8");
}

function _requireClient() {
  if (!_connected) {
    throw new Error("Lit client not connected — call litService.connect() first");
  }
  if (_apiMode) return { api: _apiClient, key: config.LIT_USAGE_API_KEY };
  if (!_client)  throw new Error("Lit node client is null — check LIT_KEY_MANAGER_ENABLED");
  return _client;
}

// ── Lifecycle ──────────────────────────────────────────────────────────────────

async function connect() {
  if (_connected) return;

  if (!config.LIT_ENABLED) {
    console.log("[litService] LIT_KEY_MANAGER_ENABLED=false — running in stub mode");
    return;
  }

  // ── API mode ─────────────────────────────────────────────────────────────
  if (config.LIT_USAGE_API_KEY) {
    // Dynamic import so CJS server boots without ESM core_sdk when key absent
    const { createClient } = await import("../core_sdk.js");
    _apiClient = createClient(config.LIT_API_BASE);

    const exists = await _apiClient.accountExists(config.LIT_USAGE_API_KEY);
    if (!exists) {
      throw new Error(
        "[litService] LIT_USAGE_API_KEY is invalid — check server/.env"
      );
    }

    _apiMode   = true;
    _connected = true;
    console.log("[litService] Connected via Lit Dev API (API mode — no gas required)");
    return;
  }

  // ── Node mode ─────────────────────────────────────────────────────────────
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
  console.log(`[litService] Connected to Lit ${network} network (node mode)`);
}

async function disconnect() {
  if (!_connected) return;
  if (!_apiMode && _client) {
    await _client.disconnect();
    console.log("[litService] Disconnected from Lit network");
  }
  _connected = false;
  _apiMode   = false;
  _client    = null;
  _apiClient = null;
}

function isConnected() { return _connected; }
function isApiMode()   { return _apiMode;   }

// ── Session signatures ─────────────────────────────────────────────────────────

/**
 * API mode:  runs cipher_auth_action.js in Phala TEE — FVM guard checked.
 *            Returns a lightweight token (no threshold sigs needed).
 * Node mode: full threshold ECDSA session sigs via LitNodeClient.
 *
 * @param {string} deviceId   Robot ETH address (CipherLocationRegistry key)
 * @param {number} [ttlMs]    Session TTL ms (node mode only, default 1h)
 */
async function getSessionSigs(deviceId, ttlMs = 60 * 60 * 1000) {
  _requireClient();

  if (!config.PKP_PUBLIC_KEY) {
    throw new Error(
      "PKP_PUBLIC_KEY not set — run npm run lit:mint-pkp or set LIT_PKP_PUBLIC_KEY"
    );
  }

  // ── API mode ───────────────────────────────────────────────────────────────
  if (_apiMode) {
    if (!config.CIPHER_REGISTRY_ADDRESS) {
      throw new Error("CIPHER_REGISTRY_ADDRESS not set in .env");
    }

    const actionCode = _readAction("cipher_auth_action.js");
    const jsParams   = {
      signingMessage: `cipher-auth:${deviceId}:${Date.now()}`,
      deviceId,
      contractAddress: config.CIPHER_REGISTRY_ADDRESS,
      rpcUrl:          config.FVM_RPC,
      pkpPublicKey:    config.PKP_PUBLIC_KEY,
    };

    const result = await _apiClient.litAction(config.LIT_USAGE_API_KEY, {
      code:     actionCode,
      jsParams,
    });

    if (result.has_error) {
      throw new Error(`CipherAuth: action failed — ${result.logs}`);
    }

    // Guard: if signEcdsa was never called, the action's FVM check rejected the device
    if (!result.signatures?.cipherAuth) {
      throw new Error(
        `CipherAuth: device ${deviceId} not authorized — ` +
        `no registry entries or RPC unreachable. Logs: ${result.logs}`
      );
    }

    // Return an API-mode session token (not real threshold sigs)
    return {
      _litApiMode:  true,
      _usageApiKey: config.LIT_USAGE_API_KEY,
      deviceId,
      expiresAt:    Date.now() + ttlMs,
      authSig:      result.signatures?.cipherAuth ?? null,
    };
  }

  // ── Node mode ──────────────────────────────────────────────────────────────
  if (!config.LIT_AUTH_ACTION_CID) {
    throw new Error(
      "LIT_AUTH_ACTION_CID not set — run node scripts/deploy_lit_actions.js"
    );
  }
  if (!config.PKP_ETH_ADDRESS) {
    throw new Error("PKP_ETH_ADDRESS not set — run npm run lit:mint-pkp");
  }

  const { LitAbility, LitActionResource, LitEncryptionResource } =
    require("@lit-protocol/auth-helpers");
  const { utils: litUtils } = require("@lit-protocol/lit-client");
  const authMethodConfig    = require("../config/authMethodConfig");

  const authData = litUtils.generateAuthData({
    uniqueDappName:       authMethodConfig.uniqueDappName,
    uniqueAuthMethodType: authMethodConfig.bigint,
    userId:               deviceId,
  });

  return _client.getSessionSigs({
    pkpPublicKey: config.PKP_PUBLIC_KEY,
    chain:        "filecoin",
    expiration:   new Date(Date.now() + ttlMs).toISOString(),
    authMethods:  [authData],
    resourceAbilityRequests: [
      { resource: new LitEncryptionResource("*"), ability: LitAbility.DecryptionShare },
      { resource: new LitActionResource("*"),     ability: LitAbility.LitActionExecution },
    ],
  });
}

// ── Zkey decryption ────────────────────────────────────────────────────────────

/**
 * API mode:  auth was verified in getSessionSigs; reads raw .zkey from KEYS_DIR.
 * Node mode: Lit TPKE decrypt using the encrypted bundle from lit_setup.js.
 *
 * @param {object} encBundle   Contents of keys/<circuit>.zkey.enc.json
 * @param {object} sessionSigs Returned by getSessionSigs()
 * @returns {Promise<Uint8Array>} Raw zkey bytes
 */
async function decryptZkey(encBundle, sessionSigs) {
  _requireClient();

  // ── API mode ───────────────────────────────────────────────────────────────
  if (_apiMode || sessionSigs?._litApiMode) {
    // Auth already verified by cipher_auth_action.js in getSessionSigs.
    // In API/dev mode we serve the raw .zkey directly (no TPKE).
    const { circuit } = encBundle;
    if (!circuit) {
      throw new Error("encBundle.circuit is required in API mode");
    }

    const rawPath = path.join(KEYS_DIR, `${circuit}.zkey`);
    if (!fs.existsSync(rawPath)) {
      throw new Error(
        `Raw zkey not found at ${rawPath}.\n` +
        `Copy ${circuit}.zkey to cipher./keys/ or run lit_setup.js for encrypted bundles.`
      );
    }

    return new Uint8Array(fs.readFileSync(rawPath));
  }

  // ── Node mode ─────────────────────────────────────────────────────────────
  const { ciphertext, dataToEncryptHash, accessControlConditions } = encBundle;
  if (!ciphertext || !dataToEncryptHash || !accessControlConditions) {
    throw new Error(
      "encBundle must contain: { ciphertext, dataToEncryptHash, accessControlConditions }"
    );
  }

  const { decryptedData } = await _client.decrypt({
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
 * API mode:  runs cipher_submit_action.js in Phala TEE via litAction().
 * Node mode: executeJs() on the Lit node network.
 *
 * @param {object} params     jsParams for cipher_submit_action.js
 * @param {object} sessionSigs
 * @returns {Promise<{ sig, unsignedTx, txHash }>}
 */
async function executeSubmitAction(params, sessionSigs) {
  _requireClient();

  // ── API mode ───────────────────────────────────────────────────────────────
  if (_apiMode || sessionSigs?._litApiMode) {
    const actionCode = _readAction("cipher_submit_action.js");

    const result = await _apiClient.litAction(config.LIT_USAGE_API_KEY, {
      code:     actionCode,
      jsParams: {
        ...params,
        pkpPublicKey:  config.PKP_PUBLIC_KEY,
        pkpEthAddress: config.PKP_ETH_ADDRESS,
        rpcUrl:        config.FVM_RPC,
        chainId:       config.CHAIN_ID,
        contractAddress: config.CIPHER_REGISTRY_ADDRESS,
      },
    });

    if (result.has_error) {
      throw new Error(`CipherSubmit failed (Lit Dev API): ${result.logs}`);
    }
    if (!result.signatures?.fvmSubmitTx) {
      throw new Error(
        `Lit Action did not return fvmSubmitTx signature.\nLogs: ${result.logs}`
      );
    }

    const { sig }              = result.signatures.fvmSubmitTx;
    const { unsignedTx, txHash } = JSON.parse(result.response);
    return { sig, unsignedTx, txHash };
  }

  // ── Node mode ─────────────────────────────────────────────────────────────
  if (!config.LIT_SUBMIT_ACTION_CID) {
    throw new Error(
      "LIT_SUBMIT_ACTION_CID not set — run npm run lit:deploy-actions"
    );
  }

  const result = await _client.executeJs({
    ipfsId:      config.LIT_SUBMIT_ACTION_CID,
    sessionSigs,
    jsParams:    params,
  });

  if (!result.signatures?.fvmSubmitTx) {
    throw new Error(
      `Lit Action did not return fvmSubmitTx signature.\nLogs: ${JSON.stringify(result.logs)}`
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
  isApiMode,
  getSessionSigs,
  decryptZkey,
  executeSubmitAction,
};
