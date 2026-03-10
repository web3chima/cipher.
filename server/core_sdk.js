/**
 * server/core_sdk.js
 *
 * Lit Protocol Dev API client — ESM
 * Docs: https://api.dev.litprotocol.com  (Swagger UI)
 * Spec: https://api.dev.litprotocol.com/openapi.json
 *
 * Usage:
 *   import { createClient } from './core_sdk.js';
 *   const client = createClient('https://api.dev.litprotocol.com');
 *
 *   const { api_key, wallet_address } = await client.newAccount({
 *     accountName: 'cipher',
 *     accountDescription: 'cipher robot network',
 *   });
 */

const DEFAULT_BASE = 'https://api.dev.litprotocol.com';

export function createClient(baseUrl = DEFAULT_BASE) {
  const base = baseUrl.replace(/\/$/, '') + '/core/v1';

  // ── HTTP helpers ────────────────────────────────────────────────────────────

  async function req(method, path, { apiKey, body, query } = {}) {
    const url = new URL(`${base}${path}`);
    if (query) {
      for (const [k, v] of Object.entries(query)) {
        if (v !== undefined) url.searchParams.set(k, v);
      }
    }

    const headers = { 'Content-Type': 'application/json' };
    if (apiKey) headers['X-Api-Key'] = apiKey;

    const res = await fetch(url.toString(), {
      method,
      headers,
      redirect: 'follow',
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });

    const text = await res.text();
    if (!text) return null;

    try {
      return JSON.parse(text);
    } catch {
      throw new Error(`Lit API non-JSON response (${res.status}): ${text}`);
    }
  }

  const get  = (path, opts) => req('GET',  path, opts);
  const post = (path, opts) => req('POST', path, opts);

  // ── Account Management ──────────────────────────────────────────────────────

  /**
   * Create a new Lit account (no gas required).
   * @returns {{ api_key: string, wallet_address: string }}
   */
  function newAccount({ accountName, accountDescription, initialBalance } = {}) {
    return post('/new_account', {
      body: {
        account_name:        accountName,
        account_description: accountDescription,
        initial_balance:     initialBalance ?? null,
      },
    });
  }

  /** Verify an API key is still valid. @returns {boolean} */
  function accountExists(apiKey) {
    return get('/account_exists', { apiKey });
  }

  /**
   * Mint a new PKP wallet under this account (no tstLPX gas required).
   * @returns {{ wallet_address: string }}
   */
  function createWallet(apiKey) {
    return get('/create_wallet', { apiKey });
  }

  /**
   * List all PKP wallets on the account.
   * @returns {Array<{ id, name, description, wallet_address, public_key }>}
   */
  function listWallets(apiKey, { pageNumber = '1', pageSize = '50' } = {}) {
    return get('/list_wallets', {
      apiKey,
      query: { page_number: pageNumber, page_size: pageSize },
    });
  }

  /**
   * Create a usage (robot) API key — scoped with expiration + balance.
   * @param {{ apiKey, expiration, balance }} opts
   *   expiration — Unix timestamp string (e.g. now + 1 year)
   *   balance    — token allowance string (e.g. "1000000")
   * @returns {{ success: boolean, usage_api_key: string }}
   */
  function addUsageApiKey(apiKey, { expiration, balance }) {
    return post('/add_usage_api_key', {
      apiKey,
      body: { expiration, balance },
    });
  }

  /** Remove a usage API key. */
  function removeUsageApiKey(apiKey, usageApiKey) {
    return post('/remove_usage_api_key', {
      apiKey,
      body: { usage_api_key: usageApiKey },
    });
  }

  /**
   * List all usage API keys on the account.
   * @returns {Array<{ id, name, description, api_key, expiration, balance }>}
   */
  function listApiKeys(apiKey, { pageNumber = '1', pageSize = '50' } = {}) {
    return get('/list_api_keys', {
      apiKey,
      query: { page_number: pageNumber, page_size: pageSize },
    });
  }

  // ── Group Management (PKPs + Actions) ───────────────────────────────────────

  /**
   * Create a group binding PKPs to permitted Lit Actions.
   * @param {{ apiKey, groupName, groupDescription, permittedActions, pkps, allWalletsPermitted, allActionsPermitted }}
   * @returns {{ success: boolean }}
   */
  function addGroup(apiKey, {
    groupName, groupDescription,
    permittedActions = [], pkps = [],
    allWalletsPermitted = false, allActionsPermitted = false,
  }) {
    return post('/add_group', {
      apiKey,
      body: {
        group_name:            groupName,
        group_description:     groupDescription,
        permitted_actions:     permittedActions,
        pkps,
        all_wallets_permitted: allWalletsPermitted,
        all_actions_permitted: allActionsPermitted,
      },
    });
  }

  /** List groups on the account. */
  function listGroups(apiKey, { pageNumber = '1', pageSize = '50' } = {}) {
    return get('/list_groups', {
      apiKey,
      query: { page_number: pageNumber, page_size: pageSize },
    });
  }

  /** Update group metadata. */
  function updateGroup(apiKey, { groupId, name, description, allWalletsPermitted = false, allActionsPermitted = false }) {
    return post('/update_group', {
      apiKey,
      body: { group_id: groupId, name, description, all_wallets_permitted: allWalletsPermitted, all_actions_permitted: allActionsPermitted },
    });
  }

  /** Add a PKP wallet to a group (grants that PKP permission to run group actions). */
  function addPkpToGroup(apiKey, { groupId, pkpPublicKey }) {
    return post('/add_pkp_to_group', { apiKey, body: { group_id: groupId, pkp_public_key: pkpPublicKey } });
  }

  /** Remove a PKP wallet from a group. */
  function removePkpFromGroup(apiKey, { groupId, pkpPublicKey }) {
    return post('/remove_pkp_from_group', { apiKey, body: { group_id: groupId, pkp_public_key: pkpPublicKey } });
  }

  /** Add a Lit Action (by IPFS CID) to a group. */
  function addActionToGroup(apiKey, { groupId, actionIpfsCid, name, description }) {
    return post('/add_action_to_group', {
      apiKey,
      body: { group_id: groupId, action_ipfs_cid: actionIpfsCid, name, description },
    });
  }

  /** Remove a Lit Action from a group. */
  function removeActionFromGroup(apiKey, { groupId, actionIpfsCid }) {
    return post('/remove_action_from_group', { apiKey, body: { group_id: groupId, action_ipfs_cid: actionIpfsCid } });
  }

  /** List actions in a group. */
  function listActions(apiKey, { groupId, pageNumber = '1', pageSize = '50' }) {
    return get('/list_actions', {
      apiKey,
      query: { group_id: groupId, page_number: pageNumber, page_size: pageSize },
    });
  }

  /** List PKP wallets in a group. */
  function listWalletsInGroup(apiKey, { groupId, pageNumber = '1', pageSize = '50' }) {
    return get('/list_wallets_in_group', {
      apiKey,
      query: { group_id: groupId, page_number: pageNumber, page_size: pageSize },
    });
  }

  /** Update action metadata within a group. */
  function updateActionMetadata(apiKey, { groupId, actionIpfsCid, name, description }) {
    return post('/update_action_metadata', {
      apiKey,
      body: { group_id: groupId, action_ipfs_cid: actionIpfsCid, name, description },
    });
  }

  /** Update usage API key metadata. */
  function updateUsageApiKeyMetadata(apiKey, { usageApiKey, name, description }) {
    return post('/update_usage_api_key_metadata', {
      apiKey,
      body: { usage_api_key: usageApiKey, name, description },
    });
  }

  // ── Lit Actions ─────────────────────────────────────────────────────────────

  /**
   * Execute a Lit Action via the API (no session sigs needed).
   * @param {{ apiKey, code, jsParams }} opts
   *   code      — JS source string or IPFS CID
   *   jsParams  — parameters passed to the action
   * @returns {{ signatures, response, logs, has_error }}
   */
  function litAction(apiKey, { code, jsParams }) {
    return post('/lit_action', {
      apiKey,
      body: { code, js_params: jsParams ?? null },
    });
  }

  /**
   * Get the IPFS CID for a Lit Action source string (server-side upload).
   * @returns {string} IPFS CID
   */
  function getLitActionIpfsId(code) {
    return get(`/get_lit_action_ipfs_id/${encodeURIComponent(code)}`);
  }

  // ── Node Info ───────────────────────────────────────────────────────────────

  /** Get the chain config of the Lit node (chain ID, RPC, contract address). */
  function getNodeChainConfig() {
    return get('/get_node_chain_config');
  }

  // ── Return public API ───────────────────────────────────────────────────────

  return {
    // Account
    newAccount,
    accountExists,
    createWallet,
    listWallets,
    addUsageApiKey,
    removeUsageApiKey,
    listApiKeys,
    // Groups
    addGroup,
    listGroups,
    updateGroup,
    addPkpToGroup,
    removePkpFromGroup,
    addActionToGroup,
    removeActionFromGroup,
    listActions,
    listWalletsInGroup,
    updateActionMetadata,
    updateUsageApiKeyMetadata,
    // Actions
    litAction,
    getLitActionIpfsId,
    // Info
    getNodeChainConfig,
  };
}
