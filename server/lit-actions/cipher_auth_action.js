/**
 * server/lit-actions/cipher_auth_action.js
 *
 * Cipher Custom PKP Authentication Lit Action
 *
 * This is the ONLY permitted auth method on every robot PKP.
 * Once pinned to IPFS the validation logic is immutable — cipher cannot
 * change it after PKPs are minted and bound to this CID.
 *
 * Flow:
 *   1. Robot calls litNodeClient.getSessionSigs() with this action as authMethod
 *   2. Each Lit node independently runs this action inside a TEE
 *   3. Guard checks CipherLocationRegistry on FVM — device must have ≥1 proof entry
 *   4. If guard passes each node produces a partial PKP signature over the SIWE challenge
 *   5. Combined threshold signature becomes the authSig → session sigs issued
 *   → No private key on robot server. Validation logic lives on IPFS, not on cipher servers.
 *
 * Guards (ALL must pass):
 *   1. deviceId is present and non-empty
 *   2. CipherLocationRegistry.entriesForDevice(deviceId) > 0  (on-chain, FVM)
 *   3. signingMessage is present (SIWE challenge built by caller)
 *
 * jsParams:
 *   signingMessage  {string}  SIWE message from createSiweMessageWithRecaps
 *   deviceId        {string}  Robot ETH address used as registry key
 *   contractAddress {string}  CipherLocationRegistry address on FVM
 *   rpcUrl          {string}  FVM JSON-RPC endpoint
 *   pkpPublicKey    {string}  0x-prefixed compressed PKP public key
 */

const go = async () => {
  // ── Guard 1: inputs present ────────────────────────────────────────────────
  if (!signingMessage) {
    throw new Error("CipherAuth: signingMessage is required");
  }
  if (!deviceId || deviceId === "0x0000000000000000000000000000000000000000") {
    throw new Error("CipherAuth: deviceId is missing or zero address");
  }

  // ── Guard 2: device has ≥1 proof entry in CipherLocationRegistry (FVM) ────
  // This is the immutable on-chain gate. A robot that has never submitted a
  // verified ZK location proof cannot authenticate.
  const provider = new ethers.providers.JsonRpcProvider(rpcUrl);
  const abi      = [
    "function entriesForDevice(address device) view returns (uint256)",
  ];
  const registry = new ethers.Contract(contractAddress, abi, provider);

  let entries;
  try {
    entries = await registry.entriesForDevice(deviceId);
  } catch (err) {
    throw new Error(
      `CipherAuth: registry call failed for device ${deviceId} — ${err.message}`
    );
  }

  if (!entries || entries.lte(0)) {
    throw new Error(
      `CipherAuth: device ${deviceId} has 0 registry entries — ` +
      `submit at least one verified ZK location proof before authenticating`
    );
  }

  // ── PKP threshold ECDSA sign the SIWE challenge ───────────────────────────
  // ethers.utils.hashMessage applies the standard Ethereum prefix:
  //   "\x19Ethereum Signed Message:\n" + message.length + message
  // This matches what generateAuthSig / verifyAuthSig expect on the client.
  const msgHash = ethers.utils.arrayify(
    ethers.utils.hashMessage(signingMessage)
  );

  await Lit.Actions.signEcdsa({
    toSign:    msgHash,
    publicKey: pkpPublicKey,
    sigName:   "cipherAuth",
  });
};

go();
