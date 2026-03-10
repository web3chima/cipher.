/**
 * lit-actions/cipher_submit_action.js
 *
 * Lit Action — FVM Proof Submission via PKP Threshold ECDSA
 *
 * This action is the only key-holder for the robot's FVM wallet.  No raw
 * private key ever exists anywhere; the PKP's key shares are distributed
 * across Lit nodes.  Signing only succeeds when all guards below pass.
 *
 * Guards (all must pass to produce a signature):
 *   1. locationHash is non-zero
 *   2. timestamp is non-zero
 *   3. ipfsCID is a valid bafy… v1 CID (proof bundle is on IPFS)
 *   4. deviceId is in the authorizedDeviceIds whitelist
 *
 * jsParams expected:
 *   a                  {string[2]}       Groth16 π_A (G1 point)
 *   b                  {string[2][2]}    Groth16 π_B (G2 point)
 *   c                  {string[2]}       Groth16 π_C (G1 point)
 *   publicSignals      {string[3]}       [locationHash, timestamp, deviceId]
 *   ipfsCID            {string}          IPFS CID of proof bundle
 *   contractAddress    {string}          CipherLocationRegistry address on FVM
 *   rpcUrl             {string}          FVM JSON-RPC endpoint
 *   chainId            {string|number}   314 (mainnet) or 314159 (calibration)
 *   pkpPublicKey       {string}          0x-prefixed compressed PKP public key
 *   pkpEthAddress      {string}          ETH address derived from PKP public key
 *   authorizedDeviceIds {string[]}       Whitelisted deviceId field elements
 */

const go = async () => {
  const [locationHash, timestamp, deviceId] = publicSignals;

  // ── Guard 1: non-zero locationHash ────────────────────────────────────────
  if (!locationHash || locationHash === "0") {
    throw new Error("CipherSubmit: locationHash is zero");
  }

  // ── Guard 2: non-zero timestamp ───────────────────────────────────────────
  if (!timestamp || timestamp === "0") {
    throw new Error("CipherSubmit: timestamp is zero");
  }

  // ── Guard 3: valid IPFS CID ───────────────────────────────────────────────
  if (!ipfsCID || !ipfsCID.startsWith("bafy")) {
    throw new Error(`CipherSubmit: invalid IPFS CID: ${ipfsCID}`);
  }

  // ── Guard 4: authorized device ────────────────────────────────────────────
  if (!authorizedDeviceIds || !authorizedDeviceIds.includes(deviceId)) {
    throw new Error(`CipherSubmit: deviceId ${deviceId} is not authorized`);
  }

  // ── Encode submitProof calldata ───────────────────────────────────────────
  const submitProofAbi = [
    "function submitProof(" +
      "uint256[2] calldata a," +
      "uint256[2][2] calldata b," +
      "uint256[2] calldata c," +
      "uint256[3] calldata publicSignals," +
      "string calldata ipfsCID" +
    ")",
  ];
  const iface = new ethers.utils.Interface(submitProofAbi);
  const calldata = iface.encodeFunctionData("submitProof", [
    a,
    b,
    c,
    publicSignals,
    ipfsCID,
  ]);

  // ── Fetch nonce and gas parameters from FVM ───────────────────────────────
  const provider = new ethers.providers.JsonRpcProvider(rpcUrl);

  const [nonce, gasPrice, estimatedGas] = await Promise.all([
    provider.getTransactionCount(pkpEthAddress),
    provider.getGasPrice(),
    provider.estimateGas({
      to:   contractAddress,
      data: calldata,
      from: pkpEthAddress,
    }),
  ]);

  // Add 20 % gas buffer for FVM variability
  const gasLimit = estimatedGas.mul(120).div(100);

  // ── Build unsigned transaction ────────────────────────────────────────────
  const unsignedTx = {
    to:       contractAddress,
    data:     calldata,
    nonce:    nonce,
    gasPrice: gasPrice,
    gasLimit: gasLimit,
    chainId:  parseInt(chainId),
  };

  // ── Hash the transaction for signing ─────────────────────────────────────
  const serialized = ethers.utils.serializeTransaction(unsignedTx);
  const txHash     = ethers.utils.keccak256(ethers.utils.arrayify(serialized));

  // ── PKP threshold ECDSA sign ──────────────────────────────────────────────
  // Each Lit node produces a partial signature; they are combined to form
  // a valid secp256k1 signature without any node holding the full key.
  await Lit.Actions.signEcdsa({
    toSign:    ethers.utils.arrayify(txHash),
    publicKey: pkpPublicKey,
    sigName:   "fvmSubmitTx",
  });

  // Return the unsigned tx so the caller can attach the signature and broadcast
  Lit.Actions.setResponse({
    response: JSON.stringify({
      unsignedTx,
      txHash,
    }),
  });
};

go();
