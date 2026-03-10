/**
 * lit-actions/cipher_ceremony_action.js
 *
 * Lit Action — Multi-Party Trusted Setup Ceremony Contribution
 *
 * Each participant calls this action to generate TEE-attested entropy for
 * their snarkjs zkey contribution.  The action runs inside every Lit node's
 * Trusted Execution Environment (V8 isolate), so the entropy is generated
 * in hardware-attested compute and the PKP signature proves which TEE
 * instance produced it.
 *
 * Flow (per participant):
 *   1. TEE generates 64 bytes of random entropy via crypto.getRandomValues()
 *   2. Computes a commitment hash = keccak256("cipher-ceremony:<id>:<circuit>:<entropyHex>")
 *   3. Signs the commitment with the participant's PKP (threshold ECDSA)
 *   4. Returns { entropyHex, commitment, circuitName } to the coordinator
 *
 * The coordinator collects N participant responses then runs:
 *   snarkjs zkey contribute *_0.zkey *.zkey --name=<id> -e=<entropyHex>
 * for each participant in sequence, building a transcript of contributions.
 *
 * jsParams expected:
 *   participantId   {string}  — unique label for this contributor
 *   circuitName     {string}  — "location_proof" | "fused_location_proof" | "lidar_scan_proof"
 *   pkpPublicKey    {string}  — 0x-prefixed compressed public key of participant PKP
 */

const go = async () => {
  // ── 1. Generate TEE randomness ──────────────────────────────────────────────
  const randomBytes = new Uint8Array(64);
  crypto.getRandomValues(randomBytes);
  const entropyHex = Array.from(randomBytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

  // ── 2. Compute commitment hash ──────────────────────────────────────────────
  // Commitment binds participant identity, circuit, and entropy together.
  // This hash is what gets signed — the raw entropy is returned separately
  // so the coordinator can pass it to snarkjs.
  const commitmentInput = `cipher-ceremony:${participantId}:${circuitName}:${entropyHex}`;
  const commitment = ethers.utils.keccak256(
    ethers.utils.toUtf8Bytes(commitmentInput)
  );

  // ── 3. PKP sign the commitment ──────────────────────────────────────────────
  // The signature proves this entropy came from a Lit TEE running this exact
  // Lit Action (identified by its IPFS CID).
  await Lit.Actions.signEcdsa({
    toSign: ethers.utils.arrayify(commitment),
    publicKey: pkpPublicKey,
    sigName: "ceremonyCommitment",
  });

  // ── 4. Return entropy + commitment ─────────────────────────────────────────
  // entropyHex → passed to `snarkjs zkey contribute -e=<entropy>`
  // commitment → stored in ceremony transcript for auditability
  Lit.Actions.setResponse({
    response: JSON.stringify({
      participantId,
      circuitName,
      entropyHex,
      commitment,
    }),
  });
};

go();
