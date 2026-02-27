const express  = require("express");
const router   = express.Router();
const { ethers } = require("ethers");
const { issueLicenseKey } = require("../services/licenseKeyService");

// CipherLicensePayment ABI — only the functions the server needs
const PAYMENT_ABI = [
  "function verifyCommitment(bytes32 commitment) external view returns (bool used, address buyer)",
  "function licensePrice() external view returns (uint256)",
];

function getPaymentContract() {
  const address = process.env.CIPHER_LICENSE_PAYMENT_ADDRESS;
  if (!address) return null;
  const rpc      = process.env.FVM_CALIBRATION_RPC || "https://api.calibration.node.glif.io/rpc/v1";
  const provider = new ethers.JsonRpcProvider(rpc);
  return new ethers.Contract(address, PAYMENT_ABI, provider);
}

const LICENSE_KEY_RE = /^CIPHER-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$/i;

function getValidKeys() {
  const raw = process.env.CIPHER_LICENSE_KEYS || "";
  return raw
    .split(",")
    .map((k) => k.trim().toUpperCase())
    .filter(Boolean);
}

// POST /api/license/validate
router.post("/validate", (req, res) => {
  const { licenseKey, orgEmail } = req.body;
  const errors = [];

  if (!licenseKey || typeof licenseKey !== "string" || !licenseKey.trim()) {
    errors.push("licenseKey is required");
  } else if (!LICENSE_KEY_RE.test(licenseKey.trim())) {
    errors.push("licenseKey format must be CIPHER-XXXX-XXXX-XXXX");
  }

  if (!orgEmail || typeof orgEmail !== "string" || !orgEmail.trim()) {
    errors.push("orgEmail is required");
  } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(orgEmail.trim())) {
    errors.push("orgEmail format is invalid");
  }

  if (errors.length > 0) {
    return res.status(400).json({ error: "Validation failed", details: errors });
  }

  const key = licenseKey.trim().toUpperCase();
  const validKeys = getValidKeys();

  if (validKeys.length === 0) {
    console.warn("[license] CIPHER_LICENSE_KEYS is not configured");
    return res.status(503).json({ error: "License validation not configured" });
  }

  if (!validKeys.includes(key)) {
    return res.status(401).json({ valid: false, error: "Invalid license key" });
  }

  console.log(`[license] Activated: ${key} for ${orgEmail.trim()}`);
  return res.json({
    valid: true,
    key,
    message: "License activated successfully",
  });
});

// POST /api/license/purchase
// Called after buyer sends FIL to CipherLicensePayment contract.
// Body: { txHash, buyerAddress, commitment, nonce }
// Server verifies the on-chain commitment then issues a license key.
router.post("/purchase", async (req, res) => {
  const { txHash, buyerAddress, commitment, nonce } = req.body;
  const errors = [];

  if (!txHash       || typeof txHash       !== "string") errors.push("txHash is required");
  if (!buyerAddress || typeof buyerAddress !== "string") errors.push("buyerAddress is required");
  if (!commitment   || typeof commitment   !== "string") errors.push("commitment is required");
  if (!nonce        || typeof nonce        !== "string") errors.push("nonce is required");

  if (errors.length > 0) {
    return res.status(400).json({ error: "Validation failed", details: errors });
  }

  // 1. Verify commitment = keccak256(buyerAddress, nonce) matches what was submitted on-chain
  const expectedCommitment = ethers.keccak256(
    ethers.solidityPacked(["address", "bytes32"], [buyerAddress, nonce])
  );

  if (expectedCommitment.toLowerCase() !== commitment.toLowerCase()) {
    return res.status(401).json({ error: "Commitment does not match buyer address and nonce" });
  }

  // 2. Verify on-chain: contract recorded this commitment from this buyer
  const contract = getPaymentContract();
  if (!contract) {
    return res.status(503).json({ error: "Payment contract not configured" });
  }

  try {
    const { used, buyer } = await contract.verifyCommitment(commitment);

    if (!used) {
      return res.status(402).json({ error: "Payment not found on-chain. Please complete the transaction first." });
    }

    if (buyer.toLowerCase() !== buyerAddress.toLowerCase()) {
      return res.status(401).json({ error: "Commitment buyer mismatch" });
    }
  } catch (err) {
    console.error("[license/purchase] contract read failed:", err.message);
    return res.status(502).json({ error: "Could not verify payment on-chain" });
  }

  // 3. Issue license key
  const key = issueLicenseKey();
  console.log(`[license] Purchased: ${key} by ${buyerAddress} tx=${txHash}`);

  return res.json({
    licenseKey: key,
    message: "License key issued successfully",
    txHash,
  });
});

module.exports = router;
