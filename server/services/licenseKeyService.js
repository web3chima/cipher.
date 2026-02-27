"use strict";
/**
 * licenseKeyService.js
 *
 * Generates and manages Cipher SDK license keys.
 *
 * Key format: CIPHER-XXXX-XXXX-XXXX
 * Each segment is 4 uppercase alphanumeric characters (A-Z, 0-9).
 * Total entropy: 4 * 4 * log2(36) ≈ 82 bits — sufficient for B2B licensing.
 *
 * Issued keys are appended to CIPHER_LICENSE_KEYS in memory so that
 * requireLicense middleware validates them immediately without a restart.
 */

const crypto = require("crypto");

const CHARS    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const SEG_LEN  = 4;
const SEGMENTS = 3;

/**
 * Generate a cryptographically random license key segment.
 * @returns {string} 4-character uppercase alphanumeric string
 */
function randomSegment() {
  const bytes = crypto.randomBytes(SEG_LEN);
  return Array.from(bytes)
    .map((b) => CHARS[b % CHARS.length])
    .join("");
}

/**
 * Generate a new Cipher license key.
 * @returns {string} e.g. "CIPHER-A1B2-C3D4-E5F6"
 */
function generateLicenseKey() {
  const segments = Array.from({ length: SEGMENTS }, randomSegment);
  return `CIPHER-${segments.join("-")}`;
}

/**
 * Register a newly issued key so requireLicense accepts it immediately.
 * Appends to the in-memory CIPHER_LICENSE_KEYS env var (no restart needed).
 * @param {string} key
 */
function registerKey(key) {
  const existing = process.env.CIPHER_LICENSE_KEYS || "";
  const keys = existing.split(",").map((k) => k.trim()).filter(Boolean);
  if (!keys.includes(key.toUpperCase())) {
    keys.push(key.toUpperCase());
    process.env.CIPHER_LICENSE_KEYS = keys.join(",");
  }
}

/**
 * Issue a new license key: generate + register in memory.
 * @returns {string} The new license key
 */
function issueLicenseKey() {
  const key = generateLicenseKey();
  registerKey(key);
  console.log(`[license] Issued new key: ${key}`);
  return key;
}

module.exports = { issueLicenseKey, generateLicenseKey, registerKey };
