/**
 * server/litAuthServer.mjs
 *
 * Standalone Lit Auth Server — port 6380 (ESM)
 *
 * Runs alongside the main Express server (port 3001).
 * Handles Lit Protocol session-sig minting, PKP management, and
 * delegation requests via @lit-protocol/auth-services.
 *
 * Required env vars:
 *   LIT_TXSENDER_RPC_URL          — Lit relayer RPC (e.g. https://yellowstone-rpc.litprotocol.com)
 *   LIT_TXSENDER_PRIVATE_KEY      — 0x-prefixed key; funds relayer gas
 *   LIT_DELEGATION_ROOT_MNEMONIC  — BIP-39 mnemonic; root key for delegation tree
 *
 * Optional env vars:
 *   NETWORK               — Lit network: "naga-dev" | "naga-test" | "naga" (default: "naga-dev")
 *   PORT                  — port for this server (default: 6380)
 *   LOG_LEVEL             — "debug" | "info" | "warn" | "error" (default: "info")
 *   REDIS_URL             — Redis connection string (default: redis://localhost:6379)
 *   ENABLE_API_KEY_GATE   — "true" to require X-API-KEY header (default: "false")
 *   MAX_REQUESTS_PER_WINDOW — rate limit max requests (default: 10)
 *   WINDOW_MS             — rate limit window in ms (default: 10000)
 *   STYTCH_PUBLIC_TOKEN   — Stytch public token (for OTP services)
 *   STYTCH_SECRET         — Stytch secret key
 *
 * Usage:
 *   node server/litAuthServer.mjs           # production
 *   node --watch server/litAuthServer.mjs   # dev / hot-reload
 */

import { createLitAuthServer } from "@lit-protocol/auth-services";
import dotenv from "dotenv";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load .env from server/ directory
dotenv.config({ path: resolve(__dirname, ".env") });

// ── Required env validation ────────────────────────────────────────────────────
const REQUIRED = [
  "LIT_TXSENDER_RPC_URL",
  "LIT_TXSENDER_PRIVATE_KEY",
  "LIT_DELEGATION_ROOT_MNEMONIC",
];

const missing = REQUIRED.filter((k) => !process.env[k]);
if (missing.length > 0) {
  console.error(
    "[litAuthServer] Missing required env vars:\n" +
    missing.map((k) => `  ${k}`).join("\n") +
    "\n\nSee server/.env.example for documentation."
  );
  process.exit(1);
}

// ── Config ────────────────────────────────────────────────────────────────────
const PORT      = parseInt(process.env.LIT_AUTH_PORT             || "6380", 10);
const LOG_LEVEL = process.env.LOG_LEVEL                         || "info";
const REDIS_URL = process.env.REDIS_URL                         || "redis://localhost:6379";
const MAX_REQ   = parseInt(process.env.MAX_REQUESTS_PER_WINDOW  || "10", 10);
const WINDOW_MS = parseInt(process.env.WINDOW_MS                || "10000", 10);
const NETWORK   = process.env.LIT_AUTH_NETWORK                  || "naga-dev";
const API_GATE  = process.env.ENABLE_API_KEY_GATE               === "true";

console.log("[litAuthServer] Starting Lit Auth Server...");
console.log(`  network  : ${NETWORK}`);
console.log(`  port     : ${PORT}`);
console.log(`  logLevel : ${LOG_LEVEL}`);
console.log(`  redis    : ${REDIS_URL}`);
console.log(`  apiGate  : ${API_GATE}`);
console.log(`  rateLimit: ${MAX_REQ} req / ${WINDOW_MS}ms`);

// ── Start ─────────────────────────────────────────────────────────────────────
const litAuthServer = createLitAuthServer({
  port:                      PORT,
  network:                   NETWORK,
  logLevel:                  LOG_LEVEL,
  redisUrl:                  REDIS_URL,
  litTxsenderRpcUrl:         process.env.LIT_TXSENDER_RPC_URL,
  litTxsenderPrivateKey:     process.env.LIT_TXSENDER_PRIVATE_KEY,
  litDelegationRootMnemonic: process.env.LIT_DELEGATION_ROOT_MNEMONIC,
  enableApiKeyGate:          API_GATE,
  maxRequestsPerWindow:      MAX_REQ,
  windowMs:                  WINDOW_MS,
  ...(process.env.STYTCH_PUBLIC_TOKEN && {
    stytchPublicToken: process.env.STYTCH_PUBLIC_TOKEN,
    stytchSecret:      process.env.STYTCH_SECRET,
  }),
});

await litAuthServer.start();

console.log(`[litAuthServer] Lit Auth Server running on http://0.0.0.0:${PORT}`);

// ── Graceful shutdown ─────────────────────────────────────────────────────────
async function shutdown(signal) {
  console.log(`\n[litAuthServer] ${signal} received — shutting down...`);
  try {
    await litAuthServer.stop?.();
  } catch (err) {
    console.error("[litAuthServer] Error during shutdown:", err.message);
  }
  process.exit(0);
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT",  () => shutdown("SIGINT"));
