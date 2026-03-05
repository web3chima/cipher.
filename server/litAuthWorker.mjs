/**
 * server/litAuthWorker.mjs
 *
 * Lit Auth Background Worker — BullMQ / Redis job processor
 *
 * Runs as a separate process alongside:
 *   server/index.js          (Express API, port 3001)
 *   server/litAuthServer.mjs (Lit Auth HTTP server, port 6380)
 *
 * Responsibilities:
 *   - Background PKP minting operations (non-blocking relative to API)
 *   - Job queue processing with BullMQ
 *   - Redis connection management
 *   - Error handling and retry logic for minting jobs
 *
 * Required env vars:
 *   REDIS_URL — Redis connection string (default: redis://localhost:6379)
 *
 * Usage:
 *   node server/litAuthWorker.mjs           # production
 *   node --watch server/litAuthWorker.mjs   # dev / hot-reload
 */

import { startAuthServiceWorker } from "@lit-protocol/auth-services";
import dotenv from "dotenv";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load .env from server/ directory
dotenv.config({ path: resolve(__dirname, ".env") });

const REDIS_URL = process.env.REDIS_URL || "redis://localhost:6379";

console.log("[litAuthWorker] Starting Lit Auth Service Worker...");
console.log(`  redis : ${REDIS_URL}`);

// Start the background worker for processing PKP minting operations
const worker = await startAuthServiceWorker();

console.log("[litAuthWorker] Worker running — listening for PKP minting jobs");

// ── Graceful shutdown ─────────────────────────────────────────────────────────
async function shutdown(signal) {
  console.log(`\n[litAuthWorker] ${signal} received — draining queue and shutting down...`);
  try {
    await worker?.close?.();
  } catch (err) {
    console.error("[litAuthWorker] Error during shutdown:", err.message);
  }
  process.exit(0);
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT",  () => shutdown("SIGINT"));
