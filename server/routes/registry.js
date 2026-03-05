"use strict";
/**
 * routes/registry.js
 *
 * Read-only endpoints for the CipherLocationRegistry contract on FVM.
 *
 *   GET /api/registry/total                        — total entry count
 *   GET /api/registry/entries?offset=0&limit=20    — paginated entry list
 *   GET /api/registry/entry/:index                 — single entry by index
 *   GET /api/registry/device/:address              — all entries for a device
 *   GET /api/registry/device/:address/latest       — most recent entry for a device
 */

const express = require("express");
const router  = express.Router();

const {
  getEntry,
  getTotalEntries,
} = require("../services/fvmService");

// ─── GET /api/registry/total ──────────────────────────────────────────────────

router.get("/total", async (_req, res) => {
  try {
    const total = await getTotalEntries();
    return res.json({ total });
  } catch (err) {
    return res.status(502).json({ error: err.message });
  }
});

// ─── GET /api/registry/entries ────────────────────────────────────────────────

router.get("/entries", async (req, res) => {
  const offset = Math.max(0, parseInt(req.query.offset) || 0);
  const limit  = Math.min(100, Math.max(1, parseInt(req.query.limit) || 20));

  try {
    const total = await getTotalEntries();

    const indices = [];
    for (let i = offset; i < Math.min(offset + limit, total); i++) {
      indices.push(i);
    }

    const entries = await Promise.all(indices.map((i) => getEntry(i)));

    return res.json({ entries, total, offset, limit });
  } catch (err) {
    return res.status(502).json({ error: err.message });
  }
});

// ─── GET /api/registry/entry/:index ──────────────────────────────────────────

router.get("/entry/:index", async (req, res) => {
  const index = parseInt(req.params.index, 10);
  if (isNaN(index) || index < 0) {
    return res.status(400).json({ error: "index must be a non-negative integer" });
  }

  try {
    const entry = await getEntry(index);
    return res.json(entry);
  } catch (err) {
    return res.status(404).json({ error: err.message });
  }
});

// ─── GET /api/registry/device/:address ───────────────────────────────────────

router.get("/device/:address", async (req, res) => {
  const { address } = req.params;
  if (!/^0x[0-9a-fA-F]{40}$/.test(address)) {
    return res.status(400).json({ error: "address must be a valid 0x Ethereum address" });
  }

  try {
    const total   = await getTotalEntries();
    const entries = [];

    for (let i = 0; i < total; i++) {
      const entry = await getEntry(i);
      if (entry.device.toLowerCase() === address.toLowerCase()) {
        entries.push({ ...entry, index: i });
      }
    }

    return res.json({ device: address, count: entries.length, entries });
  } catch (err) {
    return res.status(502).json({ error: err.message });
  }
});

// ─── GET /api/registry/device/:address/latest ─────────────────────────────────

router.get("/device/:address/latest", async (req, res) => {
  const { address } = req.params;
  if (!/^0x[0-9a-fA-F]{40}$/.test(address)) {
    return res.status(400).json({ error: "address must be a valid 0x Ethereum address" });
  }

  try {
    const total = await getTotalEntries();
    let latest  = null;

    for (let i = 0; i < total; i++) {
      const entry = await getEntry(i);
      if (entry.device.toLowerCase() === address.toLowerCase()) {
        if (!latest || entry.timestamp > latest.timestamp) {
          latest = { ...entry, index: i };
        }
      }
    }

    if (!latest) {
      return res.status(404).json({ error: `No entries found for device ${address}` });
    }

    return res.json(latest);
  } catch (err) {
    return res.status(502).json({ error: err.message });
  }
});

module.exports = router;
