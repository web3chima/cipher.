require("dotenv").config();

const express = require("express");
const cors = require("cors");
const contactRoutes  = require("./routes/contact");
const proofRoutes    = require("./routes/proof");
const licenseRoutes  = require("./routes/license");
const keysRoutes     = require("./routes/keys");
const registryRoutes = require("./routes/registry");

const app = express();
const PORT = process.env.PORT || 3001;

app.use(
  cors({
    origin: process.env.FRONTEND_URL || "http://localhost:5173",
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type", "x-cipher-license"],
  })
);

app.use(express.json());

app.use("/api/contact",  contactRoutes);
app.use("/api/proof",    proofRoutes);
app.use("/api/license",  licenseRoutes);
app.use("/api/keys",     keysRoutes);
app.use("/api/registry", registryRoutes);

app.get("/health", (_req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

app.use((err, _req, res, _next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ error: "Internal server error" });
});

const server = app.listen(PORT, () => {
  console.log(`Cipher server running on port ${PORT}`);
});

// Graceful shutdown — stop Helia IPFS node on exit
const { stopNode } = require("./services/ipfsService");

async function shutdown(signal) {
  console.log(`[${signal}] Shutting down...`);
  server.close(async () => {
    await stopNode();
    process.exit(0);
  });
  setTimeout(() => process.exit(1), 5000).unref();
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT",  () => shutdown("SIGINT"));
