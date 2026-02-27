require("dotenv").config();

const express = require("express");
const cors = require("cors");
const contactRoutes = require("./routes/contact");
const proofRoutes   = require("./routes/proof");
const licenseRoutes = require("./routes/license");

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

app.use("/api/contact", contactRoutes);
app.use("/api/proof",   proofRoutes);
app.use("/api/license", licenseRoutes);

app.get("/health", (_req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

app.use((err, _req, res, _next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ error: "Internal server error" });
});

app.listen(PORT, () => {
  console.log(`Cipher server running on port ${PORT}`);
});
