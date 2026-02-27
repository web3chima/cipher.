function getValidKeys() {
  const raw = process.env.CIPHER_LICENSE_KEYS || "";
  return raw
    .split(",")
    .map((k) => k.trim().toUpperCase())
    .filter(Boolean);
}

function requireLicense(req, res, next) {
  const validKeys = getValidKeys();

  if (validKeys.length === 0) {
    return res.status(403).json({ error: "License key not configured on server" });
  }

  const header = req.headers["x-cipher-license"];

  if (!header || typeof header !== "string" || !header.trim()) {
    return res.status(401).json({ error: "Valid license key required" });
  }

  if (!validKeys.includes(header.trim().toUpperCase())) {
    return res.status(401).json({ error: "Valid license key required" });
  }

  next();
}

module.exports = { requireLicense };
