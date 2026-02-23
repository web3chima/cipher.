function validateContact(req, res, next) {
  const { enterpriseName, contactName, email } = req.body;
  const errors = [];

  if (!enterpriseName || typeof enterpriseName !== "string" || !enterpriseName.trim()) {
    errors.push("enterpriseName is required");
  } else if (enterpriseName.length > 100) {
    errors.push("enterpriseName must be 100 characters or fewer");
  }

  if (!contactName || typeof contactName !== "string" || !contactName.trim()) {
    errors.push("contactName is required");
  }

  if (!email || typeof email !== "string" || !email.trim()) {
    errors.push("email is required");
  } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    errors.push("email format is invalid");
  }

  if (errors.length > 0) {
    return res.status(400).json({ error: "Validation failed", details: errors });
  }

  req.body.enterpriseName = enterpriseName.trim();
  req.body.contactName = contactName.trim();
  req.body.email = email.trim();

  next();
}

module.exports = { validateContact };
