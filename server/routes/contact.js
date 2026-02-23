const express = require("express");
const router = express.Router();
const { validateContact } = require("../middleware/validateContact");
const { createSupergroupWithInviteLink } = require("../services/telegramService");

router.post("/request", validateContact, async (req, res) => {
  const { enterpriseName, contactName, email } = req.body;

  try {
    // Primary: create supergroup via MTProto
    const { inviteLink, groupTitle, chatId } = await createSupergroupWithInviteLink(enterpriseName);
    console.log(`Created group "${groupTitle}" (${chatId}) for ${contactName} <${email}>`);
    return res.json({ inviteLink });
  } catch (primaryError) {
    console.error("MTProto group creation failed:", primaryError.message);

    // Fallback: return a direct bot link (frontend handles both shapes)
    const botUsername = process.env.TELEGRAM_BOT_USERNAME || "ciphersupport_bot";
    const message = encodeURIComponent(
      `Enterprise: ${enterpriseName}\nContact: ${contactName}\nEmail: ${email}`
    );
    return res.json({ telegramLink: `https://t.me/${botUsername}?text=${message}` });
  }
});

module.exports = router;
