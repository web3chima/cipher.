const { TelegramClient } = require("telegram");
const { StringSession } = require("telegram/sessions");

const apiId = parseInt(process.env.TELEGRAM_API_ID);
const apiHash = process.env.TELEGRAM_API_HASH;
const stringSession = new StringSession(process.env.TELEGRAM_STRING_SESSION || "");

let client = null;

async function getClient() {
  if (client && client.connected) return client;

  client = new TelegramClient(stringSession, apiId, apiHash, {
    connectionRetries: 5,
  });

  await client.connect();
  return client;
}

// When run directly (`npm run auth`): interactive one-time auth flow
// Generates the TELEGRAM_STRING_SESSION to save in .env
if (require.main === module) {
  require("dotenv").config({ path: require("path").resolve(__dirname, "../.env") });

  const input = require("input");

  (async () => {
    const authClient = new TelegramClient(
      new StringSession(""),
      parseInt(process.env.TELEGRAM_API_ID),
      process.env.TELEGRAM_API_HASH,
      { connectionRetries: 5 }
    );

    await authClient.start({
      phoneNumber: async () => await input.text("Phone number: "),
      password: async () => await input.text("2FA password (if set): "),
      phoneCode: async () => await input.text("Code from Telegram: "),
      onError: (err) => console.error("Auth error:", err),
    });

    console.log("\nSave this as TELEGRAM_STRING_SESSION in your .env file:\n");
    console.log(authClient.session.save());

    await authClient.disconnect();
    process.exit(0);
  })();
}

module.exports = { getClient };
