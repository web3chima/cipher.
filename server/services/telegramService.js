const { Api } = require("telegram");
const { getClient } = require("../config/telegram");

async function createSupergroupWithInviteLink(enterpriseName) {
  const client = await getClient();

  const groupTitle = `Cipher – ${enterpriseName}`;

  // Create a supergroup (megagroup: true makes it a supergroup, not a broadcast channel)
  const createResult = await client.invoke(
    new Api.channels.CreateChannel({
      title: groupTitle,
      about: `Cipher support group for ${enterpriseName}`,
      megagroup: true,
    })
  );

  const channel = createResult.chats[0];
  const chatId = channel.id;

  const inputChannel = new Api.InputPeerChannel({
    channelId: channel.id,
    accessHash: channel.accessHash,
  });

  // Export a permanent invite link for the group
  const inviteResult = await client.invoke(
    new Api.messages.ExportChatInvite({
      peer: inputChannel,
    })
  );

  const inviteLink = inviteResult.link;

  // Add and promote the configured admin user if present
  const adminUserId = process.env.CIPHER_ADMIN_USER_ID;
  if (adminUserId) {
    try {
      await client.invoke(
        new Api.channels.InviteToChannel({
          channel: inputChannel,
          users: [adminUserId],
        })
      );

      await client.invoke(
        new Api.channels.EditAdmin({
          channel: inputChannel,
          userId: adminUserId,
          adminRights: new Api.ChatAdminRights({
            changeInfo: true,
            postMessages: true,
            editMessages: true,
            deleteMessages: true,
            banUsers: true,
            inviteUsers: true,
            pinMessages: true,
            addAdmins: false,
            anonymous: false,
            manageCall: true,
          }),
          rank: "Admin",
        })
      );
    } catch (adminError) {
      // Non-fatal: group is still created and invite link is valid
      console.warn(`Could not add admin user ${adminUserId}:`, adminError.message);
    }
  }

  return { inviteLink, groupTitle, chatId };
}

module.exports = { createSupergroupWithInviteLink };
