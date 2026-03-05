"use strict";
/**
 * server/config/authMethodConfig.js
 *
 * Cipher custom auth method configuration for Lit Protocol.
 *
 * The `bigint` is the custom auth method type ID registered with Lit Protocol.
 * It must be unique across all dApps.  Once set and PKPs are minted against it,
 * this value must never change — changing it would invalidate all existing PKPs.
 *
 * The `uniqueDappName` is a human-readable identifier used by generateAuthData
 * to derive the auth method ID for each user/device.
 *
 * Usage (both mint and auth sides must use identical values):
 *
 *   const authMethodConfig = require("../config/authMethodConfig");
 *
 *   // Cipher side — mint PKP for a robot device
 *   const authData = litUtils.generateAuthData({
 *     uniqueDappName:       authMethodConfig.uniqueDappName,
 *     uniqueAuthMethodType: authMethodConfig.bigint,
 *     userId: deviceId,
 *   });
 *   await litClient.mintWithCustomAuth({ account, authData, ... });
 *
 *   // Robot side — get session sigs
 *   const authData = litUtils.generateAuthData({
 *     uniqueDappName:       authMethodConfig.uniqueDappName,
 *     uniqueAuthMethodType: authMethodConfig.bigint,
 *     userId: deviceId,
 *   });
 *   await litClient.getSessionSigs({ authMethods: [authData], ... });
 */

module.exports = Object.freeze({
  // Human-readable dApp name — must be globally unique (used as auth method ID namespace)
  uniqueDappName: process.env.LIT_DAPP_NAME || "cipher-robot-network",

  // Numeric type ID for cipher's custom auth method.
  // Register once at https://developer.litprotocol.com or via litClient.registerCustomAuthMethod()
  // Default: 7 (arbitrary non-colliding value for development)
  bigint: BigInt(process.env.LIT_AUTH_METHOD_TYPE || "7"),
});
