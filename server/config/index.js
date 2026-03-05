"use strict";

require("dotenv").config({ path: require("path").resolve(__dirname, "../.env") });

module.exports = Object.freeze({
  PORT:                    parseInt(process.env.PORT || "3001"),
  NETWORK:                 process.env.NETWORK || "calibration",
  CHAIN_ID:                parseInt(process.env.CHAIN_ID || "314159"),
  FVM_RPC:                 process.env.FVM_CALIBRATION_RPC || "https://api.calibration.node.glif.io/rpc/v1",
  LIT_ENABLED:             process.env.LIT_KEY_MANAGER_ENABLED !== "false",
  CIPHER_REGISTRY_ADDRESS: process.env.CIPHER_REGISTRY_ADDRESS || null,
  PKP_ETH_ADDRESS:         process.env.PKP_ETH_ADDRESS         || null,
  CIPHER_DEVICE_ID:        process.env.CIPHER_DEVICE_ID        || "unknown",
});
