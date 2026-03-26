# Cipher — Deep Setup Guide

For licensed users, integrators, and developers deploying the full Cipher stack.

---

## Who this is for

This guide walks through the complete setup: Python ZK prover, Node.js API server, Lit Protocol key management, FVM smart contracts, and ZK circuit trusted setup. If you only need the frontend, the `README.md` installation steps are enough. Come here when you want to run proofs end-to-end.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Node.js | ≥ 18 | 
| Python | ≥ 3.10 | 
| Redis | any | `brew install redis` / `apt install redis` |
| Lurk | v0.5.0 | see [Lurk setup]([#5-lurk-binary) ](https://github.com/argumentcomputer/yatima)|
| MetaMask | any | browser extension |

You will also need:
- A **CIPHER license key** (`CIPHER-XXXX-XXXX-XXXX`) — obtained via the in-app purchase flow or from the Cipher team
- A **FVM wallet** funded with tFIL — [faucet.calibration.fildev.network](https://faucet.calibration.fildev.network)
- A **Lit Protocol account** — [developer.litprotocol.com](https://developer.litprotocol.com) (free tier is sufficient for `naga-dev`)

---

## 1. Clone and install dependencies

```bash
git clone https://github.com/web3chima/cipher.
cd cipher.
```

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Node — frontend

```bash
npm install
```

### Node — backend server

```bash
cd server
npm install
cd ..
```

---

## 2. Environment configuration

```bash
cp server/.env.example server/.env
```

Open `server/.env`. The sections below explain what each block does.

### Server

```env
PORT=3001
FRONTEND_URL=http://localhost:5173
```

Leave as-is for local development.

### FVM / Filecoin

```env
NETWORK=calibration
FVM_CALIBRATION_RPC=https://api.calibration.node.glif.io/rpc/v1
CIPHER_REGISTRY_ADDRESS=0xEdd7aa9943cdb9335EeE530DE4523E0a5221C1C9
```

Both contract addresses are pre-deployed on Calibration testnet — leave them as-is.

### Device identity

```env
CIPHER_DEVICE_ID=robot-001
```

This value is embedded as a ZK public signal in every proof. Use a stable, unique identifier per physical device.

### Lit Protocol

```env
LIT_KEY_MANAGER_ENABLED=false   # set true after PKP is minted
LIT_NETWORK=naga-dev
PKP_PUBLIC_KEY=
PKP_ETH_ADDRESS=
PKP_TOKEN_ID=
LIT_AUTH_ACTION_CID=
LIT_SUBMIT_ACTION_CID=QmbbHhvM66VhqKPuVcwL9EaiiEKBwyEew6oqYYz7J5r5C3
```

Leave `LIT_KEY_MANAGER_ENABLED=false` until you complete the Lit setup in step 6.

### License keys

```env
CIPHER_LICENSE_KEYS=CIPHER-XXXX-XXXX-XXXX

```

Paste your license key in `CIPHER_LICENSE_KEYS`. Multiple keys are comma-separated.

### Redis

```env
REDIS_URL=redis://localhost:6379
```

Start Redis before running the Lit worker: `redis-server`

---

## 3. Activate your license key

Start the server first:

```bash
npm run server
```

Then activate via the API:

```bash
curl -X POST http://localhost:3001/api/license/validate \
  -H "Content-Type: application/json" \
  -d '{"licenseKey": "CIPHER-XXXX-XXXX-XXXX"}'
```

A `200 { "valid": true }` response means the key is accepted. You can also activate through the UI — open the frontend, click **Solutions** in the header, and enter your key in the license activation form.

From this point, proof routes (`/api/proof/*`, `/api/keys/*`) require the `x-cipher-license` header:

```bash
-H "x-cipher-license: CIPHER-XXXX-XXXX-XXXX"
```

---

## 4. FVM wallet

You need a wallet funded with tFIL to submit proofs on-chain. The contracts are already deployed — you just need gas.

Fund your robot wallet at [faucet.calibration.fildev.network](https://faucet.calibration.fildev.network). 0.1 tFIL is enough for many test submissions.

Once Lit Protocol is configured (step 7), the PKP handles all on-chain signing — your wallet credentials stay local and are never exposed to the server.

---

## 5. Lurk binary

Lurk is the ZK circuit runtime. A shim is available for development without a full Rust build:

### Option A — shim (no Rust required, dev/test only)

```bash
# The shim implements prove/compress/compile/check with real Poseidon arithmetic
# It is pre-installed at ~/.local/bin/lurk if you are on the primary dev machine
lurk --version
```

If not present:

```bash
mkdir -p ~/.local/bin
# contact the Cipher team or see architecture.md for the shim source
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

### Option B — build from source (production)

Requires ~4 GB free disk and a Rust nightly toolchain:

```bash
git clone https://https://github.com/argumentcomputer/yatima
cd ~/lurk
rustup override set nightly-2024-11-13
cargo build --release --bin lurk
# binary will be at ~/lurk/target/release/lurk
```

---

## 6. ZK circuit trusted setup

This step generates the Groth16 proving and verification keys for the three Cipher circuits.

### Requirements

- `lurk` on PATH (step 5)
- `snarkjs` installed globally: `npm install -g snarkjs`
- `~5 GB` free disk (temporary during setup)

### Run setup

```bash
bash scripts/setup_circuits.sh
```

This script:
1. Compiles each `.lurk` circuit to R1CS
2. Runs a Powers of Tau ceremony via snarkjs
3. Generates `.zkey` proving keys
4. Extracts the verification key and patches `contracts/Groth16Verifier.sol`

Output keys land in `keys/`:

```
keys/
├── location_proof.zkey
├── fused_location_proof.zkey
├── lidar_scan_proof.zkey
└── *.zkey.enc.json   ← encrypted bundles (after Lit encryption step)
```

### Encrypt zkeys with Lit TPKE (production)

After minting your PKP (step 7), run the encryption step:

```bash
node scripts/encrypt_zkeys.js
```

This wraps each `.zkey` under Lit threshold encryption with an access control condition that checks `entriesForDevice(deviceId) > 0` on the FVM registry, then writes `*.zkey.enc.json` bundles to `keys/`.

---

## 7. Lit Protocol — PKP setup

Skip this section if `LIT_KEY_MANAGER_ENABLED=false` and you are testing locally.

### Deploy Lit Actions

```bash
node scripts/deploy_lit_actions.js
```

Copy the output CIDs into `server/.env`:

```env
LIT_AUTH_ACTION_CID=<output from above>
LIT_SUBMIT_ACTION_CID=<output from above>
```

### Mint PKP

```bash
node scripts/mint_pkp.js
```

Copy the output into `server/.env`:

```env
PKP_PUBLIC_KEY=<04...>
PKP_ETH_ADDRESS=<0x...>
PKP_TOKEN_ID=<decimal token id>
```

Then enable Lit key management:

```env
LIT_KEY_MANAGER_ENABLED=true
```

---

## 8. Start the full stack

Each service runs in its own terminal.

```bash
# Terminal 1 — Redis (required by Lit worker)
redis-server

# Terminal 2 — Lit auth server (session sig issuance, port 6380)
cd server && npm run lit:auth

# Terminal 3 — Lit background worker (PKP minting jobs)
cd server && npm run lit:worker

# Terminal 4 — Cipher API server (port 3001)
npm run server

# Terminal 5 — Frontend (port 5173)
npm run dev
```

---

## 9. End-to-end smoke test

With the stack running, verify the full proof path:

### 1. Check server health

```bash
curl http://localhost:3001/api/keys/status
# expect: { "litConnected": true, "circuits": ["location_proof", ...] }
```

### 2. Request session sigs

```bash
curl -X POST http://localhost:3001/api/keys/session \
  -H "Content-Type: application/json" \
  -H "x-cipher-license: CIPHER-XXXX-XXXX-XXXX" \
  -d '{"deviceId": "0xYOUR_DEVICE_ADDRESS"}'
# expect: { "sessionSigs": { ... } }
```

> Returns `403` if the device has no entries in the FVM registry yet. Submit one proof first, or use `_litApiMode` session for initial testing.

### 3. Fetch a zkey

```bash
curl -X POST http://localhost:3001/api/keys/zkey/location_proof \
  -H "Content-Type: application/json" \
  -H "x-cipher-license: CIPHER-XXXX-XXXX-XXXX" \
  -d '{"sessionSigs": <output from step 2>}'
# expect: { "circuit": "location_proof", "zkeyBase64": "..." }
```

### 4. Run Python tests

```bash
source .venv/bin/activate
python3 -m pytest tests/ -v
# expect: 83 passed
```

---

## Troubleshooting

**`403` on `/api/keys/session`**
The device has no entries in `CipherLocationRegistry`. Submit a proof on-chain first to register the device, then retry session sig issuance.

**`Lit not connected` in `/api/keys/status`**
The Lit node client failed to connect to `naga-dev`. Check `LIT_NETWORK` in `server/.env` and that you have internet access. Run `cd server && npm run lit:auth` and watch the logs.

**`pip install` fails on `scipy`**
Ensure you are on Python ≥ 3.10 and your venv is active. On Apple Silicon: `brew install openblas` then `OPENBLAS=$(brew --prefix openblas) pip install scipy`.

**`lurk: command not found`**
Add `~/.local/bin` to your PATH: `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc`

**`redis-server: command not found`**
Install Redis: `brew install redis` (macOS) or `sudo apt install redis-server` (Linux).

**FVM proof submission fails**
Ensure your robot wallet is funded with tFIL and that Lit Protocol is configured (step 7). When `LIT_KEY_MANAGER_ENABLED=true`, the PKP handles signing — no manual key wiring needed.
