// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * CipherLicensePayment.sol
 *
 * Accepts FIL payments for Cipher SDK licenses.
 * Uses a commitment scheme for shielded transaction tracking:
 *
 *   commitment = keccak256(abi.encodePacked(buyerAddress, nonce))
 *
 * The nonce is generated client-side and never stored on-chain.
 * On-chain only the commitment hash is visible — not the nonce.
 * The server verifies the nonce off-chain to issue the license key.
 *
 * Deploy via Remix:
 *   constructor(_owner: your wallet address, _licensePrice: price in attoFIL)
 *   e.g. 100000000000000000 = 0.1 FIL
 *
 * After deployment, paste the contract address into server/.env:
 *   CIPHER_LICENSE_PAYMENT_ADDRESS=0x...
 */
contract CipherLicensePayment {

    // ─── State ────────────────────────────────────────────────────────────────

    address public owner;
    uint256 public licensePrice;   // in attoFIL (1 FIL = 1e18)
    uint256 public purchaseCount;

    /// commitment → buyer address (proves commitment was used by this buyer)
    mapping(bytes32 => address) public commitmentBuyer;

    // ─── Events ───────────────────────────────────────────────────────────────

    event LicensePurchased(
        address indexed buyer,
        bytes32 indexed commitment,
        uint256 purchaseIndex
    );

    event PriceUpdated(uint256 oldPrice, uint256 newPrice);

    // ─── Modifiers ────────────────────────────────────────────────────────────

    modifier onlyOwner() {
        require(msg.sender == owner, "CipherLicense: not owner");
        _;
    }

    // ─── Constructor ──────────────────────────────────────────────────────────

    /**
     * @param _owner        Your wallet address — receives withdrawn FIL
     * @param _licensePrice License price in attoFIL (e.g. 100000000000000000 = 0.1 FIL)
     */
    constructor(address _owner, uint256 _licensePrice) {
        require(_owner != address(0), "CipherLicense: zero owner");
        owner        = _owner;
        licensePrice = _licensePrice;
    }

    // ─── Purchase ─────────────────────────────────────────────────────────────

    /**
     * Purchase a Cipher SDK license.
     *
     * Client-side before calling:
     *   1. Generate random nonce (32 bytes)
     *   2. Compute commitment = keccak256(abi.encodePacked(msg.sender, nonce))
     *   3. Call this function with commitment + msg.value >= licensePrice
     *   4. Save txHash + nonce — needed to claim the license key from the server
     *
     * @param commitment keccak256(abi.encodePacked(buyer, nonce))
     */
    function purchaseLicense(bytes32 commitment) external payable {
        require(msg.value >= licensePrice,              "CipherLicense: insufficient payment");
        require(commitmentBuyer[commitment] == address(0), "CipherLicense: commitment already used");

        commitmentBuyer[commitment] = msg.sender;

        emit LicensePurchased(msg.sender, commitment, purchaseCount);
        purchaseCount++;

        // Refund any excess payment
        uint256 excess = msg.value - licensePrice;
        if (excess > 0) {
            payable(msg.sender).transfer(excess);
        }
    }

    // ─── Views ────────────────────────────────────────────────────────────────

    /**
     * Check if a commitment has been used and by whom.
     * Called by the server to verify a purchase before issuing a license key.
     */
    function verifyCommitment(bytes32 commitment)
        external
        view
        returns (bool used, address buyer)
    {
        buyer = commitmentBuyer[commitment];
        used  = buyer != address(0);
    }

    // ─── Owner ────────────────────────────────────────────────────────────────

    /** Withdraw accumulated FIL to owner wallet */
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "CipherLicense: nothing to withdraw");
        payable(owner).transfer(balance);
    }

    /** Update the license price */
    function setLicensePrice(uint256 newPrice) external onlyOwner {
        emit PriceUpdated(licensePrice, newPrice);
        licensePrice = newPrice;
    }

    /** Transfer contract ownership */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "CipherLicense: zero address");
        owner = newOwner;
    }

    /** View contract FIL balance */
    function getBalance() external view returns (uint256) {
        return address(this).balance;
    }
}
