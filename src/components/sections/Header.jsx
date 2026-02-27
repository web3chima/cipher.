import { useState, useEffect } from 'react';
import { ethers } from 'ethers';
import { Button } from '../ui/Button';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';
const TELEGRAM_BOT_USERNAME = import.meta.env.VITE_TELEGRAM_BOT_USERNAME || 'ciphersupport_bot';

// CipherLicensePayment contract — address set after Remix deployment
const PAYMENT_CONTRACT_ADDRESS = import.meta.env.VITE_CIPHER_LICENSE_PAYMENT_ADDRESS || '';
const LICENSE_PRICE_ATTOFIL    = import.meta.env.VITE_CIPHER_LICENSE_PRICE_ATTOFIL || '100000000000000000'; // 0.1 FIL default

const PAYMENT_ABI = [
  "function purchaseLicense(bytes32 commitment) external payable",
  "function licensePrice() external view returns (uint256)",
];

export function Header({ logoSrc }) {
  const [solutionsOpen, setSolutionsOpen] = useState(false);
  const [contactOpen, setContactOpen] = useState(false);
  const [contactForm, setContactForm] = useState({ enterpriseName: '', contactName: '', email: '' });
  const [contactStatus, setContactStatus] = useState('idle'); // idle, loading, success, error
  const [telegramLink, setTelegramLink] = useState('');

  const [licenseForm, setLicenseForm] = useState({ licenseKey: '', orgEmail: '' });
  const [licenseStatus, setLicenseStatus] = useState('idle'); // idle, loading, success, error
  const [licenseError, setLicenseError] = useState('');

  const handleContactSubmit = async (e) => {
    e.preventDefault();
    setContactStatus('loading');

    try {
      const response = await fetch(`${API_URL}/api/contact/request`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(contactForm),
      });

      if (!response.ok) throw new Error('Request failed');

      const data = await response.json();
      // Handle both inviteLink (MTProto group) and telegramLink (bot fallback)
      setTelegramLink(data.inviteLink || data.telegramLink);
      setContactStatus('success');
    } catch (error) {
      // Fallback: Direct Telegram link with pre-filled message
      const message = encodeURIComponent(
        `Hi! I'm ${contactForm.contactName} from ${contactForm.enterpriseName}.\n` +
        `Email: ${contactForm.email}\n\n` +
        `I'd like to connect with the Cipher team.`
      );
      setTelegramLink(`https://t.me/${TELEGRAM_BOT_USERNAME}?text=${message}`);
      setContactStatus('success');
    }
  };

  const resetContactForm = () => {
    setContactOpen(false);
    setContactForm({ enterpriseName: '', contactName: '', email: '' });
    setContactStatus('idle');
    setTelegramLink('');
  };

  const handleLicenseSubmit = async (e) => {
    e.preventDefault();
    setLicenseStatus('loading');
    setLicenseError('');

    try {
      const res = await fetch(`${API_URL}/api/license/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          licenseKey: licenseForm.licenseKey,
          orgEmail: licenseForm.orgEmail,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setLicenseError(data.error || 'Invalid license key');
        setLicenseStatus('error');
      } else {
        setLicenseStatus('success');
      }
    } catch {
      setLicenseError('Could not reach server. Please try again.');
      setLicenseStatus('error');
    }
  };

  const resetLicenseForm = () => {
    setLicenseForm({ licenseKey: '', orgEmail: '' });
    setLicenseStatus('idle');
    setLicenseError('');
  };

  // ── Payment flow state ────────────────────────────────────────────────────
  const [paymentOpen,   setPaymentOpen]   = useState(false);
  const [paymentStep,   setPaymentStep]   = useState('idle'); // idle | connecting | paying | verifying | done | error
  const [paymentError,  setPaymentError]  = useState('');
  const [issuedKey,     setIssuedKey]     = useState('');
  const [walletAddress, setWalletAddress] = useState('');

  useEffect(() => {
    const handler = () => setPaymentOpen(true);
    window.addEventListener('cipher:open-payment', handler);
    return () => window.removeEventListener('cipher:open-payment', handler);
  }, []);

  const resetPayment = () => {
    setPaymentOpen(false);
    setPaymentStep('idle');
    setPaymentError('');
    setIssuedKey('');
    setWalletAddress('');
  };

  const handlePurchaseLicense = async () => {
    if (!window.ethereum) {
      setPaymentError('MetaMask not detected. Please install MetaMask and add the FVM Calibration network.');
      setPaymentStep('error');
      return;
    }

    try {
      // Step 1 — connect wallet
      setPaymentStep('connecting');
      const provider = new ethers.BrowserProvider(window.ethereum);
      await provider.send('eth_requestAccounts', []);
      const signer  = await provider.getSigner();
      const address = await signer.getAddress();
      setWalletAddress(address);

      if (!PAYMENT_CONTRACT_ADDRESS) {
        setPaymentError('Payment contract not yet deployed. Please contact support.');
        setPaymentStep('error');
        return;
      }

      // Step 2 — generate shielded commitment
      // commitment = keccak256(abi.encodePacked(buyerAddress, nonce))
      const nonce      = ethers.hexlify(ethers.randomBytes(32));
      const commitment = ethers.keccak256(
        ethers.solidityPacked(['address', 'bytes32'], [address, nonce])
      );

      // Step 3 — send FIL to contract
      setPaymentStep('paying');
      const contract = new ethers.Contract(PAYMENT_CONTRACT_ADDRESS, PAYMENT_ABI, signer);
      const price    = BigInt(LICENSE_PRICE_ATTOFIL);
      const tx       = await contract.purchaseLicense(commitment, { value: price });
      await tx.wait();

      // Step 4 — verify on-chain + claim license key from server
      setPaymentStep('verifying');
      const res  = await fetch(`${API_URL}/api/license/purchase`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          txHash:       tx.hash,
          buyerAddress: address,
          commitment,
          nonce,
        }),
      });
      const data = await res.json();

      if (!res.ok) {
        setPaymentError(data.error || 'License issuance failed. Contact support with your tx hash.');
        setPaymentStep('error');
        return;
      }

      setIssuedKey(data.licenseKey);
      setPaymentStep('done');

    } catch (err) {
      const msg = err?.reason || err?.message || 'Transaction failed';
      setPaymentError(msg.includes('user rejected') ? 'Transaction cancelled.' : msg);
      setPaymentStep('error');
    }
  };

  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          {/* Logo */}
          <a href="#home" className="flex items-center gap-2">
            {logoSrc ? (
              <img src={logoSrc} alt="Cipher" className="h-10 w-10" />
            ) : (
              <div className="h-10 w-10 bg-[#485C11] rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">C</span>
              </div>
            )}
            <span className="font-semibold text-lg text-gray-900">Cipher</span>
          </a>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-1 bg-white/60 backdrop-blur-md rounded-full px-2 py-1.5 border border-gray-200/50">
            <a href="#product" className="px-4 py-2 text-sm text-gray-700 hover:text-gray-900 rounded-full hover:bg-white/80 transition-colors">
              Product
            </a>
            <button
              onClick={() => setSolutionsOpen(!solutionsOpen)}
              className="px-4 py-2 text-sm text-gray-700 hover:text-gray-900 rounded-full hover:bg-white/80 transition-colors flex items-center gap-1"
            >
              Solutions
              <svg
                className={`w-4 h-4 transition-transform ${solutionsOpen ? 'rotate-180' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <a href="#developers" className="px-4 py-2 text-sm text-gray-700 hover:text-gray-900 rounded-full hover:bg-white/80 transition-colors">
              Developers
            </a>
            <Button variant="primary" size="sm" pill onClick={() => setContactOpen(true)}>
              Contact
            </Button>
          </nav>

          {/* Mobile menu button */}
          <button className="md:hidden p-2 text-gray-700">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </header>

      {/* Solutions Tray */}
      {solutionsOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
            onClick={() => setSolutionsOpen(false)}
          />

          {/* Tray */}
          <div className="fixed top-20 left-0 right-0 z-50 px-6">
            <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden">
              <div className="p-8">
                <div className="grid md:grid-cols-2 gap-8">
                  {/* Left: B2B License Info */}
                  <div>
                    <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#f5f9ed] rounded-full mb-4">
                      <span className="w-2 h-2 bg-[#485C11] rounded-full" />
                      <span className="text-xs font-medium text-[#485C11]">B2B Enterprise</span>
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-3">
                      Enterprise License
                    </h3>
                    <p className="text-gray-600 mb-6">
                      Integrate Cipher's ZKP privacy stack into your robot products. Ship robots that protect consumer privacy by design.
                    </p>
                    <ul className="space-y-3 mb-6">
                      <li className="flex items-start gap-3">
                        <svg className="w-5 h-5 text-[#485C11] flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span className="text-sm text-gray-700">Pre-integrated ZKP stack for LiDAR & VSLAM</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <svg className="w-5 h-5 text-[#485C11] flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span className="text-sm text-gray-700">On-device proof generation (no cloud required)</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <svg className="w-5 h-5 text-[#485C11] flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span className="text-sm text-gray-700">Privacy compliance for consumer robotics</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <svg className="w-5 h-5 text-[#485C11] flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span className="text-sm text-gray-700">Volume licensing per robot unit</span>
                      </li>
                    </ul>
                    <Button
                      variant="primary"
                      size="lg"
                      pill
                      onClick={() => {
                        setSolutionsOpen(false);
                        setPaymentOpen(true);
                      }}
                    >
                      Request Enterprise License
                      <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                      </svg>
                    </Button>
                  </div>

                  {/* Right: License Key Form */}
                  <div className="bg-gray-50 rounded-xl p-6">
                    {licenseStatus === 'success' ? (
                      <div className="text-center py-2">
                        <div className="w-12 h-12 bg-[#485C11] rounded-full flex items-center justify-center mx-auto mb-4">
                          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        </div>
                        <h4 className="font-semibold text-gray-900 mb-1">License Activated</h4>
                        <p className="text-xs font-mono text-gray-500 mb-4">{licenseForm.licenseKey.toUpperCase()}</p>
                        <a
                          href="#developers"
                          onClick={() => setSolutionsOpen(false)}
                          className="inline-flex items-center gap-1 text-sm font-medium text-[#485C11] hover:underline"
                        >
                          Get started
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                          </svg>
                        </a>
                        <div className="mt-4">
                          <button onClick={resetLicenseForm} className="text-xs text-gray-400 hover:text-gray-600">
                            Activate another key
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <h4 className="font-semibold text-gray-900 mb-4">
                          Already have a license?
                        </h4>
                        <form onSubmit={handleLicenseSubmit} className="space-y-4">
                          <div>
                            <label htmlFor="license-key" className="block text-sm font-medium text-gray-700 mb-1">
                              License Key
                            </label>
                            <input
                              type="text"
                              id="license-key"
                              required
                              placeholder="CIPHER-XXXX-XXXX-XXXX"
                              value={licenseForm.licenseKey}
                              onChange={(e) => setLicenseForm({ ...licenseForm, licenseKey: e.target.value })}
                              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all font-mono text-sm"
                            />
                          </div>
                          <div>
                            <label htmlFor="org-email" className="block text-sm font-medium text-gray-700 mb-1">
                              Organization Email
                            </label>
                            <input
                              type="email"
                              id="org-email"
                              required
                              placeholder="team@company.com"
                              value={licenseForm.orgEmail}
                              onChange={(e) => setLicenseForm({ ...licenseForm, orgEmail: e.target.value })}
                              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all"
                            />
                          </div>
                          {licenseStatus === 'error' && (
                            <p className="text-sm text-red-600">{licenseError}</p>
                          )}
                          <Button
                            type="submit"
                            variant="secondary"
                            size="lg"
                            className="w-full"
                            disabled={licenseStatus === 'loading'}
                          >
                            {licenseStatus === 'loading' ? (
                              <>
                                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 inline" fill="none" viewBox="0 0 24 24">
                                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                </svg>
                                Activating...
                              </>
                            ) : (
                              'Activate License'
                            )}
                          </Button>
                        </form>
                        <p className="text-xs text-gray-500 mt-4 text-center">
                          Need help? <a href="#contact" className="text-[#485C11] hover:underline">Contact support</a>
                        </p>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Contact Modal */}
      {contactOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
            onClick={resetContactForm}
          />

          {/* Modal */}
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full overflow-hidden">
              {contactStatus === 'idle' || contactStatus === 'loading' ? (
                <>
                  {/* Header */}
                  <div className="px-6 pt-6 pb-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-xl font-bold text-gray-900">
                        Connect with Cipher
                      </h3>
                      <button
                        onClick={resetContactForm}
                        className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                    <p className="text-sm text-gray-600">
                      We'll create a dedicated Telegram group for your enterprise.
                    </p>
                  </div>

                  {/* Form */}
                  <form onSubmit={handleContactSubmit} className="px-6 pb-6">
                    <div className="space-y-4">
                      <div>
                        <label htmlFor="contact-enterprise" className="block text-sm font-medium text-gray-700 mb-1">
                          Enterprise Name *
                        </label>
                        <input
                          type="text"
                          id="contact-enterprise"
                          required
                          placeholder="Acme Robotics"
                          value={contactForm.enterpriseName}
                          onChange={(e) => setContactForm({ ...contactForm, enterpriseName: e.target.value })}
                          className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all"
                        />
                        <p className="text-xs text-gray-500 mt-1">
                          Group will be named: "{contactForm.enterpriseName || 'Enterprise'} {'<>'} Cipher"
                        </p>
                      </div>
                      <div>
                        <label htmlFor="contact-name" className="block text-sm font-medium text-gray-700 mb-1">
                          Your Name *
                        </label>
                        <input
                          type="text"
                          id="contact-name"
                          required
                          placeholder="John Smith"
                          value={contactForm.contactName}
                          onChange={(e) => setContactForm({ ...contactForm, contactName: e.target.value })}
                          className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all"
                        />
                      </div>
                      <div>
                        <label htmlFor="contact-email" className="block text-sm font-medium text-gray-700 mb-1">
                          Work Email *
                        </label>
                        <input
                          type="email"
                          id="contact-email"
                          required
                          placeholder="john@company.com"
                          value={contactForm.email}
                          onChange={(e) => setContactForm({ ...contactForm, email: e.target.value })}
                          className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all"
                        />
                      </div>
                    </div>

                    <Button
                      type="submit"
                      variant="primary"
                      size="lg"
                      className="w-full mt-6"
                      disabled={contactStatus === 'loading'}
                    >
                      {contactStatus === 'loading' ? (
                        <>
                          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          Connecting...
                        </>
                      ) : (
                        <>
                          Continue to Telegram
                          <svg className="w-4 h-4 ml-2" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.562 8.161c-.18 1.897-.962 6.502-1.359 8.627-.168.9-.5 1.201-.82 1.23-.697.064-1.226-.461-1.901-.903-1.056-.692-1.653-1.123-2.678-1.799-1.185-.781-.417-1.21.258-1.911.177-.184 3.247-2.977 3.307-3.23.007-.032.015-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.139-5.062 3.345-.479.329-.913.489-1.302.481-.428-.009-1.252-.242-1.865-.442-.751-.244-1.349-.374-1.297-.789.027-.216.324-.437.893-.663 3.498-1.524 5.831-2.529 6.998-3.015 3.333-1.386 4.025-1.627 4.477-1.635.099-.002.321.023.465.141.121.099.154.232.169.324.016.092.036.301.02.465z"/>
                          </svg>
                        </>
                      )}
                    </Button>
                  </form>
                </>
              ) : (
                /* Success State */
                <div className="p-6 text-center">
                  <div className="w-16 h-16 bg-[#485C11] rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.562 8.161c-.18 1.897-.962 6.502-1.359 8.627-.168.9-.5 1.201-.82 1.23-.697.064-1.226-.461-1.901-.903-1.056-.692-1.653-1.123-2.678-1.799-1.185-.781-.417-1.21.258-1.911.177-.184 3.247-2.977 3.307-3.23.007-.032.015-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.139-5.062 3.345-.479.329-.913.489-1.302.481-.428-.009-1.252-.242-1.865-.442-.751-.244-1.349-.374-1.297-.789.027-.216.324-.437.893-.663 3.498-1.524 5.831-2.529 6.998-3.015 3.333-1.386 4.025-1.627 4.477-1.635.099-.002.321.023.465.141.121.099.154.232.169.324.016.092.036.301.02.465z"/>
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">
                    Almost there!
                  </h3>
                  <p className="text-gray-600 mb-6">
                    Click below to open Telegram and connect with the Cipher team.<br />
                    <span className="font-medium">Group: {contactForm.enterpriseName} {'<>'} Cipher</span>
                  </p>

                  <a
                    href={telegramLink}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center justify-center gap-2 bg-[#0088cc] hover:bg-[#0077b5] text-white font-medium px-6 py-3 rounded-lg transition-colors w-full"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.562 8.161c-.18 1.897-.962 6.502-1.359 8.627-.168.9-.5 1.201-.82 1.23-.697.064-1.226-.461-1.901-.903-1.056-.692-1.653-1.123-2.678-1.799-1.185-.781-.417-1.21.258-1.911.177-.184 3.247-2.977 3.307-3.23.007-.032.015-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.139-5.062 3.345-.479.329-.913.489-1.302.481-.428-.009-1.252-.242-1.865-.442-.751-.244-1.349-.374-1.297-.789.027-.216.324-.437.893-.663 3.498-1.524 5.831-2.529 6.998-3.015 3.333-1.386 4.025-1.627 4.477-1.635.099-.002.321.023.465.141.121.099.154.232.169.324.016.092.036.301.02.465z"/>
                    </svg>
                    Open Telegram
                  </a>

                  <button
                    onClick={resetContactForm}
                    className="mt-4 text-sm text-gray-500 hover:text-gray-700"
                  >
                    Close
                  </button>
                </div>
              )}
            </div>
          </div>
        </>
      )}
      {/* Payment Modal */}
      {paymentOpen && (
        <>
          <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm" onClick={paymentStep === 'paying' || paymentStep === 'verifying' ? undefined : resetPayment} />
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full overflow-hidden">

              {/* Header */}
              <div className="px-6 pt-6 pb-4 flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-bold text-gray-900">Enterprise License</h3>
                  <p className="text-sm text-gray-500 mt-0.5">Shielded payment on Filecoin FVM</p>
                </div>
                {paymentStep !== 'paying' && paymentStep !== 'verifying' && (
                  <button onClick={resetPayment} className="p-1 text-gray-400 hover:text-gray-600">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>

              <div className="px-6 pb-6">
                {/* Step: idle */}
                {paymentStep === 'idle' && (
                  <div className="space-y-5">
                    <div className="bg-[#f5f9ed] rounded-xl p-4 flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600">License price</p>
                        <p className="text-2xl font-bold text-[#485C11]">0.1 FIL</p>
                      </div>
                      <div className="text-right text-xs text-gray-500 space-y-1">
                        <p>Full SDK access</p>
                        <p>Lifetime license</p>
                        <p>Shielded transaction</p>
                      </div>
                    </div>
                    <div className="space-y-2 text-sm text-gray-600">
                      <div className="flex items-center gap-2">
                        <svg className="w-4 h-4 text-[#485C11]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
                        <span>Commitment hash shields your identity on-chain</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <svg className="w-4 h-4 text-[#485C11]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>
                        <span>ZK proof of payment verified server-side</span>
                      </div>
                    </div>
                    <Button variant="primary" size="lg" className="w-full" onClick={handlePurchaseLicense}>
                      Connect Wallet & Pay
                      <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
                    </Button>
                    <p className="text-xs text-gray-400 text-center">Requires MetaMask with FVM Calibration network</p>
                  </div>
                )}

                {/* Step: connecting */}
                {paymentStep === 'connecting' && (
                  <div className="text-center py-6">
                    <svg className="animate-spin w-10 h-10 text-[#485C11] mx-auto mb-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    <p className="font-medium text-gray-900">Connecting wallet…</p>
                    <p className="text-sm text-gray-500 mt-1">Approve in MetaMask</p>
                  </div>
                )}

                {/* Step: paying */}
                {paymentStep === 'paying' && (
                  <div className="text-center py-6">
                    <svg className="animate-spin w-10 h-10 text-[#485C11] mx-auto mb-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    <p className="font-medium text-gray-900">Sending payment…</p>
                    <p className="text-sm text-gray-500 mt-1">Confirm the transaction in MetaMask</p>
                    <p className="text-xs font-mono text-gray-400 mt-2 truncate">{walletAddress}</p>
                  </div>
                )}

                {/* Step: verifying */}
                {paymentStep === 'verifying' && (
                  <div className="text-center py-6">
                    <svg className="animate-spin w-10 h-10 text-[#485C11] mx-auto mb-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    <p className="font-medium text-gray-900">Verifying ZK proof…</p>
                    <p className="text-sm text-gray-500 mt-1">Confirming shielded commitment on-chain</p>
                  </div>
                )}

                {/* Step: done */}
                {paymentStep === 'done' && (
                  <div className="text-center py-2">
                    <div className="w-14 h-14 bg-[#485C11] rounded-full flex items-center justify-center mx-auto mb-4">
                      <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <h4 className="text-lg font-bold text-gray-900 mb-1">License Issued</h4>
                    <p className="text-sm text-gray-500 mb-4">Save this key — it won't be shown again</p>
                    <div className="bg-gray-50 rounded-lg px-4 py-3 flex items-center justify-between gap-3 mb-5">
                      <span className="font-mono text-sm font-semibold text-[#485C11] tracking-wider">{issuedKey}</span>
                      <button
                        onClick={() => navigator.clipboard.writeText(issuedKey)}
                        className="text-gray-400 hover:text-[#485C11] flex-shrink-0"
                        title="Copy"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                      </button>
                    </div>
                    <a
                      href="#developers"
                      onClick={resetPayment}
                      className="inline-flex items-center gap-1 text-sm font-medium text-white bg-[#485C11] hover:bg-[#3d4e0e] px-5 py-2.5 rounded-xl transition-colors"
                    >
                      Get started
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
                    </a>
                  </div>
                )}

                {/* Step: error */}
                {paymentStep === 'error' && (
                  <div className="text-center py-2">
                    <div className="w-14 h-14 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <svg className="w-7 h-7 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </div>
                    <h4 className="font-semibold text-gray-900 mb-2">Payment Failed</h4>
                    <p className="text-sm text-red-600 mb-5">{paymentError}</p>
                    <Button variant="outline" size="md" onClick={() => setPaymentStep('idle')}>
                      Try Again
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}
