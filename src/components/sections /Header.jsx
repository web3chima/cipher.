import { useState } from 'react';
import { Button } from '../ui/Button';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';
const TELEGRAM_BOT_USERNAME = import.meta.env.VITE_TELEGRAM_BOT_USERNAME || 'ciphersupport_bot';

export function Header({ logoSrc }) {
  const [solutionsOpen, setSolutionsOpen] = useState(false);
  const [contactOpen, setContactOpen] = useState(false);
  const [contactForm, setContactForm] = useState({ enterpriseName: '', contactName: '', email: '' });
  const [contactStatus, setContactStatus] = useState('idle'); // idle, loading, success, error
  const [telegramLink, setTelegramLink] = useState('');

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
                        document.getElementById('sdk-license')?.scrollIntoView({ behavior: 'smooth' });
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
                    <h4 className="font-semibold text-gray-900 mb-4">
                      Already have a license?
                    </h4>
                    <form className="space-y-4">
                      <div>
                        <label htmlFor="license-key" className="block text-sm font-medium text-gray-700 mb-1">
                          License Key
                        </label>
                        <input
                          type="text"
                          id="license-key"
                          placeholder="CIPHER-XXXX-XXXX-XXXX"
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
                          placeholder="team@company.com"
                          className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all"
                        />
                      </div>
                      <Button variant="secondary" size="lg" className="w-full">
                        Activate License
                      </Button>
                    </form>
                    <p className="text-xs text-gray-500 mt-4 text-center">
                      Need help? <a href="#contact" className="text-[#485C11] hover:underline">Contact support</a>
                    </p>
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
    </>
  );
}
