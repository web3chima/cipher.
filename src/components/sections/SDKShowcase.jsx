import { useState } from 'react';
import { Button } from '../ui/Button';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

export function SDKShowcase({
  title = 'Build ROBOTS with Cipher SDK',
  description = 'Integrate security and privacy into your robotic applications with just a few lines of code.',
  explainerImage = '/cipher-architecture.jpeg',
}) {
  const [form, setForm] = useState({ licenseKey: '', orgEmail: '' });
  const [status, setStatus] = useState('idle'); // idle | loading | success | error
  const [error, setError] = useState('');

  const handleActivate = async (e) => {
    e.preventDefault();
    setStatus('loading');
    setError('');
    try {
      const res = await fetch(`${API_URL}/api/license/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ licenseKey: form.licenseKey, orgEmail: form.orgEmail }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || 'Invalid license key');
        setStatus('error');
      } else {
        setStatus('success');
      }
    } catch {
      setError('Could not reach server. Please try again.');
      setStatus('error');
    }
  };

  return (
    <section id="sdk-license" className="px-6 py-20 bg-gray-50">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            {title}
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            {description}
          </p>
        </div>

        {/* License Key + Explainer */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* License Key Panel */}
          <div className="bg-white rounded-2xl p-8 border border-gray-200 flex flex-col justify-center">
            {status === 'success' ? (
              <div className="text-center">
                <div className="w-14 h-14 bg-[#485C11] rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h4 className="text-lg font-bold text-gray-900 mb-1">License Activated</h4>
                <p className="text-xs font-mono text-gray-500 mb-5">{form.licenseKey.toUpperCase()}</p>
                <a
                  href="#developers"
                  className="inline-flex items-center gap-1 text-sm font-medium text-white bg-[#485C11] hover:bg-[#3d4e0e] px-5 py-2.5 rounded-xl transition-colors"
                >
                  Get started
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </a>
                <div className="mt-4">
                  <button
                    onClick={() => { setForm({ licenseKey: '', orgEmail: '' }); setStatus('idle'); }}
                    className="text-xs text-gray-400 hover:text-gray-600"
                  >
                    Activate another key
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="mb-6">
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    Get Your License Key
                  </h3>
                  <p className="text-gray-600 text-sm">
                    Already have a key? Activate it below. Need one?{' '}
                    <button
                      type="button"
                      onClick={() => window.dispatchEvent(new CustomEvent('cipher:open-payment'))}
                      className="text-[#485C11] font-medium hover:underline"
                    >
                      Purchase a license →
                    </button>
                  </p>
                </div>
                <form onSubmit={handleActivate} className="space-y-4">
                  <div>
                    <label htmlFor="sdk-license-key" className="block text-sm font-medium text-gray-700 mb-1">
                      License Key
                    </label>
                    <input
                      type="text"
                      id="sdk-license-key"
                      required
                      placeholder="CIPHER-XXXX-XXXX-XXXX"
                      value={form.licenseKey}
                      onChange={(e) => setForm({ ...form, licenseKey: e.target.value })}
                      className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all font-mono text-sm"
                    />
                  </div>
                  <div>
                    <label htmlFor="sdk-org-email" className="block text-sm font-medium text-gray-700 mb-1">
                      Organization Email
                    </label>
                    <input
                      type="email"
                      id="sdk-org-email"
                      required
                      placeholder="team@company.com"
                      value={form.orgEmail}
                      onChange={(e) => setForm({ ...form, orgEmail: e.target.value })}
                      className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all"
                    />
                  </div>
                  {status === 'error' && (
                    <p className="text-sm text-red-600">{error}</p>
                  )}
                  <Button
                    type="submit"
                    variant="primary"
                    size="lg"
                    className="w-full"
                    disabled={status === 'loading'}
                  >
                    {status === 'loading' ? (
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
              </>
            )}
          </div>

          {/* Cipher Architecture Image */}
          <div className="bg-gray-900 rounded-2xl overflow-hidden relative min-h-[420px]">
            <img
              src={explainerImage}
              alt="Cipher Architecture"
              className="absolute inset-0 w-full h-full object-contain"
            />
          </div>
        </div>
      </div>
    </section>
  );
}
