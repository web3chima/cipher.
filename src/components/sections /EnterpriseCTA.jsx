import { useState, useEffect } from 'react';
import { Button } from '../ui/Button';

export function EnterpriseCTA() {
  const [showForm, setShowForm] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  // Listen for hash change to open form directly
  useEffect(() => {
    const handleHashChange = () => {
      if (window.location.hash === '#enterprise-form') {
        setShowForm(true);
        // Scroll to section
        document.getElementById('enterprise')?.scrollIntoView({ behavior: 'smooth' });
      }
    };

    handleHashChange(); // Check on mount
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitted(true);
  };

  return (
    <section id="enterprise" className="px-6 py-20 bg-gray-900">
      <div className="max-w-6xl mx-auto">
        {!showForm ? (
          /* CTA View */
          <div className="text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#485C11]/20 rounded-full mb-6">
              <span className="w-2 h-2 bg-[#7a9c2e] rounded-full animate-pulse" />
              <span className="text-sm font-medium text-[#a8c76a]">For Robot Manufacturers</span>
            </div>

            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-6">
              Ship Robots with Privacy Built-In
            </h2>

            <p className="text-lg text-gray-400 max-w-2xl mx-auto mb-10">
              Integrate Cipher's ZKP stack into your robots. Your customers get privacy protection, you get a competitive advantage.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12">
              <Button
                variant="primary"
                size="lg"
                pill
                onClick={() => setShowForm(true)}
              >
                Request Enterprise License
                <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </Button>
              <a
                href="#developers"
                className="text-gray-400 hover:text-white transition-colors text-sm flex items-center gap-2"
              >
                View Developer Docs
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </a>
            </div>

            {/* Feature highlights */}
            <div className="grid md:grid-cols-3 gap-6 max-w-3xl mx-auto">
              <div className="text-center">
                <div className="w-12 h-12 bg-gray-800 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-[#7a9c2e]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-400">ZKP for LiDAR & VSLAM</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-gray-800 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-[#7a9c2e]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
                  </svg>
                </div>
                <p className="text-sm text-gray-400">On-device Processing</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-gray-800 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-[#7a9c2e]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-400">Volume Licensing</p>
              </div>
            </div>
          </div>
        ) : !submitted ? (
          /* Form View */
          <div className="max-w-xl mx-auto">
            <button
              onClick={() => setShowForm(false)}
              className="text-gray-400 hover:text-white transition-colors text-sm flex items-center gap-2 mb-8"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Back
            </button>

            <h3 className="text-2xl font-bold text-white mb-2">
              Request Enterprise License
            </h3>
            <p className="text-gray-400 mb-8">
              Tell us about your robot fleet and we'll get back to you within 24 hours.
            </p>

            <form onSubmit={handleSubmit} className="space-y-5">
              <div className="grid md:grid-cols-2 gap-5">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">
                    Full Name
                  </label>
                  <input
                    type="text"
                    id="name"
                    required
                    placeholder="John Smith"
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all"
                  />
                </div>
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                    Work Email
                  </label>
                  <input
                    type="email"
                    id="email"
                    required
                    placeholder="john@company.com"
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="company" className="block text-sm font-medium text-gray-300 mb-2">
                  Company Name
                </label>
                <input
                  type="text"
                  id="company"
                  required
                  placeholder="Acme Robotics"
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all"
                />
              </div>

              <div>
                <label htmlFor="fleet-size" className="block text-sm font-medium text-gray-300 mb-2">
                  Expected Fleet Size
                </label>
                <select
                  id="fleet-size"
                  required
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all"
                >
                  <option value="">Select fleet size</option>
                  <option value="1-100">1 - 100 robots</option>
                  <option value="100-1000">100 - 1,000 robots</option>
                  <option value="1000-10000">1,000 - 10,000 robots</option>
                  <option value="10000+">10,000+ robots</option>
                </select>
              </div>

              <div>
                <label htmlFor="use-case" className="block text-sm font-medium text-gray-300 mb-2">
                  Use Case (Optional)
                </label>
                <textarea
                  id="use-case"
                  rows={3}
                  placeholder="Tell us about your robots and how you plan to use Cipher..."
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-[#485C11] focus:border-[#485C11] outline-none transition-all resize-none"
                />
              </div>

              <Button type="submit" variant="primary" size="lg" className="w-full">
                Submit Request
              </Button>
            </form>
          </div>
        ) : (
          /* Success View */
          <div className="max-w-md mx-auto text-center">
            <div className="w-16 h-16 bg-[#485C11] rounded-full flex items-center justify-center mx-auto mb-6">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-white mb-3">
              Request Submitted
            </h3>
            <p className="text-gray-400 mb-8">
              Thanks for your interest in Cipher Enterprise. Our team will reach out within 24 hours.
            </p>
            <Button
              variant="secondary"
              size="lg"
              pill
              onClick={() => {
                setShowForm(false);
                setSubmitted(false);
              }}
            >
              Back to Home
            </Button>
          </div>
        )}
      </div>
    </section>
  );
}
