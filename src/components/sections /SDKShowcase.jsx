import { Button } from '../ui/Button';

export function SDKShowcase({
  title = 'Build ROBOTS with Cipher SDK',
  description = 'Integrate security and privacy into your robotic applications with just a few lines of code.',
  explainerImage = '/cipher-explainer.png',
}) {
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
          {/* License Key Request */}
          <div className="bg-white rounded-2xl p-8 border border-gray-200 flex flex-col justify-center">
            <div className="mb-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Get Your License Key
              </h3>
              <p className="text-gray-600 text-sm">
                Request access to the Cipher SDK and start building secure robotic applications.
              </p>
            </div>
            <form className="space-y-4">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                  Email Address
                </label>
                <input
                  type="email"
                  id="email"
                  placeholder="you@company.com"
                  className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition-all"
                />
              </div>
              <div>
                <label htmlFor="company" className="block text-sm font-medium text-gray-700 mb-1">
                  Company Name
                </label>
                <input
                  type="text"
                  id="company"
                  placeholder="Your Company"
                  className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition-all"
                />
              </div>
              <Button variant="primary" size="lg" className="w-full">
                Request License Key
              </Button>
            </form>
          </div>

          {/* Cipher Explainer Image */}
          <div className="bg-gray-900 rounded-2xl overflow-hidden">
            <img
              src={explainerImage}
              alt="Cipher ZKP Architecture"
              className="w-full h-full object-cover"
            />
          </div>
        </div>
      </div>
    </section>
  );
}
