import { useState } from 'react';

const defaultFeatures = [
  {
    id: 1,
    title: 'Secure Data Processing',
    description: 'End-to-end encrypted data processing ensures your robot sensor data never leaves your secure environment.',
  },
  {
    id: 2,
    title: 'Real-time Monitoring',
    description: 'Monitor all robotic systems in real-time with comprehensive dashboards and alerting capabilities.',
  },
  {
    id: 3,
    title: 'Compliance Ready',
    description: 'Built-in compliance tools for GDPR, SOC2, and industry-specific regulations for robotic systems.',
  },
];

export function AccordionFeatures({
  sectionTitle = 'BUILT ON OPTIMIZING CORE',
  description = 'Our platform is designed to optimize security and privacy for robotic environments.',
  features = defaultFeatures,
}) {
  const [openId, setOpenId] = useState(null);

  const toggleFeature = (id) => {
    setOpenId(openId === id ? null : id);
  };

  return (
    <section className="px-6 py-20 bg-white">
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-2 gap-12 items-start">
          {/* Left side - Title */}
          <div>
            <p className="text-sm font-semibold text-[#485C11] uppercase tracking-wider mb-4">
              {sectionTitle}
            </p>
            <p className="text-lg text-gray-600">
              {description}
            </p>
          </div>

          {/* Right side - Accordion */}
          <div className="space-y-4">
            {features.map((feature) => (
              <div
                key={feature.id}
                className="bg-black/5 rounded-2xl overflow-hidden"
              >
                <button
                  onClick={() => toggleFeature(feature.id)}
                  className="w-full px-6 py-5 flex items-center justify-between text-left"
                >
                  <span className="font-medium text-gray-900">{feature.title}</span>
                  <span
                    className={`w-8 h-8 rounded-full bg-black/10 flex items-center justify-center transition-transform duration-200 ${
                      openId === feature.id ? 'rotate-45' : ''
                    }`}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                  </span>
                </button>
                {openId === feature.id && (
                  <div className="px-6 pb-5">
                    <p className="text-gray-600">{feature.description}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
