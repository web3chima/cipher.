const defaultFeatures = [
  {
    icon: '⚡',
    title: 'Lightning Fast',
    description: 'Optimized for speed with instant page loads and smooth interactions.',
  },
  {
    icon: '🎨',
    title: 'Beautiful Design',
    description: 'Crafted with attention to detail for a premium user experience.',
  },
  {
    icon: '🔒',
    title: 'Secure by Default',
    description: 'Enterprise-grade security built into every layer of the platform.',
  },
];

export function Features({
  title = 'Why choose us',
  subtitle = 'Everything you need to succeed',
  features = defaultFeatures,
  columns = 3,
}) {
  const gridCols = {
    2: 'md:grid-cols-2',
    3: 'md:grid-cols-3',
    4: 'md:grid-cols-4',
  };

  return (
    <section className="px-6 py-20 bg-white">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">{title}</h2>
          <p className="text-lg text-gray-600">{subtitle}</p>
        </div>
        <div className={`grid gap-8 ${gridCols[columns]}`}>
          {features.map((feature, index) => (
            <div
              key={index}
              className="p-6 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors"
            >
              <div className="text-4xl mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
