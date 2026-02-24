import { Hero } from '../components/sections/Hero';
import { Features } from '../components/sections/Features';
import { CTA } from '../components/sections/CTA';

export function LandingPage() {
  return (
    <div className="min-h-screen">
      <Hero
        headline="Ship faster with confidence"
        subheadline="The all-in-one platform for modern development teams. Build, test, and deploy with ease."
        primaryCta="Start Free Trial"
        secondaryCta="See How It Works"
      />
      <Features
        title="Everything you need"
        subtitle="Powerful features to accelerate your workflow"
        features={[
          {
            icon: '🚀',
            title: 'Instant Deploy',
            description: 'Push to git and watch your changes go live in seconds.',
          },
          {
            icon: '🔄',
            title: 'Auto Scaling',
            description: 'Handle any traffic spike without breaking a sweat.',
          },
          {
            icon: '📊',
            title: 'Real-time Analytics',
            description: 'Monitor performance and user behavior as it happens.',
          },
        ]}
      />
      <CTA
        variant="gradient"
        headline="Ready to level up?"
        subheadline="Join 50,000+ developers who ship faster with our platform."
        buttonText="Get Started Free"
      />
    </div>
  );
}
