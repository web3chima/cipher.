import { Button } from '../ui/Button';

export function Hero({
  headline = 'Build something amazing',
  subheadline = 'Create beautiful, responsive websites with our powerful platform.',
  primaryCta = 'Get Started',
  secondaryCta = 'Learn More',
  onPrimaryClick,
  onSecondaryClick,
}) {
  return (
    <section className="px-6 py-24 bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-4xl mx-auto text-center">
        <h1 className="text-5xl font-bold text-gray-900 mb-6 leading-tight">
          {headline}
        </h1>
        <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto">
          {subheadline}
        </p>
        <div className="flex gap-4 justify-center">
          <Button size="lg" onClick={onPrimaryClick}>
            {primaryCta}
          </Button>
          <Button variant="outline" size="lg" onClick={onSecondaryClick}>
            {secondaryCta}
          </Button>
        </div>
      </div>
    </section>
  );
}
