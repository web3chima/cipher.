import { Button } from '../ui/Button';

export function CTA({
  headline = 'Ready to get started?',
  subheadline = 'Join thousands of satisfied customers today.',
  buttonText = 'Start Free Trial',
  variant = 'default',
  onButtonClick,
}) {
  const variants = {
    default: 'bg-gray-900 text-white',
    primary: 'bg-blue-600 text-white',
    gradient: 'bg-gradient-to-r from-blue-600 to-purple-600 text-white',
  };

  return (
    <section className={`px-6 py-20 ${variants[variant]}`}>
      <div className="max-w-3xl mx-auto text-center">
        <h2 className="text-3xl font-bold mb-4">{headline}</h2>
        <p className="text-lg opacity-90 mb-8">{subheadline}</p>
        <Button
          variant={variant === 'default' ? 'primary' : 'secondary'}
          size="lg"
          onClick={onButtonClick}
        >
          {buttonText}
        </Button>
      </div>
    </section>
  );
}
