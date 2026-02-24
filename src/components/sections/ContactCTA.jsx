import { Button } from '../ui/Button';

export function ContactCTA({
  title = 'Talk to the Team',
  description = 'Ready to secure your robotic environment? Get in touch with our experts.',
  buttonText = 'Talk to the Team',
  backgroundImage,
  calendlyUrl = 'https://calendly.com/web3chima/30min',
}) {
  const handleBookTime = () => {
    window.open(calendlyUrl, '_blank', 'noopener,noreferrer');
  };

  return (
    <section className="relative px-6 py-20">
      {/* Background */}
      <div className="absolute inset-0">
        {backgroundImage ? (
          <img
            src={backgroundImage}
            alt="Background"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-gray-100 to-gray-200" />
        )}
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-2xl mx-auto text-center">
        <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          {title}
        </h2>
        {description && (
          <p className="text-lg text-gray-600 mb-8">{description}</p>
        )}
        <Button variant="primary" size="lg" pill onClick={handleBookTime}>
          {buttonText}
        </Button>
      </div>
    </section>
  );
}
