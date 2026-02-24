import { Button } from '../ui/Button';

export function CipherHero({
  headline = 'Privacy and Security stack for Environments with Robots.',
  ctaText = 'Learn more',
  heroImage,
  onCtaClick,
}) {
  return (
    <section className="relative pt-24 pb-0 overflow-hidden bg-white">
      {/* Content */}
      <div className="relative z-10 px-6 pt-12 pb-16">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 leading-tight mb-8">
            {headline}
          </h1>
          <Button variant="primary" size="lg" pill onClick={onCtaClick}>
            {ctaText}
          </Button>
        </div>
      </div>

      {/* Hero Image - layered below text */}
      <div className="relative w-full mt-8">
        {/* Fade overlay at top */}
        <div className="absolute top-0 left-0 right-0 h-24 bg-gradient-to-b from-white to-transparent z-10" />

        {heroImage ? (
          <img
            src={heroImage}
            alt="Robotic environment"
            className="w-full h-auto object-cover max-h-[600px]"
          />
        ) : (
          /* Placeholder minimalist image */
          <div className="w-full h-[500px] bg-gradient-to-br from-sky-50 via-blue-100 to-indigo-100 relative">
            {/* Abstract robot/tech shapes */}
            <div className="absolute inset-0 flex items-center justify-center opacity-30">
              <svg className="w-96 h-96" viewBox="0 0 200 200" fill="none">
                {/* Abstract geometric shapes representing robots */}
                <rect x="60" y="40" width="80" height="60" rx="8" stroke="#485C11" strokeWidth="2" />
                <circle cx="85" cy="65" r="8" fill="#485C11" opacity="0.5" />
                <circle cx="115" cy="65" r="8" fill="#485C11" opacity="0.5" />
                <rect x="75" y="100" width="50" height="70" rx="4" stroke="#485C11" strokeWidth="2" />
                <line x1="60" y1="120" x2="40" y2="140" stroke="#485C11" strokeWidth="2" />
                <line x1="140" y1="120" x2="160" y2="140" stroke="#485C11" strokeWidth="2" />
                <rect x="70" y="170" width="20" height="20" rx="2" stroke="#485C11" strokeWidth="2" />
                <rect x="110" y="170" width="20" height="20" rx="2" stroke="#485C11" strokeWidth="2" />
              </svg>
            </div>
            {/* Grid pattern overlay */}
            <div
              className="absolute inset-0 opacity-5"
              style={{
                backgroundImage: 'radial-gradient(#485C11 1px, transparent 1px)',
                backgroundSize: '20px 20px'
              }}
            />
          </div>
        )}

        {/* Fade overlay at bottom */}
        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-white to-transparent" />
      </div>
    </section>
  );
}
