import { Button } from '../ui/Button';

export function DeveloperHero({
  title = 'Cipher Docs & APIs',
  subtitle = 'Unifies core platform capabilities, standards, and best practices to help developers and partners build and scale secure robotic applications.',
  docsUrl = '#docs',
  githubUrl = '#github',
}) {
  return (
    <section className="relative py-24 md:py-32 overflow-hidden">
      {/* Grid pattern background */}
      <div className="absolute inset-0">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `
              linear-gradient(to right, #e5e7eb 1px, transparent 1px),
              linear-gradient(to bottom, #e5e7eb 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px',
          }}
        />
        {/* Plus signs at intersections */}
        <div
          className="absolute inset-0 opacity-40"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%239ca3af' stroke-width='1'%3E%3Cpath d='M30 25v10M25 30h10'/%3E%3C/g%3E%3C/svg%3E")`,
            backgroundSize: '60px 60px',
          }}
        />
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 mb-6">
          {title}
        </h1>
        <p className="text-lg md:text-xl text-gray-600 mb-10 max-w-2xl mx-auto">
          {subtitle}
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Button
            variant="primary"
            size="lg"
            pill
            onClick={() => window.location.href = docsUrl}
          >
            Open the Docs
            <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </Button>
          <Button
            variant="secondary"
            size="lg"
            pill
            onClick={() => window.location.href = githubUrl}
          >
            View on GitHub
          </Button>
        </div>
      </div>

      {/* Bottom border */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gray-200" />
    </section>
  );
}
