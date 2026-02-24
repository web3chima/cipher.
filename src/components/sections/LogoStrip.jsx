const placeholderLogos = [
  { id: 1, name: 'Unitree', src: '/logos/IMG_0967.jpg' },
  { id: 2, name: 'OpenMind', src: '/logos/IMG_0968.jpg' },
  { id: 3, name: 'KUKA', src: '/logos/IMG_0969.jpg' },
  { id: 4, name: 'UBTECH', src: '/logos/IMG_0970.jpg' },
  { id: 5, name: 'Hanson Robotics', src: '/logos/IMG_0971.jpg' },
  { id: 6, name: 'Boston Dynamics', src: '/logos/IMG_0972.jpg' },
];

export function LogoStrip({ logos = placeholderLogos, title }) {
  // Duplicate logos for seamless infinite scroll
  const duplicatedLogos = [...logos, ...logos];

  return (
    <section className="py-12 bg-white border-y border-gray-100 overflow-hidden">
      <div className="max-w-6xl mx-auto px-6">
        {title && (
          <p className="text-center text-sm text-gray-500 mb-8">{title}</p>
        )}
      </div>

      {/* Marquee container */}
      <div className="relative">
        {/* Gradient fade on edges */}
        <div className="absolute left-0 top-0 bottom-0 w-24 bg-gradient-to-r from-white to-transparent z-10" />
        <div className="absolute right-0 top-0 bottom-0 w-24 bg-gradient-to-l from-white to-transparent z-10" />

        {/* Scrolling logos */}
        <div className="flex animate-marquee">
          {duplicatedLogos.map((logo, index) => (
            <div
              key={`${logo.id}-${index}`}
              className="flex-shrink-0 flex items-center justify-center mx-8 md:mx-12"
            >
              {logo.src ? (
                <img
                  src={logo.src}
                  alt={logo.name}
                  className="h-8 md:h-10 object-contain grayscale hover:grayscale-0 transition-all opacity-60 hover:opacity-100"
                />
              ) : (
                <div className="h-8 md:h-10 px-6 bg-gray-200 rounded flex items-center justify-center">
                  <span className="text-xs text-gray-500 font-medium">{logo.name}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <style>{`
        @keyframes marquee {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(-50%);
          }
        }
        .animate-marquee {
          animation: marquee 20s linear infinite;
        }
        .animate-marquee:hover {
          animation-play-state: paused;
        }
      `}</style>
    </section>
  );
}
