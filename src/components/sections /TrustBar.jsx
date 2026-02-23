export function TrustBar({
  title = 'Powering LIDAR and VSLAM ZD4 of Robots by Industry Leading Manufacturers',
  backgroundImage,
}) {
  return (
    <section className="relative">
      {/* Background */}
      <div className="absolute inset-0">
        {backgroundImage ? (
          <img
            src={backgroundImage}
            alt="Industrial robots"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-r from-green-900 to-emerald-800" />
        )}
        <div className="absolute inset-0 bg-black/40" />
      </div>

      {/* Content */}
      <div className="relative z-10 px-6 py-16 md:py-24">
        <div className="max-w-4xl mx-auto text-center">
          <p className="text-xl md:text-2xl lg:text-3xl text-white font-medium leading-relaxed">
            {title}
          </p>
        </div>
      </div>
    </section>
  );
}
