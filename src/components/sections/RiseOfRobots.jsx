export function RiseOfRobots() {
  return (
    <section className="px-6 py-20 bg-white">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16">
          <p className="text-sm font-semibold text-[#485C11] uppercase tracking-wider mb-4">
            BUILT ON OPENMIND OM1
          </p>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            THE RISE OF ROBOTS
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Promoting a co-operative positive sum environment with Robots crucial for systems vulnerability of being hacked/coerced
          </p>
        </div>

        {/* Stats */}
        <div className="bg-gray-50 rounded-3xl p-8 md:p-12 text-center mb-16">
          <p className="text-2xl md:text-3xl font-medium text-gray-900">
            Millions of <span className="text-[#485C11] font-bold">ROBOTS</span> by 2030 and Billions by 2040 to exist in <span className="text-[#485C11] font-bold">HOMES</span>
          </p>
        </div>

        {/* Two Column: Problem & Solution */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Problem: Spy in the House */}
          <div className="bg-red-50 border border-red-100 rounded-3xl p-8">
            <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center mb-6">
              <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Spy in the House
            </h3>
            <p className="text-gray-600 leading-relaxed">
              Traditional robots send video feeds, photos, and 3D maps of your private spaces to central servers — creating massive privacy and security risks.
            </p>
          </div>

          {/* Solution: Cipher Software Stack */}
          <div className="bg-[#f5f9ed] border border-[#DFECC6] rounded-3xl p-8">
            <div className="w-12 h-12 bg-[#DFECC6] rounded-xl flex items-center justify-center mb-6">
              <svg className="w-6 h-6 text-[#485C11]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Cipher Software Stack
            </h3>
            <p className="text-gray-600 leading-relaxed mb-4">
              Cipher applies <strong>Zero Knowledge Proofs (ZKPs)</strong> to Light Detection and Ranging (LiDAR) and Visual Simultaneous Localization and Mapping (VSLAM).
            </p>
            <p className="text-gray-600 leading-relaxed">
              Cipher ZKP allows robots to <strong>prove where they are or what they're doing</strong> without sending video feeds, photos, or 3D maps of your private data to a central server.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
