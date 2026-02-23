import { useState } from 'react';

const installCode = `npm install @cipher/sdk`;

const quickStartCode = `import { Cipher } from '@cipher/sdk';

// Initialize Cipher with your API key
const cipher = new Cipher({
  apiKey: process.env.CIPHER_API_KEY,
  environment: 'production'
});

// Connect to your robot
await cipher.connect({
  robotId: 'robot-001',
  sensors: ['lidar', 'vslam', 'camera']
});

// Enable privacy-preserving mode
cipher.enableZKP({
  lidar: true,
  vslam: true,
  proofType: 'location'
});

// Process sensor data with ZK proofs
cipher.onSensorData((data) => {
  // Data is processed locally
  // Only ZK proofs are sent to the cloud
  console.log('Proof generated:', data.proof);
});`;

export function QuickStart() {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async (text) => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section className="px-6 py-20 bg-white">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <p className="text-sm font-semibold text-[#485C11] uppercase tracking-wider mb-4">
            GET STARTED IN MINUTES
          </p>
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Quick Start
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Add privacy-preserving capabilities to your robot with just a few lines of code.
          </p>
        </div>

        {/* Code blocks */}
        <div className="max-w-3xl mx-auto space-y-6">
          {/* Install command */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">1. Install the SDK</p>
            <div className="relative bg-gray-900 rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-4 py-2 bg-gray-800">
                <span className="text-xs text-gray-400">Terminal</span>
                <button
                  onClick={() => copyToClipboard(installCode)}
                  className="text-xs text-gray-400 hover:text-white transition-colors"
                >
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              </div>
              <pre className="p-4 text-sm text-green-400 overflow-x-auto">
                <code>$ {installCode}</code>
              </pre>
            </div>
          </div>

          {/* Quick start code */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">2. Initialize and connect</p>
            <div className="relative bg-gray-900 rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-4 py-2 bg-gray-800">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500" />
                  <div className="w-3 h-3 rounded-full bg-yellow-500" />
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                  <span className="ml-2 text-xs text-gray-400">index.js</span>
                </div>
                <button
                  onClick={() => copyToClipboard(quickStartCode)}
                  className="text-xs text-gray-400 hover:text-white transition-colors"
                >
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              </div>
              <pre className="p-4 text-sm text-gray-300 overflow-x-auto max-h-[400px]">
                <code>{quickStartCode}</code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
