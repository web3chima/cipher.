import { Header } from '../components/sections/Header';
import { DeveloperHero } from '../components/sections/DeveloperHero';
import { QuickStart } from '../components/sections/QuickStart';
import { SDKFeatures } from '../components/sections/SDKFeatures';
import { DeveloperResources } from '../components/sections/DeveloperResources';
import { Footer } from '../components/sections/Footer';

export function DevelopersPage() {
  return (
    <div className="min-h-screen bg-white">
      <Header logoSrc="/logo.png" />

      <DeveloperHero
        title="Cipher Docs & APIs"
        subtitle="Unifies core platform capabilities, standards, and best practices to help developers and partners build and scale secure robotic applications."
        docsUrl="#docs"
        githubUrl="https://github.com/cipher"
      />

      <QuickStart />

      <SDKFeatures />

      <DeveloperResources />

      <Footer logoSrc="/logo.png" />
    </div>
  );
}
