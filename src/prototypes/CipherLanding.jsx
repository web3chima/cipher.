import { Header } from '../components/sections/Header';
import { CipherHero } from '../components/sections/CipherHero';
import { TrustBar } from '../components/sections/TrustBar';
import { RiseOfRobots } from '../components/sections/RiseOfRobots';
import { LogoStrip } from '../components/sections/LogoStrip';
import { SDKShowcase } from '../components/sections/SDKShowcase';
import { ContactCTA } from '../components/sections/ContactCTA';
import { Footer } from '../components/sections/Footer';

export function CipherLanding() {
  return (
    <div className="min-h-screen bg-white">
      <Header logoSrc="/logo.png" />

      <CipherHero
        headline="Privacy and Security stack for Environment with Robots."
        ctaText="GET STARTED"
        heroImage="/hero-image.png"
      />

      <TrustBar
        title="Powering LiDAR and VSLAM ZKPs of Robots by Industry Leading Manufacturers"
      />

      <RiseOfRobots />

      <LogoStrip
        title="Manufacturers"
        logos={[
          { id: 1, name: 'Unitree', src: '/logos/IMG_0967.jpg' },
          { id: 2, name: 'OpenMind', src: '/logos/IMG_0968.jpg' },
          { id: 3, name: 'KUKA', src: '/logos/IMG_0969.jpg' },
          { id: 4, name: 'UBTECH', src: '/logos/IMG_0970.jpg' },
          { id: 5, name: 'Hanson Robotics', src: '/logos/IMG_0971.jpg' },
          { id: 6, name: 'Boston Dynamics', src: '/logos/IMG_0972.jpg' },
        ]}
      />

      <SDKShowcase
        title="Build ROBOTS with Cipher SDK"
        description="You need a solution that keeps up. That's why we developed CIPHER, a developer-friendly approach to integrate your ROBOTS with Cipher for consumer privacy."
      />

      <ContactCTA
        title="Talk to the Team"
        description="Schedule a quick call to learn how Cipher can turn your work into a powerful advantage."
        buttonText="BOOK TIME"
      />

      <Footer logoSrc="/logo.png" />
    </div>
  );
}
