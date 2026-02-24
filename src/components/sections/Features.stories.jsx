import { Features } from './Features';

export default {
  title: 'Sections/Features',
  component: Features,
  parameters: {
    layout: 'fullscreen',
  },
};

export const Default = {};

export const FourColumns = {
  args: {
    columns: 4,
    features: [
      { icon: '📊', title: 'Analytics', description: 'Deep insights into your data' },
      { icon: '🔄', title: 'Sync', description: 'Real-time synchronization' },
      { icon: '🌐', title: 'Global CDN', description: 'Fast delivery worldwide' },
      { icon: '📱', title: 'Mobile Ready', description: 'Works on any device' },
    ],
  },
};

export const TwoColumns = {
  args: {
    columns: 2,
    title: 'Core Features',
    subtitle: 'The essentials you need',
    features: [
      { icon: '🚀', title: 'Fast Performance', description: 'Blazing fast load times' },
      { icon: '🛡️', title: 'Reliable', description: '99.99% uptime guarantee' },
    ],
  },
};
