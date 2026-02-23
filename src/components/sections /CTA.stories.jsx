import { CTA } from './CTA';

export default {
  title: 'Sections/CTA',
  component: CTA,
  parameters: {
    layout: 'fullscreen',
  },
};

export const Default = {};

export const Primary = {
  args: {
    variant: 'primary',
    headline: 'Start building today',
    subheadline: 'No credit card required. Free 14-day trial.',
    buttonText: 'Get Started Free',
  },
};

export const Gradient = {
  args: {
    variant: 'gradient',
    headline: 'Transform your business',
    subheadline: 'See why 10,000+ companies trust us with their growth.',
    buttonText: 'Schedule a Demo',
  },
};
