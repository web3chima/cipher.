import { Hero } from './Hero';

export default {
  title: 'Sections/Hero',
  component: Hero,
  parameters: {
    layout: 'fullscreen',
  },
};

export const Default = {
  args: {
    headline: 'Build something amazing',
    subheadline: 'Create beautiful, responsive websites with our powerful platform.',
    primaryCta: 'Get Started',
    secondaryCta: 'Learn More',
  },
};

export const SaaSProduct = {
  args: {
    headline: 'Automate your workflow',
    subheadline: 'Save hours every week with intelligent automation that learns how you work.',
    primaryCta: 'Start Free Trial',
    secondaryCta: 'Watch Demo',
  },
};

export const Agency = {
  args: {
    headline: 'We design digital experiences',
    subheadline: 'Award-winning creative agency helping brands stand out in a crowded world.',
    primaryCta: 'View Our Work',
    secondaryCta: 'Contact Us',
  },
};
