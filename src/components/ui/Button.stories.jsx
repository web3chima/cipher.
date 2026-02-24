import { Button } from './Button';

export default {
  title: 'UI/Button',
  component: Button,
  argTypes: {
    variant: {
      control: 'select',
      options: ['primary', 'secondary', 'outline', 'ghost', 'dark'],
    },
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
    },
    pill: {
      control: 'boolean',
    },
  },
};

export const Primary = {
  args: {
    children: 'Learn more',
    variant: 'primary',
    pill: true,
  },
};

export const Secondary = {
  args: {
    children: 'View Documentation',
    variant: 'secondary',
    pill: true,
  },
};

export const Dark = {
  args: {
    children: 'Talk to the Team',
    variant: 'dark',
    pill: true,
  },
};

export const AllVariants = {
  render: () => (
    <div className="flex flex-wrap gap-4 items-center">
      <Button variant="primary" pill>Primary</Button>
      <Button variant="secondary" pill>Secondary</Button>
      <Button variant="outline" pill>Outline</Button>
      <Button variant="ghost" pill>Ghost</Button>
      <Button variant="dark" pill>Dark</Button>
    </div>
  ),
};

export const AllSizes = {
  render: () => (
    <div className="flex gap-4 items-center">
      <Button size="sm" pill>Small</Button>
      <Button size="md" pill>Medium</Button>
      <Button size="lg" pill>Large</Button>
    </div>
  ),
};

export const Rounded = {
  render: () => (
    <div className="flex gap-4 items-center">
      <Button variant="primary">Rounded</Button>
      <Button variant="primary" pill>Pill</Button>
    </div>
  ),
};
