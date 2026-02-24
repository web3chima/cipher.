const variants = {
  primary: 'bg-[#485C11] hover:bg-[#3d4e0e] text-white',
  secondary: 'bg-[#DFECC6] hover:bg-[#d0e0af] text-[#485C11]',
  outline: 'border-2 border-[#485C11] text-[#485C11] hover:bg-[#f5f9ed]',
  ghost: 'text-[#485C11] hover:bg-[#f5f9ed]',
  dark: 'bg-black hover:bg-gray-900 text-white',
};

const sizes = {
  sm: 'px-4 py-2 text-sm',
  md: 'px-5 py-2.5 text-base',
  lg: 'px-6 py-3 text-lg',
};

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  pill = false,
  className = '',
  ...props
}) {
  return (
    <button
      className={`
        ${variants[variant]}
        ${sizes[size]}
        ${pill ? 'rounded-full' : 'rounded-xl'}
        font-medium transition-colors duration-200
        focus:outline-none focus:ring-2 focus:ring-[#485C11] focus:ring-offset-2
        disabled:opacity-50 disabled:cursor-not-allowed
        ${className}
      `}
      {...props}
    >
      {children}
    </button>
  );
}
