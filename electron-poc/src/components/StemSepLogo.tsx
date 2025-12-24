type StemSepLogoProps = {
  className?: string;
  title?: string;
};

/**
 * StemSep logo (inline SVG).
 *
 * Uses `currentColor` so it adapts to dark/light themes automatically.
 */
export function StemSepLogo({ className, title = "StemSep" }: StemSepLogoProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 512 512"
      className={className}
      role="img"
      aria-label={title}
    >
      <path
        d="M64 176 C144 96 256 96 336 176 C416 256 448 256 448 176"
        fill="none"
        stroke="currentColor"
        strokeWidth="40"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M64 256 C144 176 256 176 336 256 C416 336 448 336 448 256"
        fill="none"
        stroke="currentColor"
        strokeWidth="40"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M64 336 C144 256 256 256 336 336 C416 416 448 416 448 336"
        fill="none"
        stroke="currentColor"
        strokeWidth="40"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}


