import { useId } from "react";

interface LogoProps {
  className?: string;
  withTile?: boolean;
}

export function Logo({ className = "h-9 w-9", withTile = true }: LogoProps) {
  const gradientId = `janas-mark-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  return (
    <svg
      viewBox="0 0 64 64"
      className={className}
      role="img"
      aria-label="Janas Studio"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop
            offset="0%"
            style={{ stopColor: "rgb(var(--accent-300))" }}
          />
          <stop
            offset="100%"
            style={{ stopColor: "rgb(var(--accent-600))" }}
          />
        </linearGradient>
      </defs>
      {withTile ? (
        <rect
          width="64"
          height="64"
          rx="16"
          style={{ fill: "rgb(var(--ink-850))" }}
        />
      ) : null}
      <rect
        x="18"
        y="14"
        width="28"
        height="6"
        rx="2"
        fill={`url(#${gradientId})`}
      />
      <rect
        x="18"
        y="14"
        width="6"
        height="36"
        rx="2"
        fill={`url(#${gradientId})`}
      />
      <rect
        x="40"
        y="14"
        width="6"
        height="36"
        rx="2"
        fill={`url(#${gradientId})`}
      />
      <rect
        x="18"
        y="32"
        width="28"
        height="5"
        rx="2"
        fill={`url(#${gradientId})`}
        opacity="0.72"
      />
      <circle cx="32" cy="44" r="3.5" fill={`url(#${gradientId})`} />
    </svg>
  );
}
