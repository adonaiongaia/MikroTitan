import mark from "../assets/mikrotitan-mark.png";

interface LogoProps {
  className?: string;
  withTile?: boolean;
}

export function Logo({ className = "h-9 w-9", withTile = true }: LogoProps) {
  return (
    <img
      src={mark}
      alt="MikroTitan"
      draggable={false}
      className={`${className} object-contain ${
        withTile ? "drop-shadow-[0_0_14px_rgba(34,211,238,0.28)]" : ""
      }`}
    />
  );
}
