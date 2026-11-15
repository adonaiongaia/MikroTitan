import { Check } from "lucide-react";

import { cn } from "../../lib/cn";
import { THEMES, type ThemeId } from "../../lib/theme";

interface ThemePickerProps {
  theme: ThemeId;
  onChange: (theme: ThemeId) => void;
}

export function ThemePicker({ theme, onChange }: ThemePickerProps) {
  return (
    <div className="grid gap-2.5 sm:grid-cols-3" role="group" aria-label="Theme">
      {THEMES.map((option) => {
        const active = option.id === theme;
        return (
          <button
            key={option.id}
            type="button"
            onClick={() => onChange(option.id)}
            aria-pressed={active}
            className={cn(
              "rounded-xl border p-2 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-acid-400/40",
              active
                ? "border-acid-400/50 bg-acid-400/[0.06]"
                : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]",
            )}
          >
            <span
              className="relative block h-[74px] overflow-hidden rounded-lg border border-white/10 p-2"
              style={{ background: option.surface }}
            >
              <span className="flex items-center gap-1">
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{ background: option.accent }}
                />
                <span
                  className="h-1 w-9 rounded-full"
                  style={{ background: option.accentSoft, opacity: 0.5 }}
                />
                <span
                  className="ml-auto h-2.5 w-7 rounded-full"
                  style={{ background: option.accent, opacity: 0.8 }}
                />
              </span>
              <span className="mt-2 flex gap-1.5">
                <span
                  className="h-8 w-5 rounded"
                  style={{ background: option.panel }}
                />
                <span className="flex-1 space-y-1 pt-0.5">
                  <span
                    className="block h-1 w-3/4 rounded-full"
                    style={{ background: option.accent, opacity: 0.55 }}
                  />
                  <span
                    className="block h-1 w-1/2 rounded-full"
                    style={{ background: "#ffffff", opacity: 0.16 }}
                  />
                  <span
                    className="block h-1 w-2/3 rounded-full"
                    style={{ background: "#ffffff", opacity: 0.1 }}
                  />
                </span>
              </span>
              {active ? (
                <span
                  className="absolute right-1.5 top-1.5 flex h-3.5 w-3.5 items-center justify-center rounded-full"
                  style={{ background: option.accent }}
                >
                  <Check
                    className="h-2.5 w-2.5"
                    style={{ color: option.surface }}
                    strokeWidth={3.5}
                  />
                </span>
              ) : null}
            </span>
            <span className="mt-2 flex items-baseline justify-between gap-2 px-0.5">
              <span className="text-[11px] font-medium text-slate-200">
                {option.name}
              </span>
              <span className="truncate text-[9px] text-slate-600">
                {option.tagline}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}
