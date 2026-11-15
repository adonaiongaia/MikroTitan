export type ThemeId = "acid" | "ember" | "tide";

export interface ThemeDefinition {
  id: ThemeId;
  name: string;
  tagline: string;
  /** Swatches for the settings picker; the live palette lives in index.css. */
  surface: string;
  panel: string;
  accent: string;
  accentSoft: string;
}

export const THEMES: ThemeDefinition[] = [
  {
    id: "acid",
    name: "Acid",
    tagline: "Lime on graphite",
    surface: "#07080a",
    panel: "#0f1218",
    accent: "#cdf75a",
    accentSoft: "#e2ff8f",
  },
  {
    id: "ember",
    name: "Ember",
    tagline: "Copper on soot",
    surface: "#0b0806",
    panel: "#1d1813",
    accent: "#ff7a33",
    accentSoft: "#ffc09a",
  },
  {
    id: "tide",
    name: "Tide",
    tagline: "Cyan on deep water",
    surface: "#05090e",
    panel: "#111d29",
    accent: "#4fd6e8",
    accentSoft: "#9df1fb",
  },
];

export const THEME_STORAGE_KEY = "janas.theme";
export const DEFAULT_THEME: ThemeId = "acid";

export function isThemeId(value: unknown): value is ThemeId {
  return THEMES.some((theme) => theme.id === value);
}

export function readTheme(): ThemeId {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isThemeId(stored) ? stored : DEFAULT_THEME;
  } catch {
    return DEFAULT_THEME;
  }
}

export function applyTheme(theme: ThemeId, persist = true): void {
  document.documentElement.dataset.theme = theme;
  if (persist) {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Storage can be unavailable; the theme still applies for this session.
    }
  }
}
