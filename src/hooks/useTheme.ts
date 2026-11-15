import { useCallback, useState } from "react";

import { applyTheme, readTheme, type ThemeId } from "../lib/theme";

export function useTheme() {
  const [theme, setThemeState] = useState<ThemeId>(() => readTheme());

  const setTheme = useCallback((next: ThemeId) => {
    applyTheme(next);
    setThemeState(next);
  }, []);

  return { theme, setTheme };
}
