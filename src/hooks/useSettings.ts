import { useCallback, useEffect, useState } from "react";

import { api } from "../lib/api";
import { errorMessage } from "../lib/format";
import { normalizeSettings } from "../lib/llamaHardening";
import type { AppSettings } from "../types";

export function useSettings() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setSettings(normalizeSettings(await api.getSettings()));
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const save = useCallback(async (next: AppSettings) => {
    const saved = normalizeSettings(await api.saveSettings(next));
    setSettings(saved);
    setError(null);
    return saved;
  }, []);

  return { settings, loading, error, save, refresh, setSettings };
}
