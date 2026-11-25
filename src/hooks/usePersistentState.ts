import { useSyncExternalStore } from "react";

import { persistentStore } from "../lib/persistentStore";

/**
 * Persists small UI preferences (system prompt, sliders) locally. These are not
 * secrets and never leave the machine.
 *
 * Every hook for the same key shares one store, so a write made after this
 * component unmounted (an in-flight probe, for example) is still saved and is
 * seen by the view that is mounted.
 */
export function usePersistentState<T>(key: string, initial: T) {
  const store = persistentStore(key, initial);
  const value = useSyncExternalStore(store.subscribe, store.get, store.get);
  return [value, store.set] as const;
}
