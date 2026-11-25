/**
 * A tiny observable store backing `usePersistentState`.
 *
 * The value lives outside React so a write from an unmounted view still lands
 * in local storage and reaches whichever view is mounted. That matters for
 * work that outlives its view, such as an MCP probe that finishes after the
 * user switches from the MCP Hub to Chat.
 */

export interface PersistentStore<T> {
  get(): T;
  set(next: T | ((previous: T) => T)): void;
  subscribe(listener: () => void): () => void;
}

const stores = new Map<string, PersistentStore<unknown>>();

function read<T>(key: string, initial: T): T {
  try {
    const stored = window.localStorage.getItem(key);
    return stored === null ? initial : (JSON.parse(stored) as T);
  } catch {
    return initial;
  }
}

function write(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage may be disabled; the app still works for this session.
  }
}

export function persistentStore<T>(key: string, initial: T): PersistentStore<T> {
  const cached = stores.get(key) as PersistentStore<T> | undefined;
  if (cached) return cached;

  let value = read(key, initial);
  const listeners = new Set<() => void>();
  const store: PersistentStore<T> = {
    get: () => value,
    set: (next) => {
      const resolved =
        typeof next === "function"
          ? (next as (previous: T) => T)(value)
          : next;
      if (Object.is(resolved, value)) return;
      value = resolved;
      write(key, resolved);
      for (const listener of [...listeners]) listener();
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
  stores.set(key, store as PersistentStore<unknown>);
  return store;
}
