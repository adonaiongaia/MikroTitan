import { beforeEach, describe, expect, it, vi } from "vitest";

import { persistentStore } from "./persistentStore";

class MemoryStorage {
  private values = new Map<string, string>();

  get length() {
    return this.values.size;
  }

  clear() {
    this.values.clear();
  }

  getItem(key: string) {
    return this.values.get(key) ?? null;
  }

  key(index: number) {
    return [...this.values.keys()][index] ?? null;
  }

  removeItem(key: string) {
    this.values.delete(key);
  }

  setItem(key: string, value: string) {
    this.values.set(key, value);
  }
}

let storage: MemoryStorage;

beforeEach(() => {
  storage = new MemoryStorage();
  Object.defineProperty(globalThis, "window", {
    value: { localStorage: storage },
    configurable: true,
    writable: true,
  });
});

describe("persistentStore", () => {
  it("writes through to local storage", () => {
    const store = persistentStore("test.write", { enabled: false });
    store.set({ enabled: true });
    expect(store.get()).toEqual({ enabled: true });
    expect(storage.getItem("test.write")).toBe('{"enabled":true}');
  });

  it("notifies subscribers and stops after they unsubscribe", () => {
    const store = persistentStore("test.subscribe", 0);
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);

    store.set(1);
    expect(listener).toHaveBeenCalledTimes(1);

    unsubscribe();
    store.set(2);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("persists a write made after its view unsubscribed", () => {
    const store = persistentStore<string[]>("test.probe", []);
    const unsubscribe = store.subscribe(() => {});
    unsubscribe();

    store.set([{ id: "server", tokens: 57 } as unknown as string]);

    // The next view (or the next app launch) reads the same value.
    expect(persistentStore<string[]>("test.probe", []).get()).toEqual([
      { id: "server", tokens: 57 },
    ]);
    expect(JSON.parse(storage.getItem("test.probe") ?? "[]")).toEqual([
      { id: "server", tokens: 57 },
    ]);
  });

  it("applies functional updates against the latest value", () => {
    const store = persistentStore("test.updater", [1]);
    store.set((previous) => [...previous, 2]);
    store.set((previous) => [...previous, 3]);
    expect(store.get()).toEqual([1, 2, 3]);
  });

  it("shares one store per key", () => {
    const first = persistentStore("test.shared", { count: 0 });
    const second = persistentStore("test.shared", { count: 0 });
    expect(second).toBe(first);
  });

  it("falls back to the initial value when storage is corrupt", () => {
    storage.setItem("test.corrupt", "{not json");
    expect(persistentStore("test.corrupt", "fallback").get()).toBe("fallback");
  });

  it("keeps working when storage throws", () => {
    Object.defineProperty(globalThis, "window", {
      value: {
        localStorage: {
          getItem: () => {
            throw new Error("denied");
          },
          setItem: () => {
            throw new Error("denied");
          },
        },
      },
      configurable: true,
      writable: true,
    });
    const store = persistentStore("test.denied", 1);
    expect(store.get()).toBe(1);
    expect(() => store.set(2)).not.toThrow();
    expect(store.get()).toBe(2);
  });
});
