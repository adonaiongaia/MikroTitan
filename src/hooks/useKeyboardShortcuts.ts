import { useEffect, useRef } from "react";

import { resolveShortcut, type ShortcutId } from "../lib/shortcuts";
import { isVoiceRecordingActive } from "./useVoiceRecorder";

export type ShortcutHandler = () => boolean | void;
export type ShortcutHandlers = Partial<Record<ShortcutId, ShortcutHandler>>;

interface ShortcutScope {
  priority: number;
  handlers: { current: ShortcutHandlers };
}

// One listener for the whole app. Scopes are sorted by priority so an open
// modal can claim Escape before the chat beneath it cancels a stream.
const scopes = new Set<ShortcutScope>();
let listenerAttached = false;

function onKeyDown(event: KeyboardEvent) {
  if (event.defaultPrevented || event.isComposing || event.repeat) return;

  const shortcut = resolveShortcut(event);
  if (!shortcut) return;

  // The composer's recorder owns Enter and Escape while a capture is live, so
  // dictation never stops a running reply by accident.
  if (
    isVoiceRecordingActive() &&
    (shortcut.id === "send" || shortcut.id === "dismiss")
  ) {
    return;
  }

  const ordered = [...scopes].sort(
    (left, right) => right.priority - left.priority,
  );
  for (const scope of ordered) {
    const handler = scope.handlers.current[shortcut.id];
    if (!handler) continue;
    // A scope can decline (for example Escape with no drawer open, so the app
    // can cancel the running reply).
    if (handler() === false) continue;
    event.preventDefault();
    return;
  }
}

function attachListener() {
  if (listenerAttached || typeof window === "undefined") return;
  window.addEventListener("keydown", onKeyDown);
  listenerAttached = true;
}

function detachListener() {
  if (!listenerAttached || scopes.size > 0) return;
  window.removeEventListener("keydown", onKeyDown);
  listenerAttached = false;
}

/**
 * Registers shortcut handlers for as long as the calling component is mounted.
 * Handlers may be replaced on every render without re-registering the scope.
 * Higher priorities run first; the first handler that does not return `false`
 * claims the event.
 */
export function useKeyboardShortcuts(
  handlers: ShortcutHandlers,
  options: { priority?: number; enabled?: boolean } = {},
) {
  const { priority = 0, enabled = true } = options;
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    if (!enabled) return;
    const scope: ShortcutScope = { priority, handlers: handlersRef };
    scopes.add(scope);
    attachListener();
    return () => {
      scopes.delete(scope);
      detachListener();
    };
  }, [enabled, priority]);
}
