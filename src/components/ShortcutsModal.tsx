import { useEffect, useRef } from "react";
import { Keyboard, X } from "lucide-react";

import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import {
  SHORTCUTS,
  SHORTCUT_CATEGORIES,
  shortcutKeys,
} from "../lib/shortcuts";

interface ShortcutsModalProps {
  open: boolean;
  onClose: () => void;
}

export function ShortcutsModal({ open, onClose }: ShortcutsModalProps) {
  const closeRef = useRef<HTMLButtonElement>(null);

  // The sheet sits above every other modal, so its Escape runs first.
  useKeyboardShortcuts(
    {
      dismiss: () => {
        onClose();
        return true;
      },
    },
    { priority: 100, enabled: open },
  );

  useEffect(() => {
    if (open) closeRef.current?.focus();
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center bg-ink-950/80 p-6 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="shortcuts-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="panel flex max-h-[86vh] w-full max-w-2xl flex-col overflow-hidden bg-ink-900 shadow-panel animate-fade-in">
        <div className="flex items-start justify-between gap-4 border-b border-white/5 px-5 py-4">
          <div>
            <p className="flex items-center gap-2 text-[10px] uppercase tracking-[0.16em] text-slate-500">
              <Keyboard className="h-3 w-3 text-acid-400" />
              Reference
            </p>
            <h2
              id="shortcuts-title"
              className="mt-1 text-base font-semibold text-white"
            >
              Keyboard shortcuts
            </h2>
            <p className="mt-1 text-[10.5px] leading-5 text-slate-500">
              Shortcuts stay out of text fields, except sending, chat search,
              the sessions sidebar, this sheet and Escape.
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
            aria-label="Close keyboard shortcuts"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2">
            {SHORTCUT_CATEGORIES.map((category) => {
              const items = SHORTCUTS.filter(
                (shortcut) => shortcut.category === category,
              );
              if (!items.length) return null;
              return (
                <section key={category} aria-label={category}>
                  <h3 className="text-[9.5px] font-semibold uppercase tracking-[0.16em] text-slate-600">
                    {category}
                  </h3>
                  <ul className="mt-2 space-y-1">
                    {items.map((shortcut) => {
                      const keys = shortcutKeys(shortcut.id);
                      return (
                        <li
                          key={shortcut.id}
                          className="rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-1.5"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <span className="flex min-w-0 flex-wrap items-center gap-1.5 text-[11px] text-slate-300">
                              {shortcut.label}
                              {shortcut.chatOnly ? (
                                <span className="rounded border border-white/10 px-1 py-px text-[8px] font-medium uppercase tracking-[0.12em] text-slate-500">
                                  Chat only
                                </span>
                              ) : null}
                            </span>
                            <span className="flex shrink-0 flex-wrap justify-end gap-1">
                              {keys.map((chip) => (
                                <kbd
                                  key={chip}
                                  className="rounded border border-white/10 bg-ink-950/70 px-1.5 py-0.5 font-mono text-[9px] text-slate-300"
                                >
                                  {chip}
                                </kbd>
                              ))}
                            </span>
                          </div>
                          <p className="mt-0.5 text-[9.5px] leading-4 text-slate-600">
                            {shortcut.description}
                          </p>
                        </li>
                      );
                    })}
                  </ul>
                </section>
              );
            })}
          </div>
        </div>

        <div className="border-t border-white/5 px-5 py-2.5 text-[9.5px] text-slate-600">
          Voice capture takes over Enter and Escape until the recording ends.
        </div>
      </div>
    </div>
  );
}
