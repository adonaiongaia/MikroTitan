/**
 * One source of truth for every keyboard shortcut in the studio.
 *
 * `Mod` is Ctrl on Linux/Windows and Cmd on macOS. The definitions are also
 * what the cheat sheet renders, so a new binding only has to be added here and
 * in the handler that owns it.
 */

export const SHORTCUT_CATEGORIES = [
  "Navigation",
  "Chat & Composer",
  "Panels & Drawers",
  "Voice & Agent",
] as const;

export type ShortcutCategory = (typeof SHORTCUT_CATEGORIES)[number];

export type ShortcutId =
  | "search-chats"
  | "open-knowledge"
  | "open-models"
  | "open-settings"
  | "new-chat"
  | "send"
  | "toggle-agent"
  | "toggle-internet"
  | "open-shortcuts"
  | "toggle-sessions"
  | "toggle-engine"
  | "dismiss"
  | "dictation"
  | "cancel-recording";

export interface ShortcutVariant {
  /** `KeyboardEvent.key`, compared case-insensitively. */
  key: string;
  mod?: boolean;
  shift?: boolean;
  /** Overrides the definition's input policy for this exact chord. */
  allowInInput?: boolean;
}

export interface ShortcutDefinition {
  id: ShortcutId;
  variants: ShortcutVariant[];
  category: ShortcutCategory;
  label: string;
  description: string;
  /** Runs even when focus is inside an input, textarea or contenteditable. */
  allowInInput?: boolean;
  /** Owned by another listener (the voice recorder); shown here, not routed. */
  external?: boolean;
  /** Only handled while the Chat view is mounted. */
  chatOnly?: boolean;
}

export const SHORTCUTS: ShortcutDefinition[] = [
  {
    id: "search-chats",
    variants: [{ key: "k", mod: true }, { key: "p", mod: true }],
    category: "Navigation",
    label: "Search chats",
    description:
      "Focus the chat search and expand the sessions rail if it is collapsed.",
    allowInInput: true,
  },
  {
    id: "open-knowledge",
    variants: [{ key: "v", mod: true, shift: true }],
    category: "Navigation",
    label: "Knowledge vault",
    description: "Open the local document collections.",
  },
  {
    id: "open-models",
    variants: [{ key: "m", mod: true, shift: true }],
    category: "Navigation",
    label: "Models",
    description: "Open the Hugging Face model explorer.",
  },
  {
    id: "open-settings",
    variants: [{ key: ",", mod: true }],
    category: "Navigation",
    label: "Settings",
    description: "Open runtime, model path and voice settings.",
  },
  {
    id: "new-chat",
    variants: [{ key: "n", mod: true }],
    category: "Chat & Composer",
    label: "New chat",
    description:
      "Start a fresh session with the saved default system prompt.",
  },
  {
    id: "send",
    variants: [{ key: "Enter", mod: true }],
    category: "Chat & Composer",
    label: "Send message",
    description: "Send the current draft, including from inside the composer.",
    allowInInput: true,
  },
  {
    id: "toggle-agent",
    variants: [{ key: "a", mod: true, shift: true }],
    category: "Chat & Composer",
    label: "Agent mode",
    description: "Toggle Agent mode in the composer.",
  },
  {
    id: "toggle-internet",
    variants: [{ key: "i", mod: true, shift: true }],
    category: "Chat & Composer",
    label: "Internet search",
    description: "Toggle the Internet web-search switch in the composer.",
  },
  {
    id: "open-shortcuts",
    // Ctrl+/ is a global override; ? must only fire outside a text field.
    variants: [
      { key: "/", mod: true, allowInInput: true },
      { key: "?", shift: true, allowInInput: false },
    ],
    category: "Chat & Composer",
    label: "Keyboard shortcuts",
    description: "Open this cheat sheet. Press ? outside a text field.",
  },
  {
    id: "toggle-sessions",
    variants: [{ key: "b", mod: true }],
    category: "Panels & Drawers",
    label: "Sessions sidebar",
    description: "Show or hide the left sessions rail while Chat is open.",
    allowInInput: true,
    chatOnly: true,
  },
  {
    id: "toggle-engine",
    variants: [{ key: ".", mod: true }, { key: "]", mod: true }],
    category: "Panels & Drawers",
    label: "Engine settings panel",
    description: "Show or hide the right engine panel while Chat is open.",
    chatOnly: true,
  },
  {
    id: "dismiss",
    variants: [{ key: "Escape" }],
    category: "Panels & Drawers",
    label: "Dismiss or stop",
    description:
      "Close the topmost modal or drawer. With nothing open, cancel the streaming reply or agent run.",
    allowInInput: true,
  },
  {
    id: "dictation",
    variants: [{ key: " ", mod: true }],
    category: "Voice & Agent",
    label: "Voice dictation",
    description: "Start or finish a dictation recording in the composer.",
    external: true,
  },
  {
    id: "cancel-recording",
    variants: [{ key: "Escape" }],
    category: "Voice & Agent",
    label: "Cancel recording",
    description:
      "Discard the active voice recording. Enter finishes and transcribes it instead.",
    external: true,
  },
];

/** True on Apple platforms, where Mod is Cmd instead of Ctrl. */
export function isMacPlatform(): boolean {
  if (typeof navigator === "undefined") return false;
  return /mac|iphone|ipad|ipod/i.test(navigator.platform || "");
}

/**
 * Inputs, textareas and contenteditable text own their keystrokes. Shortcuts
 * marked `allowInInput` bypass this check.
 */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (typeof HTMLElement === "undefined" || !(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName.toLowerCase();
  if (tag === "input" || tag === "textarea") return true;
  return (
    target.isContentEditable ||
    Boolean(target.closest('[contenteditable="true"]'))
  );
}

function matchesVariant(
  event: KeyboardEvent,
  variant: ShortcutVariant,
  isMac: boolean,
): boolean {
  if (variant.mod) {
    const primary = isMac ? event.metaKey : event.ctrlKey;
    if (!primary) return false;
  } else if (event.ctrlKey || event.metaKey || event.altKey) {
    return false;
  }
  if (event.altKey) return false;
  if (event.shiftKey !== Boolean(variant.shift)) return false;
  return event.key.toLowerCase() === variant.key.toLowerCase();
}

/** The definition an event belongs to, or null when nothing is bound. */
export function resolveShortcut(
  event: KeyboardEvent,
  isMac = isMacPlatform(),
  inInput = isTypingTarget(event.target),
): ShortcutDefinition | null {
  for (const shortcut of SHORTCUTS) {
    if (shortcut.external) continue;
    for (const variant of shortcut.variants) {
      if (!matchesVariant(event, variant, isMac)) continue;
      // A chord can opt out of text fields even when its sibling is global
      // (Ctrl+/ opens the sheet while typing, ? does not).
      if (inInput && !(variant.allowInInput ?? shortcut.allowInInput)) {
        continue;
      }
      return shortcut;
    }
  }
  return null;
}

function formatKey(key: string): string {
  if (key === " ") return "Space";
  if (key === "Escape") return "Esc";
  if (key === "Enter") return "Enter";
  return key.length === 1 ? key.toUpperCase() : key;
}

function formatShortcutVariant(
  variant: ShortcutVariant,
  isMac = isMacPlatform(),
): string {
  const parts: string[] = [];
  if (variant.mod) parts.push(isMac ? "⌘" : "Ctrl");
  // Shift is part of typing "?" itself; showing "Shift+?" would be noise.
  if (variant.shift && variant.key !== "?") parts.push("Shift");
  parts.push(formatKey(variant.key));
  return parts.join("+");
}

/** Every chip for a shortcut, e.g. ["Ctrl+K", "Ctrl+P"]. */
export function shortcutKeys(id: ShortcutId, isMac = isMacPlatform()): string[] {
  const shortcut = SHORTCUTS.find((candidate) => candidate.id === id);
  if (!shortcut) return [];
  return shortcut.variants.map((variant) =>
    formatShortcutVariant(variant, isMac),
  );
}

/** A compact label for tooltips, e.g. "Ctrl+K / Ctrl+P". */
export function shortcutLabel(id: ShortcutId, isMac = isMacPlatform()): string {
  return shortcutKeys(id, isMac).join(" / ");
}
