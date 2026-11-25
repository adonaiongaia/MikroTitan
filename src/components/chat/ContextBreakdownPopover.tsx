import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Pin,
  RotateCcw,
  X,
} from "lucide-react";

import { cn } from "../../lib/cn";
import { formatCount } from "../../lib/format";
import type { ContextItemKind } from "../../types";

/** One row inside the breakdown: a file, a turn, a chunk, a memory or a scope. */
export interface ContextBreakdownItem {
  id: string;
  kind: ContextItemKind;
  label: string;
  tokens: number;
  detail?: string;
  pinned: boolean;
  evicted: boolean;
}

export interface ContextBreakdownCategory {
  key: string;
  label: string;
  /** Shown next to the category total when items are not token-priced. */
  note?: string;
  items: ContextBreakdownItem[];
}

interface ContextBreakdownPopoverProps {
  open: boolean;
  onClose: () => void;
  categories: ContextBreakdownCategory[];
  contextUsed: number;
  contextMax: number;
  pinnedTokens: number;
  onTogglePin: (item: ContextBreakdownItem) => void;
  onToggleEvict: (item: ContextBreakdownItem) => void;
  onRestoreEvicted: () => void;
}

const PINNED_SHARE_WARNING = 0.75;

export function ContextBreakdownPopover({
  open,
  onClose,
  categories,
  contextUsed,
  contextMax,
  pinnedTokens,
  onTogglePin,
  onToggleEvict,
  onRestoreEvicted,
}: ContextBreakdownPopoverProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  const sorted = useMemo(
    () =>
      categories
        .map((category) => ({
          ...category,
          items: [...category.items].sort(
            (left, right) => right.tokens - left.tokens,
          ),
        }))
        .filter((category) => category.items.length > 0),
    [categories],
  );

  const evictedCount = useMemo(
    () =>
      sorted.reduce(
        (total, category) =>
          total + category.items.filter((item) => item.evicted).length,
        0,
      ),
    [sorted],
  );

  if (!open) return null;

  const pinnedShare = contextMax > 0 ? pinnedTokens / contextMax : 0;
  const usedShare = contextMax > 0 ? Math.min(1, contextUsed / contextMax) : 0;

  return (
    <div
      role="dialog"
      aria-label="Context usage breakdown"
      className="absolute bottom-full right-0 z-40 mb-2 w-[22rem] overflow-hidden rounded-2xl border border-white/10 bg-ink-900/95 shadow-panel backdrop-blur"
    >
      <div className="flex items-start justify-between gap-2 border-b border-white/5 px-3.5 py-2.5">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold text-slate-200">
            Context breakdown
          </p>
          <p className="mt-0.5 font-mono text-[9.5px] text-slate-500">
            {formatCount(contextUsed)} / {formatCount(contextMax)} tokens ·{" "}
            {Math.round(usedShare * 100)}%
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-md p-1 text-slate-600 transition hover:bg-white/5 hover:text-slate-300"
          aria-label="Close context breakdown"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="px-3.5 pt-2.5">
        <div className="flex h-1.5 overflow-hidden rounded-full bg-white/5">
          <span
            className="h-full bg-acid-400 transition-all duration-300"
            style={{ width: `${Math.min(100, pinnedShare * 100)}%` }}
            title={`Pinned ${formatCount(pinnedTokens)} tokens`}
          />
          <span
            className="h-full bg-slate-500/60 transition-all duration-300"
            style={{
              width: `${Math.max(0, usedShare - pinnedShare) * 100}%`,
            }}
          />
        </div>
        <div className="mt-1.5 flex items-center justify-between text-[9px] text-slate-600">
          <span className="inline-flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-acid-400" />
            pinned {formatCount(pinnedTokens)}
          </span>
          <span>{formatCount(contextUsed)} in use</span>
        </div>
      </div>

      {pinnedShare > PINNED_SHARE_WARNING ? (
        <div className="mx-3.5 mt-2.5 flex items-start gap-1.5 rounded-lg border border-amber-400/25 bg-amber-400/5 px-2 py-1.5 text-[9.5px] leading-4 text-amber-200/90">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
          <span>
            Pinned items alone use {Math.round(pinnedShare * 100)}% of the
            window. Unpin or evict something before the next long turn.
          </span>
        </div>
      ) : null}

      <div className="max-h-[18rem] overflow-y-auto px-1.5 py-2">
        {sorted.length === 0 ? (
          <p className="px-2 py-4 text-center text-[10px] leading-5 text-slate-600">
            Nothing is attached to this turn yet. Attach a project folder or
            document collection to see its token cost here.
          </p>
        ) : (
          sorted.map((category) => {
            const isCollapsed = Boolean(collapsed[category.key]);
            const total = category.items.reduce(
              (sum, item) => sum + item.tokens,
              0,
            );
            const pinnedCount = category.items.filter(
              (item) => item.pinned,
            ).length;
            return (
              <div key={category.key} className="mb-0.5">
                <button
                  type="button"
                  onClick={() =>
                    setCollapsed((previous) => ({
                      ...previous,
                      [category.key]: !previous[category.key],
                    }))
                  }
                  className="flex w-full items-center gap-1.5 rounded-lg px-2 py-1.5 text-left transition hover:bg-white/[0.03]"
                  aria-expanded={!isCollapsed}
                >
                  {isCollapsed ? (
                    <ChevronRight className="h-3 w-3 shrink-0 text-slate-600" />
                  ) : (
                    <ChevronDown className="h-3 w-3 shrink-0 text-slate-600" />
                  )}
                  <span className="min-w-0 flex-1 truncate text-[10.5px] font-medium text-slate-300">
                    {category.label}
                  </span>
                  {pinnedCount ? (
                    <span className="inline-flex items-center gap-0.5 rounded border border-acid-400/25 bg-acid-400/10 px-1 py-px font-mono text-[8px] text-acid-200/90">
                      <Pin className="h-2 w-2" />
                      {pinnedCount}
                    </span>
                  ) : null}
                  {category.note ? (
                    <span className="font-mono text-[9px] text-slate-600">
                      {category.note}
                    </span>
                  ) : null}
                  <span className="font-mono text-[9px] text-slate-600">
                    {formatCount(total)} tok
                  </span>
                </button>

                {isCollapsed
                  ? null
                  : category.items.map((item) => (
                      <div
                        key={item.id}
                        className={cn(
                          "group/row flex items-center gap-1.5 rounded-lg px-2 py-1 pl-5 transition",
                          item.evicted
                            ? "opacity-55"
                            : "hover:bg-white/[0.03]",
                        )}
                      >
                        <div className="min-w-0 flex-1">
                          <p
                            className={cn(
                              "truncate text-[10px]",
                              item.pinned
                                ? "text-acid-200"
                                : item.evicted
                                  ? "text-slate-500 line-through"
                                  : "text-slate-400",
                            )}
                            title={item.label}
                          >
                            {item.label}
                          </p>
                          {item.detail ? (
                            <p className="truncate text-[8.5px] text-slate-600">
                              {item.detail}
                            </p>
                          ) : null}
                        </div>
                        <span className="shrink-0 font-mono text-[8.5px] text-slate-600">
                          {item.tokens ? formatCount(item.tokens) : "—"}
                        </span>
                        <button
                          type="button"
                          onClick={() => onTogglePin(item)}
                          aria-pressed={item.pinned}
                          aria-label={
                            item.pinned
                              ? `Unpin ${item.label}`
                              : `Pin ${item.label}`
                          }
                          title={
                            item.pinned
                              ? "Unpin: allow the budget to trim this item again"
                              : "Pin: never drop or truncate this item"
                          }
                          className={cn(
                            "shrink-0 rounded-md p-1 transition",
                            item.pinned
                              ? "bg-acid-400/15 text-acid-300"
                              : "text-slate-600 hover:bg-white/5 hover:text-slate-300",
                          )}
                        >
                          <Pin className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() => onToggleEvict(item)}
                          aria-pressed={item.evicted}
                          aria-label={
                            item.evicted
                              ? `Restore ${item.label}`
                              : `Evict ${item.label}`
                          }
                          title={
                            item.evicted
                              ? "Restore this item to the prompt"
                              : "Evict: drop this item from the next prompt now"
                          }
                          className={cn(
                            "shrink-0 rounded-md p-1 transition",
                            item.evicted
                              ? "text-acid-300 hover:bg-acid-400/10"
                              : "text-slate-600 hover:bg-rose-500/10 hover:text-rose-300",
                          )}
                        >
                          {item.evicted ? (
                            <Eye className="h-3 w-3" />
                          ) : (
                            <EyeOff className="h-3 w-3" />
                          )}
                        </button>
                      </div>
                    ))}
              </div>
            );
          })
        )}
      </div>

      <div className="flex items-center justify-between gap-2 border-t border-white/5 px-3.5 py-2">
        <span className="text-[9px] leading-4 text-slate-600">
          Pinned items survive trimming and compaction.
        </span>
        {evictedCount ? (
          <button
            type="button"
            onClick={onRestoreEvicted}
            className="inline-flex shrink-0 items-center gap-1 rounded-md border border-acid-400/25 bg-acid-400/10 px-2 py-1 text-[9px] text-acid-200 transition hover:bg-acid-400/20"
          >
            <RotateCcw className="h-2.5 w-2.5" />
            Restore evicted ({evictedCount})
          </button>
        ) : null}
      </div>
    </div>
  );
}
