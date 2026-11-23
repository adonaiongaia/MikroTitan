import { useMemo, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  ChevronDown,
  ChevronRight,
  Loader2,
  Search,
} from "lucide-react";

import {
  parseCsv,
  parseJsonValue,
  type Artifact,
} from "../../lib/artifacts";
import { cn } from "../../lib/cn";

interface DataTableViewerProps {
  artifact: Artifact;
}

export function DataTableViewer({ artifact }: DataTableViewerProps) {
  if (artifact.type === "csv") {
    return <CsvViewer artifact={artifact} />;
  }
  if (artifact.type === "json") {
    return <JsonViewer artifact={artifact} />;
  }
  return null;
}

function CsvViewer({ artifact }: { artifact: Artifact }) {
  const table = useMemo(
    () => parseCsv(artifact.content, 5_000),
    [artifact.content],
  );
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ column: number; direction: 1 | -1 } | null>(
    null,
  );

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const rows = needle
      ? table.rows.filter((row) =>
          row.some((cell) => cell.toLowerCase().includes(needle)),
        )
      : table.rows;
    if (!sort) return rows;
    return [...rows].sort((left, right) => {
      const a = left[sort.column] ?? "";
      const b = right[sort.column] ?? "";
      const numeric = Number(a) - Number(b);
      if (a.trim() !== "" && b.trim() !== "" && !Number.isNaN(numeric)) {
        return numeric * sort.direction;
      }
      return a.localeCompare(b, undefined, { numeric: true }) * sort.direction;
    });
  }, [query, sort, table.rows]);

  const visible = filtered.slice(0, 1_000);

  if (!table.headers.length) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
        {artifact.isStreaming ? (
          <Loader2 className="h-4 w-4 animate-spin text-slate-600" />
        ) : null}
        <p className="text-[11px] text-slate-500">
          {artifact.isStreaming
            ? "Waiting for the first CSV row…"
            : "This block has no tabular rows yet."}
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-white/5 px-2 py-1.5">
        <label className="flex min-w-0 flex-1 items-center gap-1.5 rounded-lg border border-white/10 bg-ink-950/50 px-2">
          <Search className="h-3 w-3 shrink-0 text-slate-600" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter rows…"
            className="h-7 min-w-0 flex-1 bg-transparent text-[10.5px] text-slate-200 outline-none placeholder:text-slate-600"
          />
        </label>
        <span className="shrink-0 font-mono text-[9px] text-slate-600">
          {filtered.length.toLocaleString()}
          {filtered.length !== table.rows.length
            ? `/${table.rows.length.toLocaleString()}`
            : ""}{" "}
          rows
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        <table className="w-full border-collapse text-left text-[10.5px]">
          <thead className="sticky top-0 z-10 bg-ink-900">
            <tr>
              {table.headers.map((header, index) => (
                <th
                  key={`${header}-${index}`}
                  className="border-b border-white/10 px-2 py-1.5 font-medium text-slate-400"
                >
                  <button
                    type="button"
                    onClick={() =>
                      setSort((current) =>
                        current?.column === index
                          ? current.direction === 1
                            ? { column: index, direction: -1 }
                            : null
                          : { column: index, direction: 1 },
                      )
                    }
                    className="inline-flex max-w-full items-center gap-1 truncate transition hover:text-slate-100"
                    title={`Sort by ${header}`}
                  >
                    <span className="truncate">{header}</span>
                    {sort?.column === index ? (
                      sort.direction === 1 ? (
                        <ArrowUp className="h-2.5 w-2.5 shrink-0 text-acid-400" />
                      ) : (
                        <ArrowDown className="h-2.5 w-2.5 shrink-0 text-acid-400" />
                      )
                    ) : null}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className="odd:bg-white/[0.015] hover:bg-white/[0.04]"
              >
                {table.headers.map((_, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="max-w-[260px] truncate border-b border-white/5 px-2 py-1 font-mono text-slate-300"
                    title={row[cellIndex] ?? ""}
                  >
                    {row[cellIndex] || (
                      <span className="text-slate-700">—</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
            {!visible.length ? (
              <tr>
                <td
                  colSpan={table.headers.length}
                  className="px-3 py-8 text-center text-[10.5px] text-slate-600"
                >
                  No row matches this filter.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
        {filtered.length > visible.length || table.truncated ? (
          <p className="px-3 py-2 text-[9px] text-slate-600">
            Showing the first {visible.length.toLocaleString()} rows
            {table.truncated ? " (the file is longer)" : ""}.
          </p>
        ) : null}
      </div>
    </div>
  );
}

function JsonViewer({ artifact }: { artifact: Artifact }) {
  const parsed = useMemo(
    () => parseJsonValue(artifact.content),
    [artifact.content],
  );

  if (!parsed.ok) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
        {artifact.isStreaming ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin text-slate-600" />
            <p className="text-[11px] text-slate-500">
              Waiting for a complete JSON value…
            </p>
            <p className="max-w-sm font-mono text-[9px] leading-4 text-slate-700">
              {artifact.content.slice(-180) || "The block is still empty."}
            </p>
          </>
        ) : (
          <>
            <p className="text-[11px] font-medium text-rose-300/80">
              This block is not valid JSON
            </p>
            <p className="max-w-sm font-mono text-[9.5px] leading-4 text-slate-500">
              {parsed.error}
            </p>
            <p className="text-[9px] text-slate-600">
              The raw text is available in the Code tab.
            </p>
          </>
        )}
      </div>
    );
  }

  const root = parsed.value;
  if (root === null || typeof root !== "object") {
    return (
      <div className="p-4 font-mono text-[11px] text-slate-300">
        {formatPrimitive(root)}
      </div>
    );
  }

  const isArrayRoot = Array.isArray(root);
  const rootPreview = previewEntries(root, 1_000);
  const rootSize = rootPreview.truncated
    ? "1,000+"
    : rootPreview.entries.length.toLocaleString();

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-white/5 px-2 py-1.5">
        <span className="rounded border border-white/10 bg-white/[0.03] px-1 py-px text-[8px] uppercase tracking-wider text-slate-500">
          {isArrayRoot ? "Array" : "Object"}
        </span>
        <span className="font-mono text-[9px] text-slate-600">
          {rootSize}
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-2 font-mono text-[10.5px]">
        {rootPreview.entries.map(([label, value]) => (
          <JsonNode key={label} label={label} value={value} depth={0} />
        ))}
        {rootPreview.truncated ? (
          <p className="px-2 py-2 text-[9px] text-slate-600">
            Showing the first 1,000 entries; the raw value is in the Code tab.
          </p>
        ) : null}
      </div>
    </div>
  );
}

function JsonNode({
  label,
  value,
  depth,
  isArrayIndex = false,
}: {
  label: string;
  value: unknown;
  depth: number;
  isArrayIndex?: boolean;
}) {
  const isArrayValue = Array.isArray(value);
  const expandable =
    value !== null &&
    typeof value === "object" &&
    (isArrayValue ? value.length > 0 : hasEnumerableEntry(value as object));
  const [open, setOpen] = useState(depth < 2);
  const childPreview =
    expandable && open ? previewEntries(value as object, 500) : null;
  const childCount = childPreview
    ? childPreview.truncated
      ? "500+"
      : String(childPreview.entries.length)
    : "…";

  return (
    <div className={cn(depth > 0 && "ml-3 border-l border-white/5 pl-2")}>
      <div className="flex items-start gap-1 py-0.5">
        {expandable ? (
          <button
            type="button"
            onClick={() => setOpen((current) => !current)}
            className="mt-px shrink-0 text-slate-500 transition hover:text-slate-200"
            aria-label={open ? `Collapse ${label}` : `Expand ${label}`}
          >
            {open ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
          </button>
        ) : (
          <span className="mt-px h-3 w-3 shrink-0" />
        )}
        <span className="min-w-0 break-all">
          <span
            className={cn(
              "text-[10.5px]",
              isArrayIndex ? "text-slate-500" : "text-sky-300/80",
            )}
          >
            {label}
          </span>
          <span className="text-slate-600">: </span>
          {expandable ? (
            <span className="text-slate-500">
              {isArrayValue ? `[${value.length}]` : `{${childCount}}`}
            </span>
          ) : (
            formatPrimitive(value)
          )}
        </span>
      </div>

      {childPreview ? (
        <div>
          {childPreview.entries.map(([childLabel, childValue]) => (
            <JsonNode
              key={childLabel}
              label={childLabel}
              value={childValue}
              depth={depth + 1}
              isArrayIndex={Array.isArray(value)}
            />
          ))}
          {childPreview.truncated ? (
            <p className="ml-4 py-1 text-[9px] text-slate-600">
              … more entries not shown
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

interface JsonPreview {
  entries: Array<readonly [string, unknown]>;
  truncated: boolean;
}

/**
 * Read at most `limit + 1` entries. This avoids materializing every key/value
 * pair in a large object before taking the visible slice, and the one-entry
 * probe is enough to know whether the list was truncated.
 */
function previewEntries(value: object, limit: number): JsonPreview {
  const entries: Array<readonly [string, unknown]> = [];
  if (Array.isArray(value)) {
    const stop = Math.min(value.length, limit + 1);
    for (let index = 0; index < stop; index += 1) {
      entries.push([String(index), value[index]]);
    }
  } else {
    const record = value as Record<string, unknown>;
    for (const key in record) {
      if (!Object.prototype.hasOwnProperty.call(record, key)) continue;
      entries.push([key, record[key]]);
      if (entries.length > limit) break;
    }
  }

  if (entries.length > limit) {
    entries.length = limit;
    return { entries, truncated: true };
  }
  return { entries, truncated: false };
}

function hasEnumerableEntry(value: object): boolean {
  for (const key in value) {
    if (Object.prototype.hasOwnProperty.call(value, key)) return true;
  }
  return false;
}

function formatPrimitive(value: unknown) {
  if (value === null) return <span className="text-slate-500">null</span>;
  if (value === undefined) {
    return <span className="text-slate-600">undefined</span>;
  }
  if (typeof value === "string") {
    return (
      <span className="break-all text-emerald-300/80">"{value}"</span>
    );
  }
  if (typeof value === "number") {
    return <span className="text-amber-200/90">{String(value)}</span>;
  }
  if (typeof value === "boolean") {
    return <span className="text-violet-300/90">{String(value)}</span>;
  }
  return <span className="text-slate-400">{String(value)}</span>;
}
