import type { ChatMessage } from "../types";

/**
 * Artifacts are renderable blocks detected in an assistant reply. Detection is
 * deliberately pure so it can run on every streamed delta inside a memo and be
 * covered by Vitest without a DOM.
 */
export type ArtifactType =
  | "html"
  | "svg"
  | "mermaid"
  | "markdown"
  | "json"
  | "csv";

export type ArtifactSource = "tag" | "fence";

export interface Artifact {
  /** Stable within the chat while the message exists. */
  id: string;
  type: ArtifactType;
  title: string;
  content: string;
  /** Fence label or the declared tag type. */
  language: string;
  source: ArtifactSource;
  /** Zero-based order inside the message. */
  index: number;
  messageId: string;
  /** 1-based version of this artifact's group. */
  version: number;
  /** Identity shared by every revision of the same artifact. */
  groupKey: string;
  /** Id of the first version in the group. */
  rootId: string;
  /** Id of the previous version, null for the first one. */
  previousId: string | null;
  lineCount: number;
  /** The closing fence or tag had not arrived when this was parsed. */
  partial: boolean;
  /** The reply that contains this artifact is still streaming. */
  isStreaming: boolean;
  /** Character range in the raw message, including the fence or tag. */
  start: number;
  end: number;
  /** Present only for `<artifact>` tags. */
  declaredId?: string | null;
}

export interface ArtifactGroup {
  rootId: string;
  key: string;
  type: ArtifactType;
  title: string;
  /** Oldest first; the last entry is the newest revision. */
  versions: Artifact[];
  latest: Artifact;
}

export interface ChatArtifactIndex {
  groups: ArtifactGroup[];
  /** Versioned artifacts keyed by the message that contains them. */
  byMessage: Record<string, Artifact[]>;
  /** Newest artifact in the active branch, or null when there is none. */
  latest: Artifact | null;
}

export interface ExtractArtifactsOptions {
  messageId?: string;
  /** Automatic fenced blocks must reach this many content lines. */
  minLines?: number;
  /** True while this message is the one still arriving over SSE. */
  isStreaming?: boolean;
  /** Detect explicit `<artifact …>…</artifact>` blocks. */
  explicitTags?: boolean;
  /** Detect substantial fenced code blocks. */
  autoDetect?: boolean;
}

export const DEFAULT_ARTIFACT_MIN_LINES = 12;

const TYPE_LABELS: Record<ArtifactType, string> = {
  html: "HTML",
  svg: "SVG",
  mermaid: "Mermaid",
  markdown: "Markdown",
  json: "JSON",
  csv: "CSV",
};

const TYPE_EXTENSIONS: Record<ArtifactType, string> = {
  html: "html",
  svg: "svg",
  mermaid: "mmd",
  markdown: "md",
  json: "json",
  csv: "csv",
};

const TYPE_MIME: Record<ArtifactType, string> = {
  html: "text/html;charset=utf-8",
  svg: "image/svg+xml;charset=utf-8",
  mermaid: "text/plain;charset=utf-8",
  markdown: "text/markdown;charset=utf-8",
  json: "application/json;charset=utf-8",
  csv: "text/csv;charset=utf-8",
};

const LANGUAGE_TYPES: Record<string, ArtifactType> = {
  html: "html",
  htm: "html",
  xhtml: "html",
  svg: "svg",
  "image/svg+xml": "svg",
  mermaid: "mermaid",
  mmd: "mermaid",
  markdown: "markdown",
  md: "markdown",
  mdx: "markdown",
  json: "json",
  jsonc: "json",
  geojson: "json",
  csv: "csv",
  tsv: "csv",
};

const FALLBACK_TITLES: Record<ArtifactType, string> = {
  html: "HTML document",
  svg: "SVG graphic",
  mermaid: "Mermaid diagram",
  markdown: "Markdown document",
  json: "JSON data",
  csv: "CSV table",
};

export function artifactTypeLabel(type: ArtifactType): string {
  return TYPE_LABELS[type];
}

export function artifactExtension(type: ArtifactType): string {
  return TYPE_EXTENSIONS[type];
}

export function artifactMime(type: ArtifactType): string {
  return TYPE_MIME[type];
}

/** Lowercase slug safe for a filename and stable across streamed deltas. */
export function slugifyArtifactTitle(title: string): string {
  return title
    .toLowerCase()
    .replace(/\.[a-z0-9]{1,8}$/i, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 72);
}

export function artifactFilename(artifact: Artifact): string {
  const base = slugifyArtifactTitle(artifact.title) || artifact.type;
  return `${base}.${artifactExtension(artifact.type)}`;
}

/**
 * Normalize a fence language or an explicit `type` attribute. Unknown values
 * are inferred from the body so an ordinary ```xml block that contains an SVG
 * still previews.
 */
export function artifactTypeFor(
  language: string,
  content: string,
): ArtifactType | null {
  const normalized = language.trim().toLowerCase().split(/\s+/)[0] ?? "";
  const known = LANGUAGE_TYPES[normalized];
  if (known) return known;
  const body = content.trim();
  if (!body) return null;
  if (/^<svg[\s>]/i.test(body)) return "svg";
  if (/^<!doctype\s+html/i.test(body) || /^<html[\s>]/i.test(body)) {
    return "html";
  }
  if (normalized === "xml" && /^<svg[\s>]/i.test(body)) return "svg";
  if (body.startsWith("{") || body.startsWith("[")) {
    try {
      JSON.parse(body);
      return "json";
    } catch {
      // Not valid JSON yet; during streaming it may still become valid.
    }
  }
  return null;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

interface LineInfo {
  text: string;
  start: number;
  end: number;
}

function lineTable(text: string): LineInfo[] {
  const lines: LineInfo[] = [];
  let start = 0;
  for (const value of text.split("\n")) {
    lines.push({ text: value, start, end: start + value.length });
    start += value.length + 1;
  }
  return lines;
}

interface RawBlock {
  type: ArtifactType | null;
  title: string | null;
  declaredId: string | null;
  language: string;
  content: string;
  start: number;
  end: number;
  partial: boolean;
  source: ArtifactSource;
}

function parseAttributes(source: string): Record<string, string> {
  const attributes: Record<string, string> = {};
  const pattern =
    /([A-Za-z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(source))) {
    const key = match[1].toLowerCase();
    const value = match[2] ?? match[3] ?? match[4] ?? "";
    attributes[key] = value.trim();
  }
  return attributes;
}

interface FenceInfo {
  language: string;
  title: string | null;
}

function parseFenceInfo(info: string): FenceInfo {
  const trimmed = info.trim();
  if (!trimmed) return { language: "", title: null };
  const titleMatch = /title\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s]+))/i.exec(
    trimmed,
  );
  const first = trimmed.split(/\s+/)[0] ?? "";
  return {
    language: first.replace(/^[{\[]/, "").toLowerCase(),
    title: titleMatch
      ? (titleMatch[1] ?? titleMatch[2] ?? titleMatch[3] ?? "").trim()
      : null,
  };
}

/**
 * An explicit artifact may wrap its body in a fence for readability. Remove
 * exactly one outer fence, or just the opening line while it is streaming.
 */
function stripFenceWrapper(body: string): {
  content: string;
  language: string;
  title: string | null;
} {
  const lines = body.split("\n");
  let first = 0;
  let last = lines.length - 1;
  while (first <= last && lines[first].trim() === "") first += 1;
  while (last >= first && lines[last].trim() === "") last -= 1;
  const opening = first <= last ? /^\s*(`{3,}|~{3,})(.*)$/.exec(lines[first]) : null;
  if (!opening) {
    return { content: body.replace(/^\n+|\n+$/g, ""), language: "", title: null };
  }
  const marker = opening[1];
  const info = parseFenceInfo(opening[2] ?? "");
  let ending = last;
  if (
    ending > first &&
    new RegExp(`^\\s*${escapeRegExp(marker[0])}{${marker.length},}\\s*$`).test(
      lines[ending],
    )
  ) {
    ending -= 1;
  }
  return {
    content: lines.slice(first + 1, ending + 1).join("\n"),
    language: info.language,
    title: info.title,
  };
}

function deriveTitle(type: ArtifactType, content: string): string {
  if (type === "html") {
    const title = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(content);
    if (title?.[1]?.trim()) return htmlToText(title[1]).slice(0, 80);
    const heading = /<h1[^>]*>([\s\S]*?)<\/h1>/i.exec(content);
    if (heading?.[1]?.trim()) return htmlToText(heading[1]).slice(0, 80);
  }
  if (type === "svg") {
    const title = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(content);
    if (title?.[1]?.trim()) return title[1].trim().slice(0, 80);
  }
  if (type === "markdown") {
    const heading = /^\s{0,3}#{1,3}\s+(.+)$/m.exec(content);
    if (heading?.[1]?.trim()) return heading[1].trim().slice(0, 80);
  }
  if (type === "json") {
    const name = /"name"\s*:\s*"([^"]{1,60})"/.exec(content);
    if (name?.[1]?.trim()) return name[1].trim();
  }
  if (type === "csv") {
    const firstLine = content.split("\n", 1)[0] ?? "";
    if (firstLine.trim()) return "CSV · ".concat(firstLine.trim().slice(0, 56));
  }
  return FALLBACK_TITLES[type];
}

function htmlToText(value: string): string {
  return value
    .replace(/<[^>]*>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/\s+/g, " ")
    .trim();
}

function findExplicitBlocks(
  content: string,
  streaming: boolean,
  fenceRanges: Array<{ start: number; end: number }>,
): RawBlock[] {
  const blocks: RawBlock[] = [];
  const opening = /<artifact\b([^>]*)>/gi;
  const closing = /<\/artifact\s*>/gi;
  let match: RegExpExecArray | null;
  while ((match = opening.exec(content))) {
    const tagStart = match.index;
    if (
      fenceRanges.some(
        (range) => tagStart >= range.start && tagStart < range.end,
      )
    ) {
      // This is example text inside a code fence, not a real artifact. Keep
      // scanning after the tag instead of jumping to the next closing tag,
      // which may belong to a genuine artifact further down the reply.
      continue;
    }
    const attributes = parseAttributes(match[1] ?? "");
    const bodyStart = opening.lastIndex;
    closing.lastIndex = bodyStart;
    const close = closing.exec(content);
    const partial = close === null;
    if (partial && !streaming) {
      // A finished reply with an unterminated tag is malformed. Do not invent
      // an artifact from it; the markdown renderer will show it as text.
      break;
    }
    const bodyEnd = close ? close.index : content.length;
    const body = content.slice(bodyStart, bodyEnd);
    if (!body.trim()) {
      opening.lastIndex = close ? closing.lastIndex : content.length;
      if (!close) break;
      continue;
    }
    const stripped = stripFenceWrapper(body);
    if (!stripped.content.trim()) {
      opening.lastIndex = close ? closing.lastIndex : content.length;
      if (!close) break;
      continue;
    }
    const declared = (attributes.type ?? "").trim().toLowerCase();
    const type =
      artifactTypeFor(declared, stripped.content) ??
      artifactTypeFor(stripped.language, stripped.content);
    if (type) {
      blocks.push({
        type,
        title: attributes.title ?? stripped.title ?? null,
        declaredId: attributes.id ?? null,
        language: declared || stripped.language || type,
        content: stripped.content,
        start: tagStart,
        end: close ? closing.lastIndex : content.length,
        partial,
        source: "tag",
      });
    }
    opening.lastIndex = close ? closing.lastIndex : content.length;
    if (!close) break;
  }
  return blocks;
}

/** Every fenced block, including partial ones, used to keep tags out of code. */
function fenceRanges(content: string): Array<{ start: number; end: number }> {
  const lines = lineTable(content);
  const ranges: Array<{ start: number; end: number }> = [];
  let index = 0;
  while (index < lines.length) {
    const opening = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(lines[index].text);
    if (!opening) {
      index += 1;
      continue;
    }
    const marker = opening[1];
    const closingPattern = new RegExp(
      `^ {0,3}${escapeRegExp(marker[0])}{${marker.length},}\\s*$`,
    );
    let endIndex = index + 1;
    while (endIndex < lines.length && !closingPattern.test(lines[endIndex].text)) {
      endIndex += 1;
    }
    const closed = endIndex < lines.length;
    ranges.push({
      start: lines[index].start,
      end: closed ? lines[endIndex].end : content.length,
    });
    index = closed ? endIndex + 1 : lines.length;
  }
  return ranges;
}

function findFenceBlocks(
  content: string,
  streaming: boolean,
  excluded: RawBlock[],
  minLines: number,
): RawBlock[] {
  const blocks: RawBlock[] = [];
  const lines = lineTable(content);
  const inExcluded = (position: number) =>
    excluded.some((block) => position >= block.start && position < block.end);

  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    const opening = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line.text);
    if (!opening) {
      index += 1;
      continue;
    }
    const marker = opening[1];
    const char = marker[0];
    const closingPattern = new RegExp(
      `^ {0,3}${escapeRegExp(char)}{${marker.length},}\\s*$`,
    );
    let endIndex = index + 1;
    let closedAt = -1;
    while (endIndex < lines.length) {
      if (closingPattern.test(lines[endIndex].text)) {
        closedAt = endIndex;
        break;
      }
      endIndex += 1;
    }

    if (!inExcluded(line.start)) {
      const bodyLines = lines.slice(
        index + 1,
        closedAt >= 0 ? closedAt : lines.length,
      );
      const body = bodyLines.map((entry) => entry.text).join("\n");
      const info = parseFenceInfo(opening[2] ?? "");
      const type = artifactTypeFor(info.language, body);
      const contentLines = body.length ? body.split("\n").length : 0;
      const partial = closedAt < 0;
      const substantial =
        contentLines >= minLines ||
        (streaming && partial && contentLines >= 1);
      if (type && body.trim() && substantial) {
        blocks.push({
          type,
          title: info.title,
          declaredId: null,
          language: info.language || type,
          content: body,
          start: line.start,
          end:
            closedAt >= 0
              ? lines[closedAt].end
              : content.length,
          partial,
          source: "fence",
        });
      }
    }

    index = closedAt >= 0 ? closedAt + 1 : lines.length;
  }

  return blocks;
}

/**
 * Extract every renderable artifact from one message. The order is the order
 * the reader sees them, and character offsets let the chat bubble replace the
 * original block with a compact card.
 */
export function extractArtifacts(
  content: string,
  options: ExtractArtifactsOptions = {},
): Artifact[] {
  if (!content) return [];
  const messageId = options.messageId ?? "message";
  const minLines = Math.max(
    1,
    Math.floor(options.minLines ?? DEFAULT_ARTIFACT_MIN_LINES),
  );
  const streaming = Boolean(options.isStreaming);
  const explicitTags = options.explicitTags ?? true;
  const autoDetect = options.autoDetect ?? true;

  const tagBlocks = explicitTags
    ? findExplicitBlocks(content, streaming, fenceRanges(content))
    : [];
  const fenceBlocks = autoDetect
    ? findFenceBlocks(content, streaming, tagBlocks, minLines)
    : [];
  const raw = [...tagBlocks, ...fenceBlocks]
    .filter(
      (block): block is RawBlock & { type: ArtifactType } => block.type !== null,
    )
    .sort((left, right) => left.start - right.start);

  const occurrences = new Map<string, number>();
  return raw.map((block, index) => {
    const baseKey = block.declaredId
      ? `id:${block.declaredId}`
      : block.source === "tag"
        ? `${block.type}:${slugifyArtifactTitle(block.title ?? FALLBACK_TITLES[block.type])}`
        : `${block.type}:auto`;
    const seen = occurrences.get(baseKey) ?? 0;
    occurrences.set(baseKey, seen + 1);
    const groupKey = seen > 0 ? `${baseKey}#${seen + 1}` : baseKey;
    const title =
      block.title?.trim() || deriveTitle(block.type, block.content);
    const id = `${messageId}:${index}`;
    return {
      id,
      type: block.type,
      title,
      content: block.content.replace(/\n$/, ""),
      language: block.language,
      source: block.source,
      index,
      messageId,
      version: 1,
      groupKey,
      rootId: id,
      previousId: null,
      lineCount: block.content.length
        ? block.content.replace(/\n$/, "").split("\n").length
        : 0,
      partial: block.partial,
      isStreaming: streaming && block.partial,
      start: block.start,
      end: block.end,
      declaredId: block.declaredId,
    } satisfies Artifact;
  });
}

/**
 * Build the chat-wide artifact timeline. Versions are linked across turns by
 * identity: an `<artifact id>` keeps its id, and automatic fences of the same
 * type are revisions of one canvas (a later "make the button blue" reply is
 * v2). Multiple artifacts of one type in the *same* message are kept apart.
 */
export function collectChatArtifacts(
  messages: ChatMessage[],
  options: ExtractArtifactsOptions & {
    /** The message currently receiving `chat://delta`. */
    streamingMessageId?: string | null;
  } = {},
): ChatArtifactIndex {
  const groups = new Map<string, ArtifactGroup>();
  const byMessage: Record<string, Artifact[]> = {};
  const ordered: ArtifactGroup[] = [];
  let latest: Artifact | null = null;

  for (const message of messages) {
    if (message.role !== "assistant") continue;
    const content = typeof message.content === "string" ? message.content : "";
    if (!content) continue;
    const streaming = message.id === options.streamingMessageId;
    const extracted = extractArtifacts(content, {
      ...options,
      messageId: message.id,
      isStreaming: streaming,
    });
    if (!extracted.length) continue;
    const list: Artifact[] = [];
    for (const artifact of extracted) {
      let group = groups.get(artifact.groupKey);
      if (!group) {
        group = {
          rootId: artifact.id,
          key: artifact.groupKey,
          type: artifact.type,
          title: artifact.title,
          versions: [],
          latest: artifact,
        };
        groups.set(artifact.groupKey, group);
        ordered.push(group);
      }
      const previous = group.versions[group.versions.length - 1] ?? null;
      const versioned: Artifact = {
        ...artifact,
        version: group.versions.length + 1,
        rootId: group.rootId,
        previousId: previous?.id ?? null,
      };
      group.versions.push(versioned);
      group.latest = versioned;
      group.type = versioned.type;
      group.title = versioned.title;
      list.push(versioned);
      latest = versioned;
    }
    byMessage[message.id] = list;
  }

  return { groups: ordered, byMessage, latest };
}

export function artifactGroupFor(
  index: ChatArtifactIndex,
  rootId: string | null | undefined,
): ArtifactGroup | null {
  if (!rootId) return null;
  return index.groups.find((group) => group.rootId === rootId) ?? null;
}

export function latestArtifactGroup(
  index: ChatArtifactIndex,
): ArtifactGroup | null {
  if (!index.latest) return null;
  return artifactGroupFor(index, index.latest.rootId);
}

export type ArtifactSegment =
  | { kind: "markdown"; content: string }
  | { kind: "artifact"; artifact: Artifact };

/**
 * Split a message into markdown and artifact-card segments. Anything outside a
 * detected artifact stays markdown; the artifact's raw fence or tag is
 * replaced by a compact card in the bubble.
 */
export function splitArtifactSegments(
  content: string,
  artifacts: Artifact[],
): ArtifactSegment[] {
  if (!artifacts.length) return [{ kind: "markdown", content }];
  const sorted = [...artifacts]
    .filter((artifact) => artifact.start >= 0 && artifact.end <= content.length)
    .sort((left, right) => left.start - right.start);
  const segments: ArtifactSegment[] = [];
  let cursor = 0;
  for (const artifact of sorted) {
    if (artifact.start < cursor) continue;
    if (artifact.start > cursor) {
      segments.push({
        kind: "markdown",
        content: content.slice(cursor, artifact.start),
      });
    }
    segments.push({ kind: "artifact", artifact });
    cursor = Math.max(cursor, artifact.end);
  }
  if (cursor < content.length) {
    segments.push({ kind: "markdown", content: content.slice(cursor) });
  }
  return segments;
}

export type DiffKind = "same" | "add" | "remove";

export interface DiffLine {
  kind: DiffKind;
  text: string;
  oldLine: number | null;
  newLine: number | null;
}

function coarseDiff(before: string[], after: string[]): DiffLine[] {
  let prefix = 0;
  while (
    prefix < before.length &&
    prefix < after.length &&
    before[prefix] === after[prefix]
  ) {
    prefix += 1;
  }
  let suffix = 0;
  while (
    suffix < before.length - prefix &&
    suffix < after.length - prefix &&
    before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
  ) {
    suffix += 1;
  }
  const lines: DiffLine[] = [];
  for (let i = 0; i < prefix; i += 1) {
    lines.push({
      kind: "same",
      text: before[i],
      oldLine: i + 1,
      newLine: i + 1,
    });
  }
  const removedEnd = before.length - suffix;
  for (let i = prefix; i < removedEnd; i += 1) {
    lines.push({ kind: "remove", text: before[i], oldLine: i + 1, newLine: null });
  }
  const addedEnd = after.length - suffix;
  for (let i = prefix; i < addedEnd; i += 1) {
    lines.push({ kind: "add", text: after[i], oldLine: null, newLine: i + 1 });
  }
  for (let i = 0; i < suffix; i += 1) {
    const oldIndex = before.length - suffix + i;
    const newIndex = after.length - suffix + i;
    lines.push({
      kind: "same",
      text: before[oldIndex],
      oldLine: oldIndex + 1,
      newLine: newIndex + 1,
    });
  }
  return lines;
}

/**
 * Line diff for the Diff tab. Uses an LCS table for normal artifacts and a
 * prefix/suffix fallback for very large revisions so streaming a 5 000-line
 * file cannot allocate an unbounded matrix.
 */
export function diffLines(before: string, after: string): DiffLine[] {
  const oldLines = before.replace(/\r\n/g, "\n").split("\n");
  const newLines = after.replace(/\r\n/g, "\n").split("\n");
  if (oldLines.length * newLines.length > 1_200_000) {
    return coarseDiff(oldLines, newLines);
  }
  const width = newLines.length + 1;
  const table = new Uint32Array((oldLines.length + 1) * width);
  for (let i = oldLines.length - 1; i >= 0; i -= 1) {
    for (let j = newLines.length - 1; j >= 0; j -= 1) {
      table[i * width + j] =
        oldLines[i] === newLines[j]
          ? table[(i + 1) * width + (j + 1)] + 1
          : Math.max(
              table[(i + 1) * width + j],
              table[i * width + (j + 1)],
            );
    }
  }
  const lines: DiffLine[] = [];
  let i = 0;
  let j = 0;
  while (i < oldLines.length && j < newLines.length) {
    if (oldLines[i] === newLines[j]) {
      lines.push({
        kind: "same",
        text: oldLines[i],
        oldLine: i + 1,
        newLine: j + 1,
      });
      i += 1;
      j += 1;
    } else if (
      table[(i + 1) * width + j] >= table[i * width + (j + 1)]
    ) {
      lines.push({
        kind: "remove",
        text: oldLines[i],
        oldLine: i + 1,
        newLine: null,
      });
      i += 1;
    } else {
      lines.push({
        kind: "add",
        text: newLines[j],
        oldLine: null,
        newLine: j + 1,
      });
      j += 1;
    }
  }
  while (i < oldLines.length) {
    lines.push({
      kind: "remove",
      text: oldLines[i],
      oldLine: i + 1,
      newLine: null,
    });
    i += 1;
  }
  while (j < newLines.length) {
    lines.push({
      kind: "add",
      text: newLines[j],
      oldLine: null,
      newLine: j + 1,
    });
    j += 1;
  }
  return lines;
}

export interface DiffStats {
  added: number;
  removed: number;
}

export function diffStats(lines: DiffLine[]): DiffStats {
  return lines.reduce<DiffStats>(
    (stats, line) => {
      if (line.kind === "add") stats.added += 1;
      if (line.kind === "remove") stats.removed += 1;
      return stats;
    },
    { added: 0, removed: 0 },
  );
}

export interface CsvTable {
  headers: string[];
  rows: string[][];
  delimiter: string;
  truncated: boolean;
}

function detectDelimiter(text: string): string {
  const firstLine = text.split(/\r?\n/, 1)[0] ?? "";
  const candidates = [",", ";", "\t"];
  let best = ",";
  let bestCount = 0;
  for (const candidate of candidates) {
    let count = 0;
    let quoted = false;
    for (let i = 0; i < firstLine.length; i += 1) {
      const char = firstLine[i];
      if (char === '"') quoted = !quoted;
      else if (!quoted && char === candidate) count += 1;
    }
    if (count > bestCount) {
      best = candidate;
      bestCount = count;
    }
  }
  return best;
}

/** RFC 4180-style CSV/TSV parser; incomplete quoting is tolerated. */
export function parseCsv(text: string, maxRows = 5_000): CsvTable {
  const clean = text.replace(/^\uFEFF/, "");
  const delimiter = detectDelimiter(clean);
  const parsed: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  let truncated = false;

  for (let i = 0; i < clean.length; i += 1) {
    const char = clean[i];
    if (quoted) {
      if (char === '"') {
        if (clean[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        field += char;
      }
      continue;
    }
    if (char === '"' && field.length === 0) {
      quoted = true;
    } else if (char === delimiter) {
      row.push(field);
      field = "";
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && clean[i + 1] === "\n") i += 1;
      row.push(field);
      field = "";
      parsed.push(row);
      row = [];
      if (parsed.length > maxRows + 1) {
        truncated = true;
        break;
      }
    } else {
      field += char;
    }
  }
  if (!truncated && (field.length > 0 || row.length > 0)) {
    row.push(field);
    parsed.push(row);
  }

  const nonEmpty = parsed.filter((entry) =>
    entry.some((cell) => cell.trim() !== ""),
  );
  const headerRow = nonEmpty[0] ?? [];
  const width = Math.max(
    headerRow.length,
    ...nonEmpty.slice(1).map((entry) => entry.length),
    0,
  );
  const headers = Array.from({ length: width }, (_, index) => {
    const value = headerRow[index]?.trim();
    return value || `Column ${index + 1}`;
  });
  const rows = nonEmpty
    .slice(1)
    .map((entry) =>
      Array.from({ length: width }, (_, index) => entry[index] ?? ""),
    );
  return { headers, rows, delimiter, truncated };
}

export type JsonParseResult =
  | { ok: true; value: unknown }
  | { ok: false; error: string };

export function parseJsonValue(text: string): JsonParseResult {
  try {
    return { ok: true, value: JSON.parse(text) as unknown };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Invalid JSON",
    };
  }
}

/**
 * Mermaid output and model-generated SVG are both untrusted. The iframe CSP is
 * the real boundary, but stripping active content here keeps the inline SVG
 * viewer and the PNG export from ever carrying a script or event handler.
 */
function scrubSvgUrls(markup: string): string {
  return markup
    .replace(/@import\b[^;]*(?:;|$)/gi, "")
    .replace(
      /url\(\s*(?:(["'])(.*?)\1|([^)]*?))\s*\)/gi,
      (_match, _quote: string | undefined, quoted: string | undefined, bare: string | undefined) => {
        const target = (quoted ?? bare ?? "").trim();
        // Fragments and embedded images are self-contained. Every other URL
        // could make a network request when the SVG is rasterised for PNG.
        return /^(?:#|data:image\/)/i.test(target) ? `url("${target}")` : "none";
      },
    );
}

export function sanitizeSvg(markup: string): string {
  return scrubSvgUrls(markup)
    .replace(/<\s*script\b[^>]*>[\s\S]*?<\s*\/\s*script\s*>/gi, "")
    .replace(/<\s*script\b[^>]*\/\s*>/gi, "")
    .replace(/<\s*foreignObject\b[^>]*>[\s\S]*?<\s*\/\s*foreignObject\s*>/gi, "")
    .replace(/<\s*iframe\b[^>]*>[\s\S]*?<\s*\/\s*iframe\s*>/gi, "")
    .replace(/<\s*object\b[^>]*>[\s\S]*?<\s*\/\s*object\s*>/gi, "")
    .replace(/<\s*embed\b[^>]*\/?\s*>/gi, "")
    .replace(/<\s*link\b[^>]*\/?\s*>/gi, "")
    .replace(/\s+on[a-z]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi, "")
    .replace(
      /\s(?:href|xlink:href|src)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi,
      (match, doubleQuoted: string | undefined, singleQuoted: string | undefined, bare: string | undefined) => {
        const target = (doubleQuoted ?? singleQuoted ?? bare ?? "").trim();
        return /^(?:#|data:image\/)/i.test(target) ? match : ' href="#"';
      },
    )
    .trim();
}
