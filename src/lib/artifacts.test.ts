import { describe, expect, it } from "vitest";

import {
  artifactFilename,
  artifactGroupFor,
  collectChatArtifacts,
  diffLines,
  extractArtifacts,
  parseCsv,
  parseJsonValue,
  sanitizeSvg,
  splitArtifactSegments,
  type Artifact,
} from "./artifacts";
import type { ChatMessage } from "../types";

const HTML = Array.from(
  { length: 14 },
  (_, index) => `<p>Line ${index + 1}</p>`,
).join("\n");

function assistant(id: string, content: string): ChatMessage {
  return {
    id,
    role: "assistant",
    content,
    reasoning: "",
    created_at: 0,
  };
}

function user(id: string, content: string): ChatMessage {
  return {
    id,
    role: "user",
    content,
    reasoning: "",
    created_at: 0,
  };
}

describe("extractArtifacts", () => {
  it("detects a substantial fenced HTML block", () => {
    const content = `Here is the page.\n\n\`\`\`html\n${HTML}\n\`\`\`\n\nDone.`;
    const artifacts = extractArtifacts(content, { messageId: "m1" });
    expect(artifacts).toHaveLength(1);
    expect(artifacts[0]).toMatchObject({
      type: "html",
      source: "fence",
      messageId: "m1",
      lineCount: 14,
      partial: false,
      isStreaming: false,
      version: 1,
    });
    expect(artifacts[0].content.startsWith("<p>Line 1</p>")).toBe(true);
    expect(splitArtifactSegments(content, artifacts)).toHaveLength(3);
  });

  it("ignores short fences but respects a custom threshold", () => {
    const content = "```json\n{\n  \"a\": 1\n}\n```";
    expect(extractArtifacts(content, { messageId: "m" })).toHaveLength(0);
    expect(
      extractArtifacts(content, { messageId: "m", minLines: 3 }),
    ).toHaveLength(1);
  });

  it("keeps a partial streaming fence with its streaming flag", () => {
    const content = "```html\n<div>hello</div>\n<span>still arriving";
    const streaming = extractArtifacts(content, {
      messageId: "m",
      isStreaming: true,
    });
    expect(streaming).toHaveLength(1);
    expect(streaming[0]).toMatchObject({
      type: "html",
      partial: true,
      isStreaming: true,
    });
    expect(streaming[0].content).toContain("still arriving");

    const finished = extractArtifacts(content, {
      messageId: "m",
      isStreaming: false,
    });
    expect(finished).toHaveLength(0);
  });

  it("parses explicit artifact tags, including quoted attributes", () => {
    const content = `Intro\n<artifact id="hero" type="html" title="Hero card">\n<div>hi</div>\n</artifact>\nOutro`;
    const artifacts = extractArtifacts(content, { messageId: "m" });
    expect(artifacts).toHaveLength(1);
    expect(artifacts[0]).toMatchObject({
      type: "html",
      title: "Hero card",
      declaredId: "hero",
      source: "tag",
      groupKey: "id:hero",
    });
    const segments = splitArtifactSegments(content, artifacts);
    expect(segments[0]).toEqual({ kind: "markdown", content: "Intro\n" });
    expect(segments[1].kind).toBe("artifact");
    expect(segments[2]).toEqual({ kind: "markdown", content: "\nOutro" });
  });

  it("strips an inner fence from an explicit tag body", () => {
    const content = `<artifact type="mermaid" title="Flow">\n\`\`\`mermaid\ngraph TD;\nA-->B\n\`\`\`\n</artifact>`;
    const artifacts = extractArtifacts(content, { messageId: "m" });
    expect(artifacts).toHaveLength(1);
    expect(artifacts[0].content).toBe("graph TD;\nA-->B");
    expect(artifacts[0].type).toBe("mermaid");
  });

  it("handles an unclosed explicit tag while streaming only", () => {
    const content = `<artifact type="html" title="Live">\n<div>one</div>`;
    expect(
      extractArtifacts(content, { messageId: "m", isStreaming: true })[0],
    ).toMatchObject({ partial: true, isStreaming: true, title: "Live" });
    expect(extractArtifacts(content, { messageId: "m" })).toHaveLength(0);
  });

  it("does not treat artifact examples inside a fence as tags", () => {
    const content = [
      "```markdown",
      "# Example",
      '<artifact type="html" title="sample">',
      "<div>example</div>",
      "</artifact>",
      ...Array.from({ length: 10 }, () => "more prose"),
      "```",
    ].join("\n");
    const artifacts = extractArtifacts(content, { messageId: "m" });
    expect(artifacts).toHaveLength(1);
    expect(artifacts[0].type).toBe("markdown");
  });

  it("finds multiple artifacts in one message without merging them", () => {
    const content = [
      "```html",
      HTML,
      "```",
      "Between the two blocks.",
      "```mermaid",
      "graph TD;",
      "A-->B;",
      ...Array.from({ length: 11 }, () => "B-->C;"),
      "```",
    ].join("\n");
    const artifacts = extractArtifacts(content, { messageId: "m1" });
    expect(artifacts.map((entry) => entry.type)).toEqual(["html", "mermaid"]);
    expect(new Set(artifacts.map((entry) => entry.groupKey)).size).toBe(2);
    expect(artifacts[1].index).toBe(1);
  });

  it("derives a readable title and filename", () => {
    const content = `\`\`\`html\n<html><head><title>Pricing & Plans</title></head><body>${HTML}</body></html>\n\`\`\``;
    const [artifact] = extractArtifacts(content, { messageId: "m" });
    expect(artifact.title).toBe("Pricing & Plans");
    expect(artifactFilename(artifact)).toBe("pricing-plans.html");
  });
});

describe("collectChatArtifacts", () => {
  const v1 = `\`\`\`html\n${HTML}\n\`\`\``;
  const v2 = `\`\`\`html\n${HTML}\n<button style="background: blue">Blue</button>\n\`\`\``;

  it("links revisions across turns as versions", () => {
    const index = collectChatArtifacts([
      user("u1", "Make a landing page"),
      assistant("a1", v1),
      user("u2", "Make the button blue"),
      assistant("a2", v2),
    ]);
    expect(index.groups).toHaveLength(1);
    const group = index.groups[0];
    expect(group.versions).toHaveLength(2);
    expect(group.versions.map((entry) => entry.version)).toEqual([1, 2]);
    expect(group.versions[1].previousId).toBe(group.versions[0].id);
    expect(group.versions[1].rootId).toBe(group.rootId);
    expect(index.byMessage.a1).toHaveLength(1);
    expect(index.byMessage.a2[0].version).toBe(2);
    expect(index.latest?.version).toBe(2);
    expect(artifactGroupFor(index, group.rootId)?.latest.content).toContain(
      "background: blue",
    );
  });

  it("links explicit ids across turns even when titles change", () => {
    const index = collectChatArtifacts([
      assistant(
        "a1",
        `<artifact id="pricing" type="html" title="Pricing">\n<div>one</div>\n</artifact>`,
      ),
      assistant(
        "a2",
        `<artifact id="pricing" type="html" title="Pricing v2">\n<div>two</div>\n</artifact>`,
      ),
    ]);
    expect(index.groups).toHaveLength(1);
    expect(index.groups[0].versions).toHaveLength(2);
    expect(index.groups[0].latest.title).toBe("Pricing v2");
  });

  it("marks only the streaming message's partial artifact", () => {
    const partial = "```html\n<div>partial</div>";
    const index = collectChatArtifacts(
      [assistant("a1", partial), assistant("a2", v1)],
      { streamingMessageId: "a1" },
    );
    expect(index.byMessage.a1[0].isStreaming).toBe(true);
    expect(index.byMessage.a2[0].isStreaming).toBe(false);
  });

  it("shows only artifacts on the active branch's messages", () => {
    const index = collectChatArtifacts([
      assistant("a1", v1),
      assistant("a2", "No artifact in this sibling reply."),
    ]);
    expect(index.byMessage.a2).toBeUndefined();
    expect(index.latest?.messageId).toBe("a1");
  });
});

describe("diffLines", () => {
  it("reports added, removed and unchanged lines", () => {
    const lines = diffLines("one\ntwo\nthree", "one\nTWO\nthree\nfour");
    expect(lines.filter((line) => line.kind === "same")).toHaveLength(2);
    expect(lines.filter((line) => line.kind === "remove")).toHaveLength(1);
    expect(lines.filter((line) => line.kind === "add")).toHaveLength(2);
    const removed = lines.find((line) => line.kind === "remove");
    expect(removed?.text).toBe("two");
    expect(removed?.oldLine).toBe(2);
  });

  it("is empty for identical content", () => {
    expect(diffLines(HTML, HTML).every((line) => line.kind === "same")).toBe(
      true,
    );
  });
});

describe("parseCsv", () => {
  it("handles quoted fields, embedded commas and CRLF", () => {
    const table = parseCsv(
      'id,name,note\r\n1,"Ada, A.","line one\r\nline two"\r\n2,Bo,\r\n',
    );
    expect(table.headers).toEqual(["id", "name", "note"]);
    expect(table.rows).toHaveLength(2);
    expect(table.rows[0][1]).toBe("Ada, A.");
    expect(table.rows[0][2]).toBe("line one\r\nline two");
    expect(table.rows[1][0]).toBe("2");
  });

  it("detects semicolon and tab files and fills missing cells", () => {
    const table = parseCsv("a;b;c\n1;2\n");
    expect(table.delimiter).toBe(";");
    expect(table.rows[0]).toEqual(["1", "2", ""]);
    expect(parseCsv("a\tb\n1\t2").delimiter).toBe("\t");
  });

  it("returns an empty table for empty input", () => {
    const table = parseCsv("");
    expect(table.headers).toEqual([]);
    expect(table.rows).toEqual([]);
  });
});

describe("JSON and SVG helpers", () => {
  it("reports invalid JSON without throwing", () => {
    expect(parseJsonValue('{"a":1}')).toEqual({ ok: true, value: { a: 1 } });
    const invalid = parseJsonValue('{"a":');
    expect(invalid.ok).toBe(false);
  });

  it("strips scripts, handlers and javascript URLs from SVG", () => {
    const dirty = sanitizeSvg(
      '<svg><script>alert(1)</script><rect onclick="x()" width="5"/><a href="javascript:alert(1)">x</a></svg>',
    );
    expect(dirty).not.toContain("<script");
    expect(dirty).not.toContain("onclick");
    expect(dirty).not.toContain("javascript:");
    expect(dirty).toContain("<rect");
  });

  it("strips external resources from SVG CSS and links", () => {
    const dirty = sanitizeSvg(
      '<svg><style>@import url("https://evil.test/a.css"); .x{fill:url(https://evil.test/p.svg)}</style><rect fill="url(https://evil.test/p.svg)"/><image href="https://evil.test/x.png"/><a href="https://evil.test">x</a><use href="#safe"/></svg>',
    );
    expect(dirty).not.toContain("evil.test");
    expect(dirty).not.toContain("@import");
    expect(dirty).toContain('href="#safe"');
  });
});

describe("artifact offsets", () => {
  it("keeps offsets stable when a fence is complete", () => {
    const content = `before\n\`\`\`html\n${HTML}\n\`\`\`\nafter`;
    const [artifact] = extractArtifacts(content, { messageId: "m" });
    expect(content.slice(artifact.start, artifact.end)).toContain("```html");
    expect(content.slice(artifact.start, artifact.end)).not.toContain("after");
    expect(splitArtifactSegments(content, [artifact])).toHaveLength(3);
  });

  it("treats an unclosed fence as a streaming artifact with the tail", () => {
    const content = `intro\n\`\`\`svg\n<svg><circle r="4"/>\n</svg>`;
    const [artifact] = extractArtifacts(content, {
      messageId: "m",
      isStreaming: true,
    });
    expect(artifact.partial).toBe(true);
    expect(artifact.end).toBe(content.length);
    expect(
      splitArtifactSegments(content, [artifact]).map((entry) => entry.kind),
    ).toEqual(["markdown", "artifact"]);
  });
});

describe("Artifact type contract", () => {
  it("keeps the external Artifact shape usable for a pane selection", () => {
    const artifact: Artifact | undefined = extractArtifacts(
      `\`\`\`json\n${JSON.stringify({ hello: "world" }, null, 2)}\n\`\`\``,
      { messageId: "m", minLines: 2 },
    )[0];
    expect(artifact?.type).toBe("json");
    expect(artifactFilename(artifact!)).toBe("json-data.json");
  });
});
