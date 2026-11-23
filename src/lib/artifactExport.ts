import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";

import {
  artifactExtension,
  artifactFilename,
  artifactMime,
  slugifyArtifactTitle,
  type Artifact,
} from "./artifacts";

/** Tauri 2 exposes its internals on the webview's window object. */
export function isTauriRuntime(): boolean {
  return (
    typeof window !== "undefined" && "__TAURI_INTERNALS__" in window
  );
}

function joinPath(directory: string, filename: string): string {
  const trimmed = directory.replace(/[\\/]+$/, "");
  const separator =
    trimmed.includes("\\") && !trimmed.includes("/") ? "\\" : "/";
  return `${trimmed}${separator}${filename}`;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/**
 * Write text through the native save dialog. The Rust command creates the file
 * only after the user has chosen the path, so the model never controls it.
 */
export async function saveTextFile(
  contents: string,
  filename: string,
  mime: string,
  directory?: string,
  extension?: string,
): Promise<string | null> {
  if (!isTauriRuntime()) {
    downloadBlob(new Blob([contents], { type: mime }), filename);
    return filename;
  }
  const path = await save({
    defaultPath: directory?.trim()
      ? joinPath(directory.trim(), filename)
      : filename,
    filters: extension
      ? [{ name: extension.toUpperCase(), extensions: [extension] }]
      : undefined,
  });
  if (!path) return null;
  return invoke<string>("artifact_save_file", {
    path,
    contents,
    base64: false,
  });
}

export async function saveArtifactFile(
  artifact: Artifact,
  directory?: string,
): Promise<string | null> {
  return saveTextFile(
    artifact.content,
    artifactFilename(artifact),
    artifactMime(artifact.type),
    directory,
    artifactExtension(artifact.type),
  );
}

/** Save straight into the chat's pinned workspace root, without a dialog. */
export async function saveArtifactToWorkspace(
  artifact: Artifact,
  workspaceRoot: string,
): Promise<string> {
  const root = workspaceRoot.trim();
  if (!root) throw new Error("Choose a workspace root before saving artifacts.");
  const path = joinPath(root, artifactFilename(artifact));
  return invoke<string>("artifact_save_file", {
    path,
    contents: artifact.content,
    base64: false,
  });
}

export async function saveDataUrlFile(
  dataUrl: string,
  filename: string,
  directory?: string,
): Promise<string | null> {
  if (!isTauriRuntime()) {
    const anchor = document.createElement("a");
    anchor.href = dataUrl;
    anchor.download = filename;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    return filename;
  }
  const path = await save({
    defaultPath: directory?.trim()
      ? joinPath(directory.trim(), filename)
      : filename,
    filters: [{ name: "PNG", extensions: ["png"] }],
  });
  if (!path) return null;
  return invoke<string>("artifact_save_file", {
    path,
    contents: dataUrl,
    base64: true,
  });
}

export function svgFilename(title: string): string {
  return `${slugifyArtifactTitle(title) || "artifact"}.svg`;
}

export function pngFilename(title: string): string {
  return `${slugifyArtifactTitle(title) || "artifact"}.png`;
}

