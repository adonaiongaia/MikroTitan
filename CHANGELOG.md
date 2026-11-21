# Changelog

All notable changes to Studio for Janas are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses semantic versioning. Release tags use the `janas-studio-vX.Y.Z` form; the early unversioned `janas-studio` tag predates that convention.

## [Unreleased]

### Added

- **Speculative decoding and MTP for llama.cpp.** The llama engine now mirrors the Janas speculation settings: an MTP/speculation mode (`--spec-type draft-mtp` or `none`), an optional MTP head GGUF, a draft model (`--spec-draft-model`), and draft placement (`--spec-draft-device none` or `--spec-draft-ngl all`). The adapter only emits the flags the probed build advertises and rejects a missing or non-GGUF draft file instead of launching without it.
- **Branching chat history.** Editing a user turn or regenerating a reply now keeps the old continuation as an archived branch. Previous/next arrows switch sibling versions, and **Branch** creates a new chat from a selected message's prefix.
- **Schema-v2 Rust chat store.** Each chat is stored as one JSON file under the app data directory. Saves are atomic, a previous version is kept per chat, images are extracted to `assets/<chat-id>/`, and orphaned images can be cleaned up.
- **Chat history backups.** Settings → Chat history and backups shows store sizes, exports and imports the whole shelf, restores a chat's previous version, and exposes storage maintenance.
- **Per-chat drafts and context.** Composer text and pending images persist per chat across restarts. Attached projects, workspace roots and MCP server scopes also follow the chat.
- **Model integrity tools.** Local models can be verified against the SHA-256 recorded at download time, with a clear result when no checksum is available. Model deletion now uses an awaited confirmation and refuses to remove the active model.
- **MCP scope controls.** The Tools drawer has a scope badge, All/None controls and a zero-tools warning. An empty selection stores `[]` and sends no MCP tools instead of silently restoring every server.
- **MCP scope reconciliation.** Scopes store selected server names with their IDs. On chat load, removed servers are pruned and servers re-created under the same name are remapped to the new ID.
- **Agent hardening.** Workspace pinning, diff previews, per-file undo/keep, planning, auto-compaction, verification commands, sensitive-path checks, secret redaction, bounded checkpoints and a prompt-mode tool protocol for models without native tool calling.
- **Automated coverage.** Vitest coverage for session normalization, branching, MCP request building, empty-scope sends, persistence boundaries and scope reconciliation, plus Rust tests for chat-store save/export/import/branch round-trips and agent safety behavior.

### Changed

- Renamed the app to **Studio for Janas** and added an explicit independent, unofficial disclaimer in the README and the About panel. The engine attribution now reads **Janas-LLM by Maurizio Cammalleri** with a link to the upstream project.
- Removed the pinned Janas binaries from the repository and the AppImage. Startup setup now downloads `janas-server`, `janas-chat`, `gguf2jns` and `janas-mcp` from the current `prabanta-dev/janas` release, and bare tool names such as `gguf2jns` and `janas-mcp` resolve from the managed `bin` folder.
- Manifest license metadata now reports `GPL-3.0-only`, matching the repository's GPL-3.0 `LICENSE`.
- AppImage builds now bundle the full GPL-3.0 text at `usr/share/licenses/studio-for-janas/LICENSE`; `bundle.licenseFile` also points at the repository `LICENSE`. A `THIRD_PARTY_NOTICES.md` with the Janas-LLM copyright, licence and source link ships at `usr/share/doc/studio-for-janas/`.
- Chat sessions moved from the old `localStorage` blob to the Rust-backed schema-v2 store.
- Closing the window now flushes pending chat writes against a short deadline before destroying the window, so a debounced save cannot be stranded.
- The Tools drawer's checkbox toggling now normalizes against the current server list, dropping stale IDs and preserving the all-selected state.

### Fixed

- Unchecking every MCP server in a chat no longer collapses the scope back to `undefined` and re-enables all servers.
- Model deletion no longer removes a file before the confirmation dialog is answered; **No** preserves the file and **Yes** deletes it.
- Window close now persists a pending chat change instead of leaving the window open or dropping the save.
- Persisted MCP selections survive re-created servers and no longer leave dead entries when a server is removed.

## [0.1.1] - 2026-11-16

### Added

- **Agent mode** with a bounded Think → Act → Observe loop and an inline trace.
- Workspace tools: `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `run_command`, `web_search` and MCP tools.
- Approval policies for destructive actions, all actions and full auto.
- Three selectable themes: Acid, Ember and Tide.

### Changed

- The Engine Settings panel gained agent configuration and live trace controls.

## [0.1.0] - 2026-11-11

### Added

- Initial public build.
- Dual-engine support for the Janas C engine and `llama.cpp`, including runtime provisioning, binary updates and rollback.
- Streaming chat workspace with reasoning blocks, context budgeting, project ingestion, web search, persistent memory and telemetry.
- Hugging Face GGUF explorer with quantization discovery, download progress, pause/resume and destination previews.
- GGUF → JNS compiler with a live console and conversion progress.
- MCP Server Hub for local stdio servers, probing and chat tool activity.
- Settings for binaries, model paths, engine flags, sampling and local API access.
