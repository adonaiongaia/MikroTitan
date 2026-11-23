# Changelog

All notable changes to Studio for Janas are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses semantic versioning. Release tags use the `janas-studio-vX.Y.Z` form; the early unversioned `janas-studio` tag predates that convention.

## [Unreleased]

### Added

- **Smart model companions and bundles.** A new `companions.rs` module reads GGUF v2/v3 headers with a bounds-checked `std::io::{Read, Seek}` parser and classifies each file as a text model, vision model, vision projector (`mmproj`) or draft-sized model from real metadata: `general.architecture`, `general.name`, `tokenizer.ggml.model`, the `tokenizer.ggml.tokens` count, `<architecture>.context_length`, `clip.*` markers and tensor shapes. Parameter counts and vocabulary sizes come from the tensor directory, and tensor payloads are never loaded. Results are cached by path, size and modification time.
- **Vision and speculative pairing.** Projectors are ranked by vision family, shared folder, name overlap and quantization (f16, bf16, q8_0); cross-family projectors are marked incompatible instead of guessed. Draft models are only offered when their architecture **and** exact vocabulary size match the main model, because a mismatch aborts llama-server, and every candidate lists its compatibility reasons. Pairings include the combined Main + Draft + MMPROJ footprint, a resident RAM estimate and a fully-offloaded VRAM estimate.
- **Saved model bundles.** `(main_model_path, mmproj_path, draft_model_path, draft_max_tokens, draft_min_tokens, draft_p_min)` tuples persist in `settings.json` with an active bundle pointer. Saving or applying a bundle validates the main model, projector family and draft vocabulary first, then sets the llama.cpp paths in one action. Pressing **Use** on a library model that has a saved bundle re-applies its projector and draft paths automatically. `delete_local_model` refuses to remove an mmproj or draft file bound by the active bundle.
- **Model Library companion UI.** Rows now carry **Vision ready**, **Missing MMPROJ · Pair**, **Draft paired (+Speculative)** and **Draft available** badges. A companion drawer lists every local projector and draft-sized model with compatibility notes, shows the combined memory footprint, validates the tuple with **Test pair**, and applies it with **Use bundle**. Library refreshes also inspect the whole GGUF folder for companion metadata.
- **Hugging Face bundle inspection.** `inspect_repo_bundles` splits `mmproj-*.gguf` files out of the main quantization list, preselects the reference projector (f16, then bf16, then q8_0), and returns a non-intrusive hint when the publisher has a 0.5B / 1B / 1.5B sibling. The download dialog gains an **Include Vision Projector (mmproj)** checkbox, a projector picker, the combined download size, and one download job for both files.
- **Draft tuning and pre-flight image guard.** llama.cpp now receives `--draft-max` (default 8), `--draft-min` and `--draft-p-min` only when the probed build advertises those flags. Attaching an image to a vision model without a bound projector shows an inline composer banner with **Pair detected MMPROJ** or **Download MMPROJ**, and sending is blocked until the projector is paired.
- **Remembered agent commands.** An approval card can allow one plain shell command for the current workspace. Rules persist in `agent-approvals.json` per canonical workspace, cover the same command with extra arguments, and can be listed, revoked individually or cleared in Engine Settings. Compound commands, likely secrets, irreversible commands and sensitive paths never match.
- **Per-chat system prompts and personas.** Each chat can keep its own system prompt, fall back to a saved default, or promote its prompt with **Make default**. A machine-local persona library saves, applies and deletes reusable prompts; branches inherit the source chat's prompt.
- **Hugging Face token support.** Settings accepts an optional access token for gated or private repositories. It is applied only to `huggingface.co` search, tree/checksum and download requests. A 401 now points to an invalid or expired token, while a 403 points to the repository licence or the token's access.
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

- **llama.cpp draft settings gained explicit controls.** Engine Settings exposes Draft max, Draft min and Draft p-min; activating a bundle clears the MTP head path and returns the speculation mode to Auto so a plain draft model is not launched with `--spec-type draft-mtp`.
- **The editable system prompt is now per chat.** A saved default seeds new chats and chats without an override; editing the prompt in Engine Settings no longer silently changes every conversation. Reset clears the active chat's prompt and the saved default.
- **Agent approval copy names remembered commands.** The policy descriptions now state that a remembered shell command can skip its prompt, and the approval checkbox explains the extra-argument match.
- **Settings and approval-rule files are owner-only on Unix, including their temporary copies.** `settings.json` (local API key and Hugging Face token) and `agent-approvals.json` are written with mode `0600`; the `.json.tmp` file is created with that mode from the outset, and a failed rename falls back to a direct write followed by the owner-only restriction.
- Renamed the app to **Studio for Janas** and added an explicit independent, unofficial disclaimer in the README and the About panel. The engine attribution now reads **Janas-LLM by Maurizio Cammalleri** with a link to the upstream project.
- Removed the pinned Janas binaries from the repository and the AppImage. Startup setup now downloads `janas-server`, `janas-chat`, `gguf2jns` and `janas-mcp` from the current `prabanta-dev/janas` release, and bare tool names such as `gguf2jns` and `janas-mcp` resolve from the managed `bin` folder.
- Manifest license metadata now reports `GPL-3.0-only`, matching the repository's GPL-3.0 `LICENSE`.
- AppImage builds now bundle the full GPL-3.0 text at `usr/share/licenses/studio-for-janas/LICENSE`; `bundle.licenseFile` also points at the repository `LICENSE`. A `THIRD_PARTY_NOTICES.md` with the Janas-LLM copyright, licence and source link ships at `usr/share/doc/studio-for-janas/`.
- Chat sessions moved from the old `localStorage` blob to the Rust-backed schema-v2 store.
- Closing the window now flushes pending chat writes against a short deadline before destroying the window, so a debounced save cannot be stranded.
- The Tools drawer's checkbox toggling now normalizes against the current server list, dropping stale IDs and preserving the all-selected state.

### Fixed

- Copied diagnostics now redact both stored credentials: the local API key and the Hugging Face token are replaced with `[redacted]`.
- Settings and approval-rule temp files are now created with owner-only permissions rather than tightened after the write, so they are never briefly readable by other local users. A stale temp path, including a symlink, is replaced rather than followed, and a partially written temp file is removed.
- Settings saves now report write and rename failures instead of silently reporting success, and the previous in-memory settings remain active when persistence fails.
- Hugging Face checksum-metadata failures now stop a download instead of silently skipping SHA-256 verification, and 401/403 errors distinguish an invalid token from a licence or scope denial.
- Approval cards now reset their checkboxes and feedback for each step, so “allow file edits” or “always allow” cannot carry over to a later command.
- Remembered-command writes are serialized with the rule file, so two approvals in different workspaces cannot overwrite each other.
- Commands the secret scrubber recognises are never stored as remembered rules, and rule matching rejects newline-separated or other compound commands.
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
