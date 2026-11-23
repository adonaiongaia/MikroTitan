import { useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  Eye,
  Link2,
  Loader2,
  Sparkles,
  X,
} from "lucide-react";

import { api } from "../../lib/api";
import { cn } from "../../lib/cn";
import { errorMessage, formatBytes, formatParameters } from "../../lib/format";
import type {
  BundleCheck,
  CompanionCandidate,
  CompanionPairing,
  GgufCompanionInfo,
  ModelBundle,
} from "../../types";
import { Button } from "../ui/Button";

interface CompanionDrawerProps {
  pairing: CompanionPairing;
  models: GgufCompanionInfo[];
  onClose: () => void;
  onUse: (mainPath: string) => Promise<void>;
  onChanged: () => void;
}

export function CompanionDrawer({
  pairing,
  models,
  onClose,
  onUse,
  onChanged,
}: CompanionDrawerProps) {
  const [mmproj, setMmproj] = useState(
    pairing.vision.bound_path ?? pairing.vision.recommended?.path ?? "",
  );
  const [draft, setDraft] = useState(
    pairing.draft.bound_path ?? pairing.draft.recommended?.path ?? "",
  );
  const [draftMax, setDraftMax] = useState(8);
  const [name, setName] = useState(
    pairing.main.file_name.replace(/\.gguf$/i, ""),
  );
  const [check, setCheck] = useState<BundleCheck | null>(null);
  const [testing, setTesting] = useState(false);
  const [using, setUsing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const visionCandidates = useMemo(() => {
    const map = new Map<string, CompanionCandidate>();
    for (const candidate of pairing.vision.candidates) {
      map.set(candidate.path, candidate);
    }
    return models
      .filter((model) => model.role === "mmproj")
      .map((model) => ({
        info: model,
        compatibility: map.get(model.path) ?? null,
      }));
  }, [models, pairing.vision.candidates]);

  const draftCandidates = useMemo(() => {
    const map = new Map<string, CompanionCandidate>();
    for (const candidate of pairing.draft.candidates) {
      map.set(candidate.path, candidate);
    }
    return models
      .filter((model) => model.role === "draft_candidate")
      .map((model) => ({
        info: model,
        compatibility: map.get(model.path) ?? null,
      }));
  }, [models, pairing.draft.candidates]);

  const mmprojInfo = models.find((model) => model.path === mmproj) ?? null;
  const draftInfo = models.find((model) => model.path === draft) ?? null;
  const combined =
    pairing.main.byte_size +
    (mmprojInfo?.byte_size ?? 0) +
    (draftInfo?.byte_size ?? 0);

  const bundle: ModelBundle = {
    id: pairing.active_bundle_id ?? "",
    name,
    main_model_path: pairing.main.path,
    mmproj_path: mmproj,
    draft_model_path: draft,
    draft_max_tokens: draftMax,
    draft_min_tokens: 0,
    draft_p_min: 0.75,
    created_at_ms: 0,
    updated_at_ms: 0,
  };

  const testPair = async () => {
    setTesting(true);
    setError(null);
    try {
      setCheck(await api.validateModelBundle(bundle));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setTesting(false);
    }
  };

  const useBundle = async () => {
    setUsing(true);
    setError(null);
    try {
      const result = await api.validateModelBundle(bundle);
      setCheck(result);
      if (!result.ok) {
        setError(result.errors.join(" "));
        return;
      }
      await api.saveModelBundle(bundle);
      onChanged();
      await onUse(pairing.main.path);
      onClose();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setUsing(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/80 p-6 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={`Companions for ${pairing.main.file_name}`}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !using) onClose();
      }}
    >
      <div className="panel flex max-h-[86vh] w-full max-w-3xl flex-col overflow-hidden bg-ink-900 shadow-panel animate-fade-in">
        <div className="flex items-start justify-between gap-4 border-b border-white/5 px-5 py-4">
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-[0.14em] text-slate-500">
              Companion bundling
            </p>
            <h2 className="mt-1 truncate text-base font-semibold text-white">
              {pairing.main.file_name}
            </h2>
            <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
              <RoleBadge role={pairing.main.role} />
              {pairing.main.vision_family ? (
                <span className="rounded border border-sky-400/20 bg-sky-400/5 px-1.5 py-0.5 text-[8px] text-sky-300">
                  {pairing.main.vision_family}
                </span>
              ) : null}
              {pairing.main.architecture ? (
                <span className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-[8px] text-slate-400">
                  {pairing.main.architecture}
                </span>
              ) : null}
              {pairing.main.vocab_size ? (
                <span className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-[8px] text-slate-400">
                  {pairing.main.vocab_size.toLocaleString()} vocab
                </span>
              ) : null}
              {pairing.main.parameter_estimate ? (
                <span className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-[8px] text-slate-400">
                  {formatParameters(pairing.main.parameter_estimate)}
                </span>
              ) : null}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
            <p className="label">Bundle name</p>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              className="field mt-1.5 !py-1.5 text-[11px]"
              placeholder="My smart bundle"
            />
          </div>

          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-[11px] font-medium text-slate-200">
                <Eye className="h-3.5 w-3.5 text-sky-300" /> Vision projector
              </span>
              <span className="text-[9.5px] text-slate-600">
                {visionCandidates.length} local mmproj
              </span>
            </div>
            {pairing.main.role === "vision_main" ? (
              <p
                className={cn(
                  "rounded-lg border px-2.5 py-2 text-[9.5px] leading-4",
                  pairing.vision.bound_path
                    ? "border-emerald-400/20 bg-emerald-400/5 text-emerald-200/80"
                    : "border-amber-400/20 bg-amber-400/5 text-amber-200/80",
                )}
              >
                {pairing.vision.message}
              </p>
            ) : (
              <p className="text-[9.5px] leading-4 text-slate-600">
                Text-only model: a projector is not required. Overriding is
                allowed for testing but will fail at launch on a text model.
              </p>
            )}
            <select
              value={mmproj}
              onChange={(event) => setMmproj(event.target.value)}
              className="field !py-1.5 font-mono text-[10.5px]"
            >
              <option value="">No projector</option>
              {visionCandidates.map(({ info, compatibility }) => (
                <option key={info.path} value={info.path}>
                  {info.file_name} · {info.quant} ·{" "}
                  {info.vision_family ?? "unknown family"}
                  {compatibility
                    ? compatibility.compatible
                      ? " · compatible"
                      : ` · ${compatibility.reasons[0] ?? "check family"}`
                    : " · not scored"}
                </option>
              ))}
            </select>
            {mmprojInfo && mmprojInfo.vision_markers.length ? (
              <p className="truncate text-[9px] text-slate-600">
                Markers:{" "}
                <span className="font-mono text-slate-500">
                  {mmprojInfo.vision_markers.slice(0, 4).join(", ")}
                </span>
              </p>
            ) : null}
          </section>

          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-[11px] font-medium text-slate-200">
                <Sparkles className="h-3.5 w-3.5 text-violet-300" /> Speculative
                draft model
              </span>
              <span className="text-[9.5px] text-slate-600">
                {draftCandidates.length} draft-sized model
                {draftCandidates.length === 1 ? "" : "s"}
              </span>
            </div>
            <p
              className={cn(
                "rounded-lg border px-2.5 py-2 text-[9.5px] leading-4",
                draftCandidates.some(
                  (candidate) => candidate.compatibility?.compatible,
                )
                  ? "border-violet-400/20 bg-violet-400/5 text-violet-200/80"
                  : "border-white/5 bg-white/[0.02] text-slate-500",
              )}
            >
              {pairing.draft.message}
            </p>
            <select
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              className="field !py-1.5 font-mono text-[10.5px]"
            >
              <option value="">No draft model</option>
              {draftCandidates.map(({ info, compatibility }) => (
                <option key={info.path} value={info.path}>
                  {info.file_name} · {info.quant}
                  {info.parameter_estimate
                    ? ` · ${formatParameters(info.parameter_estimate)}`
                    : ""}
                  {compatibility
                    ? compatibility.compatible
                      ? " · exact vocab + architecture"
                      : ` · ${compatibility.reasons[0] ?? "incompatible"}`
                    : " · unchecked"}
                </option>
              ))}
            </select>
            <label className="flex items-center justify-between gap-3 rounded-lg border border-white/5 bg-ink-950/50 px-2.5 py-2">
              <span className="text-[10px] text-slate-500">
                Draft max tokens{" "}
                <span className="font-mono text-slate-600">--draft-max</span>
              </span>
              <input
                type="number"
                min={1}
                max={64}
                step={1}
                value={draftMax}
                onChange={(event) =>
                  setDraftMax(Number(event.target.value) || 8)
                }
                className="field !w-20 !py-1 font-mono text-[10.5px]"
              />
            </label>
            <p className="text-[9px] leading-4 text-slate-600">
              A draft is only offered when its architecture and vocabulary size
              match the main model exactly. A mismatch aborts llama-server.
            </p>
          </section>

          {check ? (
            <div
              className={cn(
                "space-y-1.5 rounded-xl border px-3 py-2.5 text-[10px] leading-4",
                check.ok
                  ? "border-emerald-400/20 bg-emerald-400/5 text-emerald-200/80"
                  : "border-rose-400/25 bg-rose-500/5 text-rose-200/90",
              )}
            >
              <p className="flex items-center gap-1.5 font-medium">
                {check.ok ? (
                  <Check className="h-3 w-3" />
                ) : (
                  <AlertTriangle className="h-3 w-3" />
                )}
                {check.ok ? "Bundle checks out" : "Bundle cannot launch"}
              </p>
              {check.errors.map((message) => (
                <p key={message}>• {message}</p>
              ))}
              {check.warnings.map((message) => (
                <p key={message} className="text-amber-200/80">
                  • {message}
                </p>
              ))}
            </div>
          ) : null}

          {error ? (
            <div className="flex items-start gap-2 rounded-xl border border-rose-400/20 bg-rose-500/5 px-3 py-2 text-[10px] leading-4 text-rose-300">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          ) : null}

          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
            <p className="label">Combined memory footprint</p>
            <div className="mt-2 grid grid-cols-3 gap-2 text-[10px]">
              <Footprint
                label="Main"
                value={formatBytes(pairing.main.byte_size)}
              />
              <Footprint
                label="MMPROJ"
                value={mmprojInfo ? formatBytes(mmprojInfo.byte_size) : "—"}
                accent={Boolean(mmprojInfo)}
              />
              <Footprint
                label="Draft"
                value={draftInfo ? formatBytes(draftInfo.byte_size) : "—"}
                accent={Boolean(draftInfo)}
              />
            </div>
            <div className="mt-2 flex items-center justify-between border-t border-white/5 pt-2">
              <span className="text-[10px] text-slate-500">
                Main + Draft + MMPROJ
              </span>
              <span className="font-mono text-[11px] text-slate-200">
                {formatBytes(combined)}
              </span>
            </div>
            <p className="mt-1 text-[9px] leading-4 text-slate-600">
              About {formatBytes(pairing.estimated_ram_bytes)} system RAM with
              the KV cache and runtime headroom ·{" "}
              {formatBytes(pairing.estimated_vram_bytes)} VRAM if fully
              offloaded.
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-white/5 bg-ink-850/60 px-5 py-3.5">
          <p className="text-[9.5px] text-slate-600">
            Activation sets the llama.cpp model, projector and draft model, then
            restarts the engine.
          </p>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="secondary"
              busy={testing}
              onClick={() => void testPair()}
              icon={
                testing ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Link2 className="h-3 w-3" />
                )
              }
            >
              Test pair
            </Button>
            <Button
              size="sm"
              variant="primary"
              busy={using}
              onClick={() => void useBundle()}
              icon={<Check className="h-3.5 w-3.5" />}
            >
              Use bundle
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Footprint({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-white/5 bg-ink-950/60 px-2.5 py-1.5">
      <p className="text-[8px] uppercase tracking-wider text-slate-600">
        {label}
      </p>
      <p
        className={cn(
          "mt-0.5 font-mono text-[10.5px]",
          accent ? "text-acid-300" : "text-slate-400",
        )}
      >
        {value}
      </p>
    </div>
  );
}

function RoleBadge({ role }: { role: GgufCompanionInfo["role"] }) {
  const label =
    role === "vision_main"
      ? "Vision model"
      : role === "mmproj"
        ? "Projector"
        : role === "draft_candidate"
          ? "Draft-sized"
          : "Text model";
  return (
    <span
      className={cn(
        "rounded border px-1.5 py-0.5 text-[8px] font-medium uppercase tracking-wider",
        role === "vision_main"
          ? "border-sky-400/25 bg-sky-400/10 text-sky-200"
          : role === "mmproj"
            ? "border-sky-400/20 bg-sky-400/5 text-sky-300/80"
            : role === "draft_candidate"
              ? "border-violet-400/20 bg-violet-400/5 text-violet-300/80"
              : "border-white/10 bg-white/5 text-slate-400",
      )}
    >
      {label}
    </span>
  );
}
