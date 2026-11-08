import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { AlertTriangle, ExternalLink, Loader } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";

import { api } from "../lib/api";
import { bytes, cx, ENGINE_LABEL, memoryPool, params, tokens } from "../lib/format";
import { useDebounced } from "../lib/hooks";
import { useLive } from "../lib/live";
import type { CatalogItem, DeploymentSpec, Variant } from "../lib/types";
import { FitGauge } from "./meters";
import { ModelLogo } from "./ModelLogo";
import { CapabilityTags } from "./CapabilityTags";
import { effortLabel } from "../lib/capabilities";
import { Button, Field, Input, Select, Textarea, Toggle, VerdictTag } from "./ui";

export interface ModelChoice {
  item: CatalogItem;
  spec: DeploymentSpec;
}

/** Settings that change a model's size or fit, re-checked by the server as they change. */
const FIT_OPTIONS = ["ctx", "offload", "quantize", "weights"];

const CTX = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

/** Selector options: standard steps up to this model's window, then the window itself. */
function ctxOptions(max?: number | null, current?: number): number[] {
  const values = max ? CTX.filter((c) => c < max) : [...CTX];
  if (max) values.push(max);
  if (current && (!max || current <= max) && !values.includes(current)) values.push(current);
  return [...new Set(values)].sort((a, b) => a - b);
}

/** Engine options shown per modality. Values are merged into spec.options. */
function EngineOptions({
  item,
  options,
  onChange,
}: {
  item: Pick<CatalogItem, "modality" | "engine" | "source" | "custom_code" | "context_length" | "bf16_saved">;
  options: Record<string, any>;
  onChange: (o: Record<string, any>) => void;
}) {
  const set = (k: string, v: unknown) => onChange({ ...options, [k]: v === "" ? undefined : v });
  const m = item.modality;
  if (m === "text") {
    const current = typeof options.ctx === "number" ? options.ctx : undefined;
    const fallback = Math.min(8192, item.context_length ?? 8192);
    const ladder = ctxOptions(item.context_length, current);
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <Field
          label="Context length"
          hint={
            item.context_length
              ? `This model supports up to ${tokens(item.context_length)} tokens. Longer context uses more VRAM for the KV cache.`
              : "Context limit is not published for this model. Pick the window you need; longer context uses more VRAM for the KV cache."
          }
        >
          <Select value={String(ladder.includes(current ?? fallback) ? current ?? fallback : fallback)} onChange={(e) => set("ctx", Number(e.target.value))}>
            {ladder.map((c) => (
              <option key={c} value={c}>
                {c.toLocaleString()} tokens
              </option>
            ))}
          </Select>
        </Field>
        {item.engine === "vllm" && (
          <Field label="GPU memory use" hint="Fraction of VRAM vLLM reserves.">
            <Select value={String(options.gpu_memory_utilization ?? 0.9)} onChange={(e) => set("gpu_memory_utilization", Number(e.target.value))}>
              {[0.8, 0.85, 0.9, 0.95].map((v) => (
                <option key={v} value={v}>
                  {Math.round(v * 100)}%
                </option>
              ))}
            </Select>
          </Field>
        )}
        {item.engine === "sglang" && (
          <Field label="GPU memory use" hint="Fraction of VRAM SGLang reserves for weights and the KV cache.">
            <Select value={String(options.mem_fraction_static ?? 0.88)} onChange={(e) => set("mem_fraction_static", Number(e.target.value))}>
              {[0.7, 0.8, 0.85, 0.88, 0.9, 0.95].map((v) => (
                <option key={v} value={v}>
                  {Math.round(v * 100)}%
                </option>
              ))}
            </Select>
          </Field>
        )}
      </div>
    );
  }
  if (m === "image" || m === "video" || (m === "world" && item.engine === "diffusers_world")) {
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Memory mode" hint="Offload keeps idle parts in RAM: slower, fits bigger models.">
          <Select value={String(options.offload ?? (m === "video" ? "model" : "none"))} onChange={(e) => set("offload", e.target.value)}>
            <option value="none">All on GPU (fastest)</option>
            <option value="model">Model offload (balanced)</option>
            <option value="sequential">Sequential offload (smallest, slowest)</option>
          </Select>
        </Field>
        <Field label="Weights" hint="4-bit shrinks big models like FLUX and Wan 14B to fit a T4.">
          <Select value={String(options.quantize ?? "")} onChange={(e) => set("quantize", e.target.value)}>
            <option value="">16-bit (full quality)</option>
            <option value="nf4">4-bit nf4 (bitsandbytes)</option>
          </Select>
        </Field>
        {item.bf16_saved ? (
          <Field label="Download as" hint="The repo is published in fp32; the runner uses 16 bits either way.">
            <Select value={String(options.weights ?? "bf16")} onChange={(e) => set("weights", e.target.value === "bf16" ? "" : e.target.value)}>
              <option value="bf16">BF16 (saves {bytes(item.bf16_saved)})</option>
              <option value="original">Original fp32 files</option>
            </Select>
          </Field>
        ) : null}
        <Field label="Default steps">
          <Input type="number" min={1} max={100} value={options.steps ?? ""} placeholder="model default" onChange={(e) => set("steps", e.target.value ? Number(e.target.value) : "")} />
        </Field>
        {m === "image" ? (
          <Field label="Default guidance">
            <Input type="number" step={0.5} min={0} max={20} value={options.guidance ?? ""} placeholder="model default" onChange={(e) => set("guidance", e.target.value ? Number(e.target.value) : "")} />
          </Field>
        ) : (
          <Field label="Default frames">
            <Input type="number" min={5} max={241} value={options.frames ?? ""} placeholder="model default" onChange={(e) => set("frames", e.target.value ? Number(e.target.value) : "")} />
          </Field>
        )}
      </div>
    );
  }
  if (m === "world") {
    if (options.recipe === "hunyuan_world_1" || options.recipe === "hy_world_2") {
      return <p className="text-[13px] text-fg-2">Scene description, panorama input and reconstruction settings are available in Studio after launch.</p>;
    }
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Default frames">
          <Input type="number" min={4} max={600} value={options.frames ?? ""} placeholder="recipe default" onChange={(e) => set("frames", e.target.value ? Number(e.target.value) : "")} />
        </Field>
        <Field label="Default steps">
          <Input type="number" min={1} max={50} value={options.steps ?? ""} placeholder="recipe default" onChange={(e) => set("steps", e.target.value ? Number(e.target.value) : "")} />
        </Field>
      </div>
    );
  }
  return <p className="text-[13px] text-fg-3">No extra settings for this engine.</p>;
}

export function OptionsForm(props: Parameters<typeof EngineOptions>[0]) {
  const { item, options, onChange } = props;
  const customCode = ["vllm", "sglang", "transformers", "transformers_tts", "transformers_audio", "transformers_asr"].includes(item.engine ?? "");
  return <div className="space-y-4">
    <EngineOptions {...props} />
    {customCode && <div className="space-y-3 border-t border-line pt-4">
      <Toggle checked={options.trust_remote_code === true}
        onChange={v => onChange({ ...options, trust_remote_code: v })}
        label="Allow model custom code"
        hint={item.custom_code ? "This checkpoint declares custom code. Enable only if you trust its source; its Python code runs in your notebook." : "For checkpoints that need their own Python code. Enable only for a source you trust."} />
      {item.source === "hf" && <Field label="Model revision" hint="Optional branch, tag or commit SHA. Use a commit SHA to pin the model and its code.">
        <Input value={options.revision ?? ""} placeholder="main" onChange={e => onChange({ ...options, revision: e.target.value || undefined })} />
      </Field>}
    </div>}
  </div>;
}

function VariantRow({ v, active, onPick }: { v: Variant; active: boolean; onPick: () => void }) {
  return (
    <button
      type="button"
      onClick={onPick}
      className={cx(
        "flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
        active ? "border-signal bg-signal-soft" : "border-line hover:border-line-2",
      )}
    >
      <span className={cx("size-3 shrink-0 rounded-full border-2", active ? "border-signal bg-signal" : "border-line-2")} />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-mono text-[12.5px]">{v.label}</span>
        {v.fit?.reasons?.[0] && <span className="block truncate text-[11.5px] text-fg-3">{v.fit.reasons[0]}</span>}
      </span>
      <span className="font-mono text-[11.5px] text-fg-2">{bytes(v.size)}</span>
      {v.fit && <VerdictTag verdict={v.fit.verdict} size="sm" />}
    </button>
  );
}

export function ModelDetail({
  item: base,
  actionLabel,
  onAction,
  busy,
  secondary,
  onImport,
}: {
  item: CatalogItem;
  actionLabel: string;
  onAction: (choice: ModelChoice) => void;
  busy?: boolean;
  secondary?: (choice: ModelChoice) => React.ReactNode;
  onImport?: (url: string) => void;
}) {
  const { overview } = useLive();
  const nav = useNavigate();
  const pool = overview ? memoryPool(overview.system) : null;
  const devices = pool?.devices ?? [];
  const canDetail = base.source === "hf" || base.source === "ollama";
  const detail = useQuery({
    queryKey: ["detail", base.key],
    queryFn: () => api.detail(base.key),
    enabled: canDetail,
    staleTime: 10 * 60_000,
  });
  const loaded = detail.data;
  // "Best" is relative to the list it came from; keep it when the detail view agrees the model fits well.
  const item =
    loaded && base.fit.verdict === "best" && loaded.fit.verdict === "good" ? { ...loaded, fit: { ...loaded.fit, verdict: "best" as const } } : (loaded ?? base);
  const [variant, setVariant] = useState<string | null>(base.variant);
  const [engine, setEngine] = useState<string | null>(base.engine);
  const [options, setOptions] = useState<Record<string, any>>({ ...(base.spec?.options ?? {}) });
  const [swap, setSwap] = useState(base.spec?.lifecycle === "llama_swap");
  const [name, setName] = useState("");

  // Server-side serving options live in the same free-form dict as engine options.
  const setOption = (key: string, value: unknown) =>
    setOptions((o) => {
      const next = { ...o };
      if (value === undefined || value === "" || value === "inherit") delete next[key];
      else next[key] = value;
      return next;
    });
  const compress = Array.isArray(options.compress) ? (options.compress as string[]) : [];
  const toggleCompress = (kind: string) =>
    setOption("compress", compress.includes(kind)
      ? compress.filter((k) => k !== kind)
      : [...compress, kind]);

  useEffect(() => {
    if (detail.data) {
      setVariant((v) => v ?? detail.data.variant);
      setOptions((o) => ({ ...(detail.data.spec?.options ?? {}), ...o }));
    }
  }, [detail.data]);

  const engines = useMemo(() => {
    // BigMoeOnEdge only pays off when the experts cannot sit in VRAM, and only for a MoE GGUF.
    const streamable = item.moe && (item.fit.mode === "cpu" || item.fit.mode === "gpu_cpu");
    if (item.modality === "text") {
      // Every GGUF model runs on either runner; both are real choices with different strengths.
      if (item.engine === "ollama") return ["ollama", "llamacpp", ...(streamable ? ["bigmoe"] : [])];
      if (item.source === "hf") return pool?.tpu ? ["vllm", "transformers"] : ["vllm", "sglang", "transformers"];
    }
    return item.engine ? [item.engine] : [];
  }, [item, pool?.tpu]);

  const picked = item.variants.find((v) => v.id === variant);
  // Re-check size and fit whenever a setting that changes them does (variant, engine, context, memory mode).
  const current = JSON.stringify({
    variant: variant ?? item.variant,
    engine: engine ?? item.engine,
    options: Object.fromEntries(FIT_OPTIONS.map((k) => [k, options[k]]).filter(([, v]) => v !== undefined && v !== "")),
  });
  const request = useDebounced(current, 200);
  const assessed = useQuery({
    queryKey: ["assess", base.key, request],
    queryFn: () => api.assess({ key: base.key, ...JSON.parse(request) }),
    enabled: canDetail && !!item.engine && detail.isSuccess,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
    retry: false,
  });
  const assessedFit = assessed.data?.fit;
  const fit = assessedFit ? { ...assessedFit, verdict: assessedFit.verdict === "good" && item.fit.verdict === "best" ? ("best" as const) : assessedFit.verdict } : item.fit;
  const size = assessed.data?.size ?? picked?.size ?? item.size;
  const updating = assessed.isFetching || request !== current;
  const choice: ModelChoice | null = item.spec
    ? {
        item: { ...item, size, fit, variant: variant ?? item.variant },
        spec: {
          ...item.spec,
          model: { ...item.spec.model, variant: variant ?? item.spec.model.variant, engine: engine ?? item.spec.model.engine },
          name: name.trim() || null,
          lifecycle: (engine ?? item.engine) === "llamacpp" && swap ? "llama_swap" : "direct",
          options: Object.fromEntries(
            Object.entries<unknown>({
              ...options,
              // Placement hint for the GPU scheduler; a different variant than the default changes it.
              vram_estimate: fit.vram_needed ?? undefined,
            }).filter(([, v]) => v !== undefined && v !== ""),
          ),
        },
      }
    : null;

  return (
    <div className="flex min-h-full flex-col">
      <div className="space-y-6 p-5 sm:p-6">
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            {item.engine ? <VerdictTag verdict={fit.verdict} /> : <span className="text-[12px] text-fg-2">Runner unavailable</span>}
            {item.experimental && <span className="rounded-[5px] bg-tight-soft px-1.5 py-px text-[11px] font-medium text-tight">Experimental</span>}
            {item.finetune && (
              <span className="rounded-[5px] bg-bg-2 px-1.5 py-px text-[11px] font-medium text-fg-2" title="Model Studio can fine-tune or abliterate this base repo.">
                Fine-tunable
              </span>
            )}
            {detail.isFetching && <Loader className="size-3.5 animate-spin text-fg-3" />}
          </div>
          <div className="flex items-start gap-4"><ModelLogo id={item.id} name={item.name} source={item.source} className="size-12" /><h2 className="display min-w-0 break-words text-[32px] leading-tight">{item.name}</h2></div>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11.5px] text-fg-2">
            <span>{item.id}</span>
            {item.brand && <span>{item.brand.label}</span>}
            {item.params ? <span>{params(item.params)} params</span> : null}
            {item.context_length ? <span>{tokens(item.context_length)} context</span> : null}
            {item.max_output_tokens && item.max_output_tokens !== item.context_length ? (
              <span>{tokens(item.max_output_tokens)} max output</span>
            ) : null}
            {item.engine_label && <span>{item.engine_label}</span>}
            {item.url && (
              <a href={item.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-fg-2 hover:text-signal">
                page <ExternalLink className="size-3" />
              </a>
            )}
          </div>
          {item.models_dev && (
            <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-fg-3">
              {item.models_dev.family && <span>{item.models_dev.family} family</span>}
              {item.models_dev.knowledge && <span>Knowledge {item.models_dev.knowledge}</span>}
              {item.models_dev.release_date && <span>Released {item.models_dev.release_date}</span>}
              {item.models_dev.attachment && <span>Accepts file inputs</span>}
              {item.models_dev.reasoning && !item.reasoning?.supported && <span>Reasoning</span>}
              {item.models_dev.tool_call && !item.capabilities.includes("tools") && <span>Tool calls</span>}
              <span>Facts from models.dev</span>
            </div>
          )}
          {item.description && <p className="mt-3 text-[14px] text-fg-2">{item.description}</p>}
          {item.capabilities.length > 0 && <div className="mt-4"><CapabilityTags capabilities={item.capabilities} /></div>}
          {item.reasoning?.supported && <p className="mt-2 text-[12px] text-fg-3">Reasoning: {item.reasoning.efforts.length ? ["Auto", ...item.reasoning.efforts.map(effortLabel)].join(" · ") : "Model default"}. {item.reasoning.source === "family" ? "Estimated from the model family; verified by the runner when loaded." : "Detected from model metadata or its chat template."}</p>}
          {!!item.runtime_repos?.length && <div className="mt-3 text-[12px] text-fg-3"><p>The download estimate includes these base checkpoints:</p><div className="mt-1 flex flex-wrap gap-x-3 gap-y-1">{item.runtime_repos.map(repo => <a key={repo} href={`https://huggingface.co/${repo}`} target="_blank" rel="noreferrer" className="text-fg-2 hover:text-signal hover:underline">{repo}</a>)}</div></div>}
        </div>

        {!item.engine ? (
          <div className="rounded-xl border border-line bg-bg-2 p-4">
            <div className="mb-2 text-[14px] font-medium">Runner not available yet</div>
            <p className="text-[13px] text-fg-2">This model is published on Hugging Face. KaggleUI doesn't have a runtime for it yet, so it can't be deployed from this panel.</p>
            {item.url && <a href={item.url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-[13px] text-signal hover:underline">Open model documentation <ExternalLink className="size-3.5" /></a>}
            {item.url && onImport && <div className="mt-3"><Button onClick={() => onImport(item.url!)}>Download model files</Button></div>}
          </div>
        ) : <div className="rounded-xl border border-line bg-bg-2 p-4">
          <div className="mb-3 flex items-baseline justify-between gap-3">
            <span className="readout">Fit with these settings</span>
            <span className={cx("font-mono text-[11.5px] text-fg-2 transition-opacity", updating && "opacity-50")}>
              {size ? `${bytes(size)} download` : "size unknown"}
              {updating && <Loader className="ml-1.5 inline size-3 animate-spin align-[-2px]" />}
            </span>
          </div>
          <div className={cx("transition-opacity", updating && "opacity-60")}>
          <FitGauge fit={fit} devices={devices} />
          <ul className="mt-3 space-y-1 text-[13px]">
            {fit.reasons.map((r) => (
              <li key={r} className="text-fg-2">
                {r}
              </li>
            ))}
            {fit.warnings.map((w) => (
              <li key={w} className="flex items-start gap-1.5 text-tight">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>
                  {w}
                  {w.startsWith("Gated") && (
                    <Link to="/access#secrets" className="ml-2 whitespace-nowrap font-medium underline underline-offset-4">
                      Add token →
                    </Link>
                  )}
                </span>
              </li>
            ))}
          </ul>
          </div>
          {assessed.error && <p className="mt-2 text-[12px] text-tight">Couldn't re-check this configuration: {(assessed.error as Error).message}</p>}
          {item.bf16_saved && options.weights !== "original" ? (
            <p className="mt-3 text-[12.5px] text-fg-2">
              Published in fp32. KaggleUI stores the weights as BF16 while downloading, which saves {bytes(item.bf16_saved)} of disk. Generation quality is unchanged: the runner loads 16-bit weights either way.
            </p>
          ) : null}
          {Object.keys(fit.suggest ?? {}).length > 0 && (
            <div className="mt-3 font-mono text-[11px] text-fg-3">auto-applied: {Object.entries(fit.suggest).map(([k, v]) => `${k}=${v}`).join(", ")}</div>
          )}
        </div>}

        {!!item.presets?.length && (
          <div>
            <div className="readout mb-2">Presets</div>
            <div className="space-y-1.5">
              {item.presets.map((preset) => (
                <button
                  key={preset.name}
                  type="button"
                  onClick={() => {
                    if (preset.engine) setEngine(preset.engine);
                    if (preset.lifecycle) setSwap(preset.lifecycle === "llama_swap");
                    if (preset.quant) {
                      const match = item.variants.find((v) => `${v.id} ${v.label}`.toLowerCase().includes(preset.quant!.toLowerCase()));
                      if (match) setVariant(match.id);
                      else setOption("quantize", preset.quant);
                    }
                    if (preset.ctx) setOption("ctx", preset.ctx);
                    for (const [key, value] of Object.entries(preset.options ?? {})) setOption(key, value);
                  }}
                  className="w-full rounded-lg border border-line px-3 py-2 text-left text-[13px] hover:border-fg-3"
                >
                  <span className="font-medium">{preset.name}</span>
                  <span className="ml-2 font-mono text-[11.5px] text-fg-3">
                    {[preset.engine, preset.quant, preset.ctx && `${preset.ctx} ctx`, preset.lifecycle].filter(Boolean).join(" · ")}
                  </span>
                  {preset.note && <span className="mt-0.5 block text-[11.5px] text-fg-3">{preset.note}</span>}
                </button>
              ))}
            </div>
          </div>
        )}

        {item.variants.length > 1 && (
          <div>
            <div className="readout mb-2">{item.source === "ollama" ? "Size / tag" : "Quantization"}</div>
            <div className="max-h-[300px] space-y-1.5 overflow-y-auto pr-1 scroll-thin">
              {item.variants.map((v) => (
                <VariantRow key={v.id} v={v} active={v.id === (variant ?? item.variant)} onPick={() => setVariant(v.id)} />
              ))}
            </div>
            {picked?.context && <div className="mt-2 font-mono text-[11px] text-fg-3">context window {picked.context}</div>}
          </div>
        )}

        {engines.length > 1 && (
          <Field label="Engine" hint={item.engine === "ollama"
            ? "Ollama shares one server across models; llama.cpp runs a dedicated llama-server for this model and exposes explicit GPU placement."
            : "vLLM and SGLang are the fast runners (experimental on T4s); Transformers is the safe fallback."}>
            <Select value={engine ?? ""} onChange={(e) => setEngine(e.target.value)}>
              {engines.map((e) => (
                <option key={e} value={e}>
                  {ENGINE_LABEL[e] ?? e}
                </option>
              ))}
            </Select>
          </Field>
        )}

        {(engine ?? item.engine) === "llamacpp" && (
          <Toggle
            checked={swap}
            onChange={setSwap}
            label="Run under llama-swap"
            hint="One shared process loads GGUF models on demand, so several models can share VRAM and the API can switch between them without a deployment each. Models that are available but idle unload after 5 minutes."
          />
        )}

        {item.engine && (
          <div>
            <div className="readout mb-2">Settings</div>
            <OptionsForm
              item={{ ...item, engine: engine ?? item.engine, context_length: picked?.context_length ?? item.context_length }}
              options={options}
              onChange={setOptions}
            />
          </div>
        )}

        {item.modality === "text" && (
          <details className="rounded-xl border border-line bg-bg-2/40 px-4 py-3">
            <summary className="cursor-pointer text-[13px] font-medium text-fg-2">Model options</summary>
            <div className="mt-3 space-y-3" onClick={(e) => e.stopPropagation()}>
              <Field label="System prompt" hint="Sent ahead of every request to this model. Empty falls back to the global default.">
                <Textarea
                  rows={3}
                  value={options.system_prompt ?? ""}
                  placeholder="Leave empty to use the global default"
                  onChange={(e) => setOption("system_prompt", e.target.value)}
                />
              </Field>
              <Field label="System prompt mode">
                <Select
                  value={options.system_prompt_mode === "replace" ? "replace" : "prepend"}
                  onChange={(e) => setOption("system_prompt_mode", e.target.value === "replace" ? "replace" : undefined)}
                >
                  <option value="prepend">Prepend to the app's prompt</option>
                  <option value="replace">Replace the app's prompt</option>
                </Select>
              </Field>
              <Toggle
                checked={Boolean(options.slim_harness)}
                onChange={(v) => setOption("slim_harness", v ? true : undefined)}
                label="Slim harness"
                hint="Drop the static instruction sections of recognised coding harnesses from system messages."
              />
              <Toggle
                checked={compress.includes("tools")}
                onChange={() => toggleCompress("tools")}
                label="Compress tool output"
                hint="Strip ANSI, collapse repeated lines, drop install progress and truncate long results at 200 lines."
              />
              <Toggle
                checked={compress.includes("prose")}
                onChange={() => toggleCompress("prose")}
                label="Compress prose"
                hint="Remove stock acknowledgements and closing offers from earlier assistant turns."
              />
              <Field label="Response cache">
                <Select
                  value={typeof options.cache === "string" ? options.cache : "inherit"}
                  onChange={(e) => setOption("cache", e.target.value)}
                >
                  <option value="inherit">Follow global settings</option>
                  <option value="off">Off for this model</option>
                  <option value="exact">Exact match</option>
                  <option value="semantic">Semantic match</option>
                </Select>
              </Field>
            </div>
          </details>
        )}

        {item.engine && (
          <Field label="API model name (optional)" hint="What clients send as 'model'. Defaults to a short name.">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={item.source === "ollama" ? `${item.id}:${variant ?? "latest"}` : item.id.split("/").pop()?.toLowerCase()} />
          </Field>
        )}
      </div>

      <div className="sticky bottom-0 mt-auto flex flex-wrap items-center justify-end gap-2 border-t border-line bg-panel/95 px-5 py-4 backdrop-blur sm:px-6">
        {!item.engine ? (
          <span className="mr-auto text-[13px] text-fg-3">This architecture needs a model-specific runner.</span>
        ) : (
          <>
            {item.modality === "text" && (item.source === "hf" || item.source === "ollama") && (
              <button
                type="button"
                className="mr-auto text-[13px] text-fg-2 underline-offset-4 hover:text-fg hover:underline"
                onClick={async () => {
                  // Ollama/GGUF can't be edited: start from the original full-precision repo.
                  let repo = item.abliterate_source ?? "";
                  let guessed = Boolean(item.abliterate_guessed);
                  if (item.source === "ollama") {
                    repo = (await api.original(item.id, variant ?? item.variant ?? "").catch(() => ({ repo: null }))).repo ?? "";
                    guessed = true;
                  }
                  nav("/create?tab=edit", { state: { abliterate: repo, abliterateFrom: guessed || repo !== item.id ? item.name : undefined } });
                }}
              >
                Abliterate…
              </button>
            )}
            {choice && secondary?.(choice)}
            <button
              disabled={!choice || busy}
              onClick={() => choice && onAction(choice)}
              className="inline-flex h-11 items-center gap-2 rounded-lg bg-signal px-5 text-[15px] font-semibold text-signal-ink transition hover:brightness-110 disabled:opacity-50"
            >
              {busy && <Loader className="size-4 animate-spin" />}
              {actionLabel}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
