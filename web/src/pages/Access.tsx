import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { SecretsCard } from "../components/Secrets";
import { AccessForm, ApiForm, ExposurePicker, StreamingNote, type ApiState } from "../components/settings";
import { Button, CopyButton, CopyField, Field, Segmented, SectionTitle, Select, useToast } from "../components/ui";
import { api } from "../lib/api";
import { useLive } from "../lib/live";
import type { Settings, TunnelProvider } from "../lib/types";

function Card({ title, kicker, children }: { title: string; kicker?: string; children: React.ReactNode }) {
  return (
    <section className="panel min-w-0 p-5 sm:p-6">
      {kicker && <div className="readout mb-1.5">{kicker}</div>}
      <h2 className="display mb-5 text-[28px]">{title}</h2>
      {children}
    </section>
  );
}

type Lang = "curl" | "python" | "anthropic" | "js" | "ollama";

const CLAUDE_MODEL_VARS = [
  "ANTHROPIC_MODEL",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL",
  "ANTHROPIC_DEFAULT_SONNET_MODEL",
  "ANTHROPIC_DEFAULT_OPUS_MODEL",
];

function claudeEnv(base: string, key: string | null, model: string, ctx: string | null): Record<string, string> {
  const env: Record<string, string> = { ANTHROPIC_BASE_URL: base, ANTHROPIC_AUTH_TOKEN: key ?? "unused" };
  for (const name of CLAUDE_MODEL_VARS) env[name] = model;
  if (ctx) env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = ctx;
  return env;
}

function claudeScript(env: Record<string, string>): string {
  const entries = Object.entries(env)
    .map(([k, v]) => `    "${k}": ${JSON.stringify(v)},`)
    .join("\n");
  return `mkdir -p ~/.claude
python3 - <<'PY'
import json, pathlib

path = pathlib.Path.home() / ".claude" / "settings.json"
data = json.loads(path.read_text()) if path.exists() else {}
data.setdefault("env", {}).update({
${entries}
})
path.write_text(json.dumps(data, indent=2) + "\\n")
print("Wrote", path)
PY`;
}

function claudeShell(env: Record<string, string>): string {
  return Object.entries(env).map(([k, v]) => `export ${k}=${JSON.stringify(v)}`).join("\n");
}

function snippet(lang: Lang, base: string, key: string | null, model: string): string {
  const k = key ?? "unused";
  switch (lang) {
    case "curl":
      return `curl ${base}/v1/chat/completions \\
  -H "Authorization: Bearer ${k}" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello!"}]}'`;
    case "python":
      return `from openai import OpenAI

client = OpenAI(base_url="${base}/v1", api_key="${k}")
reply = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)`;
    case "anthropic":
      return `import anthropic

client = anthropic.Anthropic(base_url="${base}", api_key="${k}")
message = client.messages.create(
    model="${model}",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)

# Claude-style tools work too, e.g. Claude Code:
#   ANTHROPIC_BASE_URL=${base} ANTHROPIC_AUTH_TOKEN=${k} claude --model ${model}`;
    case "js":
      return `import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${base}/v1", apiKey: "${k}" });
const reply = await client.chat.completions.create({
  model: "${model}",
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(reply.choices[0].message.content);`;
    case "ollama":
      return `# Any Ollama client: point OLLAMA_HOST at KaggleUI
export OLLAMA_HOST=${base}
curl ${base}/api/chat -H "Authorization: Bearer ${k}" -d '{
  "model": "${model}",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'`;
  }
}

const ENDPOINTS: [string, string, string][] = [
  ["openai", "POST /v1/chat/completions", "Chat (text, vision, tools, streaming)"],
  ["openai", "POST /v1/embeddings", "Embeddings (embedding models)"],
  ["openai", "POST /v1/images/generations", "Text-to-image"],
  ["openai", "POST /v1/audio/speech", "Text-to-speech"],
  ["openai", "GET /v1/voices", "All voices from running TTS models; optionally filter by ?model=name"],
  ["openai", "POST /v1/audio/transcriptions", "Speech-to-text (json, text, srt, vtt, verbose_json)"],
  ["openai", "POST /v1/videos", "Video and world-model jobs; poll GET /v1/videos/{id}, then /content"],
  ["openai", "POST /v1/worlds", "3D world jobs; poll GET /v1/worlds/{id}, then /assets or /content"],
  ["openai", "GET /v1/models", "Running models"],
  ["anthropic", "POST /v1/messages", "Anthropic Messages API (tools, images, streaming)"],
  ["ollama", "POST /api/chat · /api/generate · /api/embed", "Ollama API"],
];

function claudeCodeEnv(base: string, key: string | null, model: string, ctx: number | null): string {
  const vars: [string, string][] = [
    ["ANTHROPIC_BASE_URL", base],
    ["ANTHROPIC_AUTH_TOKEN", key ?? "unused"],
    ["ANTHROPIC_MODEL", model],
    ["ANTHROPIC_DEFAULT_HAIKU_MODEL", model],
    ["ANTHROPIC_DEFAULT_SONNET_MODEL", model],
    ["ANTHROPIC_DEFAULT_OPUS_MODEL", model],
  ];
  if (ctx) vars.push(["CLAUDE_CODE_MAX_CONTEXT_TOKENS", String(ctx)]);
  return vars.map(([name, value]) => `export ${name}=${JSON.stringify(value)}`).join("\n");
}

function claudeCodeSettings(base: string, key: string | null, model: string, ctx: number | null): string {
  const env = claudeCodeEnv(base, key, model, ctx)
    .split("\n")
    .map((line) => line.replace(/^export /, "    "))
    .join("\n");
  return `mkdir -p ~/.claude
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / ".claude" / "settings.json"
data = json.loads(path.read_text()) if path.exists() else {}
env = data.setdefault("env", {})
merged = {
${env}
}
for name, value in merged.items():
    env[name] = value
path.write_text(json.dumps(data, indent=2) + "\\n")
PY`;
}

export default function Access() {
  const { overview, dispatch } = useLive();
  const toast = useToast();
  const [lang, setLang] = useState<Lang>("python");
  const [api_, setApi] = useState<ApiState | null>(null);
  const [locked, setLocked] = useState<boolean | null>(null);
  const [ccId, setCcId] = useState("");

  useEffect(() => {
    if (location.hash === "#secrets") window.setTimeout(() => document.getElementById("secrets")?.scrollIntoView({ behavior: "smooth" }), 150);
  }, []);

  useEffect(() => {
    if (overview && api_ === null) {
      setApi(overview.settings.api);
      setLocked(overview.settings.access.password_enabled);
    }
  }, [overview, api_]);

  if (!overview || !api_ || locked === null) return null;

  async function save(patch: Partial<{ [K in keyof Settings]: Partial<Settings[K]> }>) {
    try {
      const settings = await api.updateSettings(patch);
      dispatch({ type: "settings", data: settings });
      toast("Saved");
    } catch (e) {
      toast(e instanceof Error ? e.message : "Couldn't save", "error");
    }
  }

  const base = overview.main_url;
  const model = overview.deployments.find((d) => d.status === "ready" && d.modality === "text")?.name ?? "<model>";
  const key = api_.require_key ? overview.credentials.api_key : null;
  const surfaces = ENDPOINTS.filter(([s]) => api_[s as "openai" | "anthropic" | "ollama"]);
  const textDeps = overview.deployments.filter((d) => d.status === "ready" && d.modality === "text");
  const selectedText = textDeps.find((d) => d.id === ccId) ?? textDeps[0] ?? null;
  const ctxOption = Number(selectedText?.spec.options?.ctx);
  const textCtx = Number.isFinite(ctxOption) && ctxOption > 0 ? ctxOption : null;

  return (
    <div>
      <SectionTitle title="Access & API" description="Who can reach this machine, which APIs are open, and the keys other apps need to connect." />
      <div className="grid gap-4 xl:grid-cols-2">
        <SecretsCard />
        <Card kicker="Who can get in" title="Access">
          <div className="mb-5">
            <CopyField label="Admin token (control panel)" value={overview.credentials.admin_token} secret />
            <p className="mt-1.5 text-[12.5px] text-fg-3">Always required for this panel. Set a KAGGLEUI_ADMIN_TOKEN secret to keep it the same every session.</p>
          </div>
          <AccessForm
            enabled={locked}
            onChange={(v) => {
              setLocked(v);
              save({ access: { password_enabled: v } });
            }}
          />
          <div className="mt-5 flex items-center gap-2">
            <CopyField label="Studio link to share" value={`${base}/studio`} />
          </div>
        </Card>

        <Card kicker="Public URL" title="Exposure">
          <ExposurePicker
            value={overview.settings.exposure.provider}
            onChange={(v: TunnelProvider) => save({ exposure: { provider: v } })}
          />
          <div className="mt-4 space-y-2">
            {overview.tunnels.map((t) => (
              <div key={t.name} className="flex flex-wrap items-center gap-2 rounded-lg border border-line bg-bg-2 px-3 py-2 text-[12.5px]">
                <span className={t.status === "up" ? "dot dot-live" : "dot dot-warn"} />
                <span className="readout">{t.name}</span>
                <span className="min-w-0 flex-1 truncate font-mono text-fg-2">{t.url ?? t.status}</span>
                {t.error && <span className="w-full text-worst">{t.error}</span>}
              </div>
            ))}
            <Button size="sm" variant="ghost" onClick={() => api.restartTunnel().then(() => toast("Restarting tunnel. A quick tunnel gets a new URL."))}>
              <RefreshCw className="size-3.5" /> Restart main tunnel
            </Button>
          </div>
        </Card>

        <Card kicker="Programmatic access" title="API">
          <ApiForm
            value={api_}
            onChange={(v) => {
              setApi(v);
              save({ api: v });
            }}
          />
          <div className="mt-4">
            <StreamingNote />
          </div>
        </Card>

        <Card kicker="Copy & paste" title="Use it from code">
          <div className="mb-3">
            <CopyField label="Base URL" value={base} />
          </div>
          <Segmented<Lang>
            size="sm"
            value={lang}
            onChange={setLang}
            options={[
              { value: "python", label: "Python · OpenAI" },
              { value: "anthropic", label: "Python · Anthropic" },
              { value: "js", label: "JavaScript" },
              { value: "curl", label: "curl" },
              { value: "ollama", label: "Ollama" },
            ]}
          />
          <div className="relative mt-3">
            <pre className="overflow-x-auto rounded-lg border border-line bg-bg p-4 font-mono text-[11.5px] leading-relaxed scroll-thin">
              {snippet(lang, base, key, model)}
            </pre>
            <div className="absolute right-2 top-2">
              <CopyButton text={snippet(lang, base, key, model)} label="Copy" />
            </div>
          </div>
          {api_.enabled && (
            <div className="mt-4 divide-y divide-line rounded-lg border border-line">
              {surfaces.map(([, ep, desc]) => (
                <div key={ep} className="flex flex-col gap-0.5 px-3 py-2 sm:flex-row sm:items-baseline sm:gap-4">
                  <code className="shrink-0 font-mono text-[11.5px] text-signal">{ep}</code>
                  <span className="text-[12.5px] text-fg-2">{desc}</span>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card kicker="Client setup" title="Claude Code">
          {!selectedText ? (
            <p className="text-[13px] text-fg-3">Deploy a text model to generate the setup for it.</p>
          ) : (
            <>
              <Field label="Model">
                <Select value={selectedText.id} onChange={(e) => setCcId(e.target.value)}>
                  {textDeps.map((d) => (
                    <option key={d.id} value={d.id}>{d.name}</option>
                  ))}
                </Select>
              </Field>
              <p className="mt-2 text-[12.5px] text-fg-3">
                Claude Code speaks the Anthropic Messages API. These variables point it at this machine; the model name is substituted for its built-in Sonnet/Haiku/Opus names.
              </p>
              <div className="relative mt-3">
                <pre className="overflow-x-auto rounded-lg border border-line bg-bg p-4 font-mono text-[11.5px] leading-relaxed scroll-thin">
                  {claudeCodeEnv(base, key, selectedText.name, textCtx)}
                </pre>
                <div className="absolute right-2 top-2">
                  <CopyButton text={claudeCodeEnv(base, key, selectedText.name, textCtx)} label="Copy" />
                </div>
              </div>
              <details className="mt-3">
                <summary className="cursor-pointer text-[12.5px] text-fg-2">Persist in ~/.claude/settings.json</summary>
                <div className="relative mt-2">
                  <pre className="overflow-x-auto rounded-lg border border-line bg-bg p-4 font-mono text-[11.5px] leading-relaxed scroll-thin">
                    {claudeCodeSettings(base, key, selectedText.name, textCtx)}
                  </pre>
                  <div className="absolute right-2 top-2">
                    <CopyButton text={claudeCodeSettings(base, key, selectedText.name, textCtx)} label="Copy" />
                  </div>
                </div>
              </details>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}
