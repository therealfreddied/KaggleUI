import { useQuery, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, GraduationCap, MessagesSquare, Upload, Wand2 } from "lucide-react";
import { useState } from "react";

import { Button, Chip, Empty, Field, Input, Segmented, Select, Textarea, useToast } from "../components/ui";
import { api } from "../lib/api";
import { useLive } from "../lib/live";
import type { DatasetReport, EvalReport } from "../lib/types";

type Tab = "datasets" | "training" | "teacher" | "evaluate" | "edit" | "export";

const TABS: { value: Tab; label: string }[] = [
  { value: "datasets", label: "Datasets" },
  { value: "training", label: "Training" },
  { value: "teacher", label: "Teacher" },
  { value: "evaluate", label: "Evaluate" },
  { value: "edit", label: "Edit" },
  { value: "export", label: "Export" },
];

function Datasets() {
  const cache = useQueryClient();
  const toast = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [busy, setBusy] = useState(false);
  const datasets = useQuery({ queryKey: ["ml-datasets"], queryFn: api.mlDatasets });

  async function validate() {
    if (!file) return;
    setBusy(true);
    try {
      setReport(await api.validateDataset(file));
    } catch (error) {
      toast(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function upload() {
    if (!file) return;
    setBusy(true);
    try {
      const saved = await api.uploadDataset(file, name || file.name.replace(/\.[^.]+$/, ""));
      setReport(saved.report);
      toast(`Saved ${saved.artifact}`);
      await cache.invalidateQueries({ queryKey: ["ml-datasets"] });
    } catch (error) {
      toast(String(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="grid gap-4 rounded-xl border border-line p-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] sm:items-end">
        <Field label="File" hint="jsonl, json or csv. messages / sharegpt / alpaca shapes.">
          <input type="file" accept=".jsonl,.json,.csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="block w-full text-[13px] text-muted file:mr-3 file:rounded-md file:border file:border-line file:bg-bg-2 file:px-3 file:py-1.5 file:text-[13px] file:text-fg" />
        </Field>
        <Field label="Artifact name" hint="Defaults to the file name.">
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-dataset" />
        </Field>
        <div className="flex gap-2">
          <Button variant="ghost" busy={busy} onClick={validate} disabled={!file}>Validate</Button>
          <Button busy={busy} onClick={upload} disabled={!file}><Upload className="size-4" /> Upload</Button>
        </div>
      </div>

      {report && (
        <div className="rounded-xl border border-line p-4 text-[13px]">
          <div className="flex flex-wrap items-center gap-2">
            <Chip tone={report.ok ? "plain" : "warn"}>{report.ok ? "valid" : "invalid"}</Chip>
            <span className="text-muted">{report.samples} samples · {report.detected_format} · ~{report.token_estimate} tokens</span>
          </div>
          {report.errors.length > 0 && (
            <ul className="mt-2 list-disc space-y-1 pl-5 text-tight">{report.errors.map((e) => <li key={e}>{e}</li>)}</ul>
          )}
          {report.warnings.length > 0 && (
            <ul className="mt-2 list-disc space-y-1 pl-5 text-muted">{report.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
          )}
        </div>
      )}

      <div className="overflow-x-auto scroll-thin rounded-xl border border-line">
        <table className="w-full min-w-[560px] text-left text-[13px]">
          <thead className="bg-bg-2 text-muted">
            <tr><th className="px-3 py-2 font-medium">Artifact</th><th className="px-3 py-2 font-medium">Format</th>
              <th className="px-3 py-2 font-medium">Samples</th><th className="px-3 py-2 font-medium">Created</th></tr>
          </thead>
          <tbody>
            {(datasets.data?.datasets ?? []).map((d) => (
              <tr key={`${d.artifact_id}@${d.version}`} className="border-t border-line">
                <td className="px-3 py-2 font-mono text-[12.5px]">{d.artifact_id}</td>
                <td className="px-3 py-2 text-muted">{String(d.dataset.format ?? d.dataset.source_name ?? "—")}</td>
                <td className="px-3 py-2 text-muted">{String(d.dataset.samples ?? "—")}</td>
                <td className="px-3 py-2 text-muted">{new Date(d.created_at * 1000).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {(datasets.data?.datasets ?? []).length === 0 && (
          <p className="px-3 py-6 text-center text-[13px] text-muted">No datasets yet.</p>
        )}
      </div>
    </div>
  );
}

function Training() {
  const cache = useQueryClient();
  const toast = useToast();
  const datasets = useQuery({ queryKey: ["ml-datasets"], queryFn: api.mlDatasets });
  const jobs = useQuery({ queryKey: ["jobs", "train"], queryFn: () => api.jobs({ kind: "train", limit: 20 }), refetchInterval: 3000 });
  const [form, setForm] = useState({
    base: "Qwen/Qwen2.5-0.5B-Instruct", dataset: "", method: "lora" as "lora" | "qlora",
    r: "16", epochs: "1", seq_length: "2048", lr: "0.0002", output_name: "",
  });
  const [busy, setBusy] = useState(false);
  const set = (key: keyof typeof form) => (value: string) => setForm((f) => ({ ...f, [key]: value }));

  async function start() {
    setBusy(true);
    try {
      const job = await api.startTraining({
        base: { source: "hf", id: form.base, variant: null, engine: "transformers" },
        dataset_artifact: form.dataset,
        method: form.method,
        r: Number(form.r),
        epochs: Number(form.epochs),
        seq_length: Number(form.seq_length),
        lr: Number(form.lr),
        output_name: form.output_name || undefined,
      });
      toast(`Training ${job.id} started`);
      await cache.invalidateQueries({ queryKey: ["jobs"] });
    } catch (error) {
      toast(String(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="grid gap-4 rounded-xl border border-line p-4 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="Base model" hint="Hugging Face repo id.">
          <Input value={form.base} onChange={(e) => set("base")(e.target.value)} />
        </Field>
        <Field label="Dataset">
          <Select value={form.dataset} onChange={(e) => set("dataset")(e.target.value)}>
            <option value="">Pick a dataset artifact</option>
            {(datasets.data?.datasets ?? []).map((d) => (
              <option key={d.artifact_id} value={`${d.artifact_id}@${d.version}`}>{d.artifact_id}</option>
            ))}
          </Select>
        </Field>
        <Field label="Method" hint="QLoRA needs a CUDA GPU.">
          <Select value={form.method} onChange={(e) => set("method")(e.target.value)}>
            <option value="lora">LoRA</option><option value="qlora">QLoRA</option>
          </Select>
        </Field>
        <Field label="Output name" hint="Defaults to base-dataset-method.">
          <Input value={form.output_name} onChange={(e) => set("output_name")(e.target.value)} placeholder="auto" />
        </Field>
        <Field label="Rank"><Input value={form.r} onChange={(e) => set("r")(e.target.value)} inputMode="numeric" /></Field>
        <Field label="Epochs"><Input value={form.epochs} onChange={(e) => set("epochs")(e.target.value)} inputMode="numeric" /></Field>
        <Field label="Sequence length"><Input value={form.seq_length} onChange={(e) => set("seq_length")(e.target.value)} inputMode="numeric" /></Field>
        <Field label="Learning rate"><Input value={form.lr} onChange={(e) => set("lr")(e.target.value)} inputMode="decimal" /></Field>
        <div className="sm:col-span-2 lg:col-span-4">
          <Button busy={busy} onClick={start} disabled={!form.dataset || !form.base}>
            <GraduationCap className="size-4" /> Start training
          </Button>
        </div>
      </div>

      <div className="space-y-2">
        {(jobs.data ?? []).map((job) => (
          <div key={job.id} className="flex items-center justify-between gap-3 rounded-lg border border-line px-3 py-2 text-[13px]">
            <span className="font-mono text-[12.5px]">{job.id}</span>
            <Chip tone={job.state === "failed" ? "warn" : job.state === "succeeded" ? "plain" : "signal"}>{job.state}</Chip>
            <span className="min-w-0 flex-1 truncate text-muted">{job.error ?? job.message}</span>
            {job.resume_supported && ["failed", "cancelled", "interrupted"].includes(job.state) && (
              <Button size="sm" variant="ghost" onClick={() => api.resumeTraining(job.id)}>Resume</Button>
            )}
          </div>
        ))}
        {(jobs.data ?? []).length === 0 && <Empty title="No training runs yet">Pick a dataset and start a run.</Empty>}
      </div>
    </div>
  );
}

function Teacher() {
  const { overview } = useLive();
  const toast = useToast();
  const cache = useQueryClient();
  const [deployment, setDeployment] = useState("");
  const [prompts, setPrompts] = useState("");
  const [budget, setBudget] = useState("200000");
  const [busy, setBusy] = useState(false);
  const text = (overview?.deployments ?? []).filter((d) => d.modality === "text" && d.status === "ready");

  async function collect() {
    setBusy(true);
    try {
      const lines = prompts.split("\n").map((p) => p.trim()).filter(Boolean);
      const job = await api.startTeacher({ deployment, prompts: lines, budget_tokens: Number(budget) });
      toast(`Teacher job ${job.id} started`);
      await cache.invalidateQueries({ queryKey: ["jobs"] });
      await cache.invalidateQueries({ queryKey: ["ml-datasets"] });
    } catch (error) {
      toast(String(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="grid gap-4 rounded-xl border border-line p-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] sm:items-end">
        <Field label="Teacher deployment" hint="A ready local text model.">
          <Select value={deployment} onChange={(e) => setDeployment(e.target.value)}>
            <option value="">Pick a model</option>
            {text.map((d) => <option key={d.id} value={d.name}>{d.name}</option>)}
          </Select>
        </Field>
        <Field label="Token budget" hint="Aborts before exceeding it.">
          <Input value={budget} onChange={(e) => setBudget(e.target.value)} inputMode="numeric" />
        </Field>
        <Button busy={busy} onClick={collect} disabled={!deployment}>
          <MessagesSquare className="size-4" /> Collect
        </Button>
        <div className="sm:col-span-3">
          <Field label="Prompts" hint="One per line; answers become a dataset artifact.">
            <Textarea rows={6} value={prompts} onChange={(e) => setPrompts(e.target.value)}
              placeholder={"Name one primary color.\nExplain what a tensor is."} />
          </Field>
        </div>
      </div>
    </div>
  );
}

function Evaluate() {
  const { overview } = useLive();
  const profiles = useQuery({ queryKey: ["harness"], queryFn: api.harnessProfiles });
  const [deployment, setDeployment] = useState("");
  const [profile, setProfile] = useState("chat_basic");
  const [report, setReport] = useState<EvalReport | null>(null);
  const [busy, setBusy] = useState(false);
  const text = (overview?.deployments ?? []).filter((d) => d.modality === "text" && d.status === "ready");

  async function run() {
    setBusy(true);
    try {
      setReport(await api.runHarness(deployment, profile));
    } catch (error) {
      setReport({ profile, version: "", score: 0, passed: 0, total: 0, at: 0, cases: [], report_path: String(error) });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="grid gap-4 rounded-xl border border-line p-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] sm:items-end">
        <Field label="Deployment">
          <Select value={deployment} onChange={(e) => setDeployment(e.target.value)}>
            <option value="">Pick a model</option>
            {text.map((d) => <option key={d.id} value={d.name}>{d.name}</option>)}
          </Select>
        </Field>
        <Field label="Profile" hint="Graders are deterministic; answers are not stored.">
          <Select value={profile} onChange={(e) => setProfile(e.target.value)}>
            {(profiles.data?.profiles ?? []).map((p) => (
              <option key={p.name} value={p.name}>{p.name} ({p.cases} cases)</option>
            ))}
          </Select>
        </Field>
        <Button busy={busy} onClick={run} disabled={!deployment}><FlaskConical className="size-4" /> Run</Button>
      </div>

      {report && (
        <div className="space-y-3">
          <div className="flex items-center gap-3 text-[13px]">
            <Chip tone={report.passed === report.total ? "plain" : "warn"}>
              {report.passed}/{report.total}
            </Chip>
            <span className="text-muted">{(report.score * 100).toFixed(1)}% · {report.artifact ?? report.report_path}</span>
          </div>
          <div className="overflow-x-auto scroll-thin rounded-xl border border-line">
            <table className="w-full min-w-[520px] text-left text-[13px]">
              <thead className="bg-bg-2 text-muted">
                <tr><th className="px-3 py-2 font-medium">Case</th><th className="px-3 py-2 font-medium">Pass</th>
                  <th className="px-3 py-2 font-medium">Detail</th></tr>
              </thead>
              <tbody>
                {report.cases.map((c) => (
                  <tr key={c.id} className="border-t border-line">
                    <td className="px-3 py-2 font-mono text-[12.5px]">{c.id}</td>
                    <td className="px-3 py-2">{c.passed ? "yes" : "no"}</td>
                    <td className="px-3 py-2 text-muted">{c.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function Export() {
  const toast = useToast();
  const cache = useQueryClient();
  const [artifact, setArtifact] = useState("");
  const [quant, setQuant] = useState("Q4_K_M");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const data = useQuery({ queryKey: ["ml-convert"], queryFn: api.conversions, refetchInterval: 3000 });
  const sources = data.data?.artifacts ?? [];

  async function start() {
    setBusy(true);
    try {
      const job = await api.convertArtifact({ artifact_id: artifact, quant, out_name: name });
      toast(`Conversion ${job.id} started`);
      await cache.invalidateQueries({ queryKey: ["ml-convert"] });
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="grid gap-4 rounded-xl border border-line p-4 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="Artifact" hint="An adapter (merged automatically) or a merged model.">
          <Select value={artifact} onChange={(e) => setArtifact(e.target.value)}>
            <option value="">Pick an artifact</option>
            {sources.map((a) => (
              <option key={`${a.artifact_id}@${a.version}`} value={`${a.artifact_id}@${a.version}`}>
                {a.artifact_id} ({a.kind})
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Quantization">
          <Select value={quant} onChange={(e) => setQuant(e.target.value)}>
            <option>Q4_K_M</option><option>Q5_K_M</option><option>Q8_0</option><option>F16</option>
          </Select>
        </Field>
        <Field label="Output name" hint="Becomes gguf-&lt;name&gt;.">
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="auto" />
        </Field>
        <div className="flex items-end">
          <Button busy={busy} onClick={start} disabled={!artifact}><Wand2 className="size-4" /> Convert to GGUF</Button>
        </div>
      </div>

      <div className="space-y-2">
        {(data.data?.jobs ?? []).map((job) => (
          <div key={job.id} className="flex items-center justify-between gap-3 rounded-lg border border-line px-3 py-2 text-[13px]">
            <span className="font-mono text-[12.5px]">{job.id}</span>
            <Chip tone={job.state === "failed" ? "warn" : job.state === "succeeded" ? "plain" : "signal"}>{job.state}</Chip>
            <span className="min-w-0 flex-1 truncate text-muted">
              {job.error ?? job.message ?? (job.artifacts.length ? job.artifacts.join(", ") : "")}
            </span>
          </div>
        ))}
        {(data.data?.jobs ?? []).length === 0 && (
          <Empty title="No conversions yet">Pick an adapter or model and quantize it.</Empty>
        )}
      </div>
    </div>
  );
}

function Edit() {
  const toast = useToast();
  const cache = useQueryClient();
  const [repo, setRepo] = useState("");
  const [method, setMethod] = useState<"quick" | "heretic">("quick");
  const [trials, setTrials] = useState("30");
  const [quant, setQuant] = useState("yes");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const data = useQuery({ queryKey: ["ml-edit"], queryFn: api.edits, refetchInterval: 3000 });

  async function start() {
    setBusy(true);
    try {
      const job = await api.studioEdit({
        source_repo: repo, method, trials: Number(trials), quantize: quant === "yes", name,
      });
      toast(`Model edit ${job.id} started`);
      await cache.invalidateQueries({ queryKey: ["ml-edit"] });
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="grid gap-4 rounded-xl border border-line p-4 sm:grid-cols-2 lg:grid-cols-5">
        <Field label="Source repo" hint="A Hugging Face model to edit.">
          <Input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="org/model" />
        </Field>
        <Field label="Method">
          <Select value={method} onChange={(e) => setMethod(e.target.value as "quick" | "heretic")}>
            <option value="quick">Quick abliteration</option><option value="heretic">Heretic</option>
          </Select>
        </Field>
        <Field label="Trials" hint="10–300; more trials take longer.">
          <Input value={trials} onChange={(e) => setTrials(e.target.value)} inputMode="numeric" />
        </Field>
        <Field label="Quantize result">
          <Select value={quant} onChange={(e) => setQuant(e.target.value)}>
            <option value="yes">Yes</option><option value="no">No, keep safetensors</option>
          </Select>
        </Field>
        <Field label="Output name"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="auto" /></Field>
        <div className="sm:col-span-2 lg:col-span-5">
          <Button busy={busy} onClick={start} disabled={!repo || repo.split("/").length !== 2}>
            <Wand2 className="size-4" /> Start edit
          </Button>
        </div>
      </div>

      <div className="space-y-2">
        {(data.data?.jobs ?? []).map((job) => (
          <div key={job.id} className="flex items-center justify-between gap-3 rounded-lg border border-line px-3 py-2 text-[13px]">
            <span className="font-mono text-[12.5px]">{job.id}</span>
            <Chip tone={job.state === "failed" ? "warn" : job.state === "succeeded" ? "plain" : "signal"}>{job.state}</Chip>
            <span className="min-w-0 flex-1 truncate text-muted">{job.error ?? job.message}</span>
          </div>
        ))}
        {(data.data?.jobs ?? []).length === 0 && <Empty title="No edits yet">Abliteration runs keep their own job record.</Empty>}
      </div>
    </div>
  );
}

export default function CreatePage() {
  const wanted = new URLSearchParams(window.location.search).get("tab") as Tab | null;
  const [tab, setTab] = useState<Tab>(TABS.some((t) => t.value === wanted) ? wanted! : "datasets");
  function switchTab(next: Tab) {
    setTab(next);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", next);
    window.history.replaceState(null, "", url);
  }
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-[22px] font-semibold tracking-[-0.02em]">Model Studio</h1>
        <p className="mt-1 text-[13.5px] text-muted">
          Datasets in, adapters out: validate data, train a LoRA, collect teacher answers, score a deployment, then edit or convert a model for llama.cpp.
        </p>
      </div>
      <Segmented value={tab} options={TABS} onChange={switchTab} />
      {tab === "datasets" && <Datasets />}
      {tab === "training" && <Training />}
      {tab === "teacher" && <Teacher />}
      {tab === "evaluate" && <Evaluate />}
      {tab === "edit" && <Edit />}
      {tab === "export" && <Export />}
    </div>
  );
}

