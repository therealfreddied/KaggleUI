import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../lib/api";
import { bytes } from "../lib/format";
import type { CatalogItem } from "../lib/types";
import { ProgressBar } from "./meters";
import { Button, Field, Input, Select, useToast } from "./ui";

const ACTIVE = ["queued", "downloading", "extracting", "scanning", "abliterating"];
const KIND_LABEL: Record<string, string> = { hf: "Hugging Face", git: "Git", file: "File", abliterate: "Abliteration" };


export function ModelImporter({ initialUrl = "", onCustomize }: { initialUrl?: string; onCustomize: (item: CatalogItem) => void }) {
  const [url, setUrl] = useState(initialUrl);
  const [kind, setKind] = useState("auto");
  const [name, setName] = useState("");
  const [revision, setRevision] = useState("");
  const [busy, setBusy] = useState(false);
  const toast = useToast();
  const cache = useQueryClient();
  const [customizing, setCustomizing] = useState<string | null>(null);
  const query = useQuery({ queryKey: ["model-imports"], queryFn: api.modelImports,
    refetchInterval: q => q.state.data?.some(j => ACTIVE.includes(j.status)) ? 2000 : false });

  return <div className="space-y-6">
    <form className="space-y-4" onSubmit={async (event) => {
      event.preventDefault();
      setBusy(true);
      try {
        await api.importModel({ url, kind, name, revision: revision || undefined });
        await cache.invalidateQueries({ queryKey: ["model-imports"] });
        setUrl("");
        toast("Model download started");
      } catch (error) { toast(error instanceof Error ? error.message : "Download failed", "error"); }
      finally { setBusy(false); }
    }}>
      <p className="max-w-[65ch] text-[14px] text-fg-2">Bring a checkpoint from Hugging Face, a Git repository, or a direct file URL. Once it downloads, choose Customize to configure and deploy it.</p>
      <Field label="Model URL"><Input required type="url" value={url} onChange={e => setUrl(e.target.value)} placeholder="https://huggingface.co/owner/model" /></Field>
      <Field label="Download type"><Select value={kind} onChange={e => setKind(e.target.value)}>
        <option value="auto">Detect automatically</option><option value="hf">Hugging Face repository</option>
        <option value="git">Git repository</option><option value="file">Direct file / archive</option>
      </Select></Field>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Name (optional)"><Input value={name} onChange={e => setName(e.target.value)} placeholder="My model" /></Field>
        <Field label="Revision / branch (optional)"><Input value={revision} onChange={e => setRevision(e.target.value)} placeholder="Default revision" /></Field>
      </div>
      <p className="text-[12px] text-fg-3">Gated Hugging Face models use your saved HF token. Git LFS repositories need git-lfs installed. Downloading source code does not automatically create a model runner.</p>
      <Button type="submit" variant="primary" busy={busy} disabled={!url.trim()}>Download model</Button>
    </form>
    <section className="space-y-3" aria-label="Model downloads">
      <h3 className="text-[15px] font-semibold">Downloads</h3>
      {query.error && <p role="alert" className="text-[13px] text-worst">{String(query.error.message)}</p>}
      {query.isLoading && <p className="text-[13px] text-fg-3">Loading downloads…</p>}
      {query.data?.length === 0 && <p className="text-[13px] text-fg-3">Your downloads will appear here.</p>}
      {query.data?.map(job => {
        const active = ACTIVE.includes(job.status);
        return <div key={job.id} className="border-t border-line pt-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0"><h4 className="break-words text-[14px] font-medium">{job.name}</h4><p className="mt-1 text-[12px] text-fg-3">{KIND_LABEL[job.kind] ?? job.kind} · {job.status} · {bytes(job.bytes)}</p></div>
            {active && <Button size="sm" variant="ghost" onClick={async () => {
              try { await api.cancelImport(job.id); await cache.invalidateQueries({ queryKey: ["model-imports"] }); }
              catch (error) { toast(error instanceof Error ? error.message : "Cancel failed", "error"); }
            }}>Cancel</Button>}
          </div>
          {active && <div className="my-2"><ProgressBar value={job.progress} /></div>}
          <p className={`mt-2 break-words text-[12px] ${job.status === "error" ? "text-worst" : "text-fg-2"}`} role="status">{job.message}</p>
          {job.status === "complete" && job.models.map((model, index) => <div key={model.path} className="mt-3 flex items-center justify-between gap-3 rounded-lg bg-panel-2 px-3 py-2.5">
            <div className="min-w-0"><span className="block truncate text-[13px] font-medium">{model.name}</span><span className="text-[12px] text-fg-3">{model.modality} · {model.engine || "No compatible runner detected"}</span></div>
            {model.engine && <Button size="sm" busy={customizing === `${job.id}:${index}`} onClick={async () => {
              setCustomizing(`${job.id}:${index}`);
              try { onCustomize(await api.customizeImport(job.id, index)); }
              catch (error) { toast(error instanceof Error ? error.message : "Couldn't read checkpoint", "error"); }
              finally { setCustomizing(null); }
            }}>Customize →</Button>}
          </div>)}
        </div>;
      })}
    </section>
  </div>;
}
