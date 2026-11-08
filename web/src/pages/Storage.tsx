import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Boxes, Check, CloudUpload, HardDriveDownload, Plus, ShieldCheck, Trash2 } from "lucide-react";
import { useState } from "react";

import { Button, Chip, Drawer, Empty, Field, Input, SectionTitle, Select, useToast } from "../components/ui";
import { api } from "../lib/api";
import { bytes } from "../lib/format";
import type { ArtifactManifest, Job, StorageConnection } from "../lib/types";

type Kind = StorageConnection["kind"];

const KIND_LABEL: Record<Kind, string> = {
  local: "Local path",
  rclone: "rclone remote",
  hf: "Hugging Face repo",
  github: "GitHub metadata",
};

const KIND_FIELDS: Record<Kind, { key: string; label: string; placeholder: string; hint?: string }[]> = {
  local: [{ key: "root", label: "Directory", placeholder: "/kaggle/working/kaggleui", hint: "Must already exist." }],
  rclone: [
    { key: "remote", label: "Remote", placeholder: "gdrive:kaggleui", hint: "Only copyto is used; sync is never run." },
    { key: "config_file", label: "rclone.conf", placeholder: "/root/.config/rclone/rclone.conf", hint: "Optional if rclone is already configured." },
  ],
  hf: [
    { key: "repo", label: "Repository", placeholder: "you/my-models" },
    { key: "repo_type", label: "Type", placeholder: "model", hint: "model or dataset" },
  ],
  github: [
    { key: "repo", label: "Repository", placeholder: "you/catalog" },
    { key: "path", label: "Directory", placeholder: "kaggleui", hint: "Text metadata only: files must be under 1 MiB." },
    { key: "branch", label: "Branch", placeholder: "main" },
  ],
};

function jobLine(job: Job | undefined) {
  if (!job) return null;
  return `${job.kind} ${job.state}${job.error ? `: ${job.error}` : job.message ? `: ${job.message}` : ""}`;
}

type BackendField = { key: string; label: string; placeholder?: string; secret?: boolean };

// Only backends rclone can configure without a browser round-trip; OAuth ones need `rclone config`.
const RCLONE_BACKENDS: { value: string; label: string; fields: BackendField[] }[] = [
  { value: "s3", label: "S3 / R2 / MinIO", fields: [
    { key: "access_key_id", label: "Access key" },
    { key: "secret_access_key", label: "Secret key", secret: true },
    { key: "endpoint", label: "Endpoint", placeholder: "https://s3.example.com" },
    { key: "region", label: "Region", placeholder: "us-east-1" },
  ] },
  { value: "b2", label: "Backblaze B2", fields: [
    { key: "account", label: "Key ID" },
    { key: "key", label: "Application key", secret: true },
  ] },
  { value: "sftp", label: "SFTP", fields: [
    { key: "host", label: "Host" },
    { key: "user", label: "User" },
    { key: "pass", label: "Password", secret: true },
    { key: "key_file", label: "Key file", placeholder: "~/.ssh/id_ed25519" },
  ] },
  { value: "webdav", label: "WebDAV", fields: [
    { key: "url", label: "URL", placeholder: "https://dav.example.com" },
    { key: "vendor", label: "Vendor", placeholder: "other" },
    { key: "user", label: "User" },
    { key: "pass", label: "Password", secret: true },
  ] },
];

function Remotes() {
  const cache = useQueryClient();
  const toast = useToast();
  const remotes = useQuery({ queryKey: ["rclone-remotes"], queryFn: api.rcloneRemotes, retry: false });
  const [backend, setBackend] = useState("s3");
  const [name, setName] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const spec = RCLONE_BACKENDS.find((b) => b.value === backend) ?? RCLONE_BACKENDS[0];

  if (remotes.error) {
    return (
      <div className="panel p-5">
        <h3 className="text-[16px] font-semibold">rclone remotes</h3>
        <p className="mt-2 text-[13px] text-fg-2">{remotes.error instanceof Error ? remotes.error.message : String(remotes.error)}</p>
      </div>
    );
  }

  async function add(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      await api.createRcloneRemote({ name, backend, params: values });
      await cache.invalidateQueries({ queryKey: ["rclone-remotes"] });
      toast(`Configured ${name}: — connect it under Destinations`);
      setName(""); setValues({});
    } catch (error) {
      toast(error instanceof Error ? error.message : "Couldn't create the remote", "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel p-5">
      <h3 className="text-[16px] font-semibold">rclone remotes</h3>
      <p className="mt-1 max-w-[65ch] text-[13px] text-fg-2">
        Credentials for rclone's own config, written non-interactively. Then add a destination whose remote is
        <span className="font-mono"> name:path</span>.
      </p>
      {(remotes.data?.remotes ?? []).length === 0 ? (
        <p className="mt-4 text-[13px] text-fg-3">No remotes configured.</p>
      ) : (
        <ul className="mt-4 divide-y divide-line">
          {(remotes.data?.remotes ?? []).map((remote) => (
            <li key={remote.name} className="flex flex-wrap items-center gap-2 py-2.5">
              <span className="font-medium">{remote.name}</span>
              <Chip>{remote.options.type ?? "rclone"}</Chip>
              <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-fg-3">
                {Object.entries(remote.options).filter(([k]) => k !== "type").map(([k, v]) => `${k}=${v}`).join("  ")}
              </span>
              <Button variant="ghost" onClick={async () => {
                if (!window.confirm(`Delete the rclone remote ${remote.name}?`)) return;
                try {
                  await api.deleteRcloneRemote(remote.name);
                  await cache.invalidateQueries({ queryKey: ["rclone-remotes"] });
                  toast(`Deleted ${remote.name}`);
                } catch (error) { toast(error instanceof Error ? error.message : "Couldn't delete", "error"); }
              }}><Trash2 className="size-4" /> Delete</Button>
            </li>
          ))}
        </ul>
      )}
      <form className="mt-4 grid gap-3 border-t border-line pt-4 sm:grid-cols-2" onSubmit={add}>
        <Field label="Name" hint="Letters, digits, - and _ only.">
          <Input required value={name} onChange={(e) => setName(e.target.value)} placeholder="archive" />
        </Field>
        <Field label="Backend">
          <Select value={backend} onChange={(e) => { setBackend(e.target.value); setValues({}); }}>
            {RCLONE_BACKENDS.map((b) => <option key={b.value} value={b.value}>{b.label}</option>)}
          </Select>
        </Field>
        {spec.fields.map((field) => (
          <Field key={field.key} label={field.label}>
            <Input value={values[field.key] ?? ""} placeholder={field.placeholder} type={field.secret ? "password" : "text"}
              onChange={(e) => setValues((prev) => ({ ...prev, [field.key]: e.target.value }))} />
          </Field>
        ))}
        <div className="sm:col-span-2"><Button type="submit" busy={busy}><Plus className="size-4" /> Add remote</Button></div>
      </form>
    </div>
  );
}

function Connections() {
  const cache = useQueryClient();
  const toast = useToast();
  const connections = useQuery({ queryKey: ["storage"], queryFn: api.storageConnections });
  const secrets = useQuery({ queryKey: ["secrets"], queryFn: api.secrets });
  const [kind, setKind] = useState<StorageConnection["kind"]>("local");
  const [name, setName] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [secret, setSecret] = useState("");
  const [readOnly, setReadOnly] = useState(false);

  const rows = connections.data?.connections ?? [];

  return (
    <div className="panel p-5">
      <h3 className="text-[16px] font-semibold">Destinations</h3>
      <p className="mt-1 max-w-[65ch] text-[13px] text-fg-2">Where artifacts are exported and restored. Credentials stay in the secret store; only the secret's name is saved here.</p>

      {rows.length === 0 ? (
        <p className="mt-4 text-[13px] text-fg-3">No destinations yet.</p>
      ) : (
        <ul className="mt-4 divide-y divide-line">
          {rows.map(connection => (
            <li key={connection.name} className="flex flex-wrap items-center gap-2 py-2.5">
              <span className="font-medium">{connection.name}</span>
              <Chip>{KIND_LABEL[connection.kind]}</Chip>
              {connection.read_only && <Chip tone="warn">read-only</Chip>}
              {connection.secret && <span className="text-[12.5px] text-fg-3">secret {connection.secret}</span>}
              <span className="min-w-0 flex-1 truncate text-[12.5px] text-fg-3">
                {Object.entries(connection.config).map(([k, v]) => `${k}=${v}`).join("  ")}
              </span>
              <Button variant="ghost" onClick={async () => {
                try {
                  const report = await api.testStorageConnection(connection.name);
                  toast(report.ok ? `${connection.name} is reachable` : `${connection.name}: ${report.error ?? "failed"}`,
                    report.ok ? "ok" : "error");
                } catch (error) { toast(error instanceof Error ? error.message : "Test failed", "error"); }
              }}>Test</Button>
              <Button variant="ghost" onClick={async () => {
                try {
                  await api.deleteStorageConnection(connection.name);
                  await cache.invalidateQueries({ queryKey: ["storage"] });
                  toast(`Removed ${connection.name}`);
                } catch (error) { toast(error instanceof Error ? error.message : "Couldn't remove", "error"); }
              }}><Trash2 className="size-4" /></Button>
            </li>
          ))}
        </ul>
      )}

      <form className="mt-5 grid gap-3 border-t border-line pt-4 sm:grid-cols-2" onSubmit={async event => {
        event.preventDefault();
        try {
          await api.saveStorageConnection({ name, kind, config: values, secret: secret || null, read_only: readOnly });
          await cache.invalidateQueries({ queryKey: ["storage"] });
          setName(""); setValues({}); setSecret(""); setReadOnly(false);
          toast(`Saved ${name}`);
        } catch (error) { toast(error instanceof Error ? error.message : "Couldn't save", "error"); }
      }}>
        <Field label="Name"><Input required value={name} onChange={e => setName(e.target.value)} placeholder="drive" /></Field>
        <Field label="Kind">
          <Select value={kind} onChange={e => { setKind(e.target.value as Kind); setValues({}); }}>
            {(Object.entries(KIND_LABEL) as [Kind, string][]).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </Select>
        </Field>
        {KIND_FIELDS[kind].map(field => (
          <Field key={field.key} label={field.label} hint={field.hint}>
            <Input value={values[field.key] ?? ""} placeholder={field.placeholder}
              onChange={e => setValues(prev => ({ ...prev, [field.key]: e.target.value }))} />
          </Field>
        ))}
        <Field label="Secret" hint="Optional. Needed for Hugging Face and GitHub uploads.">
          <Select value={secret} onChange={e => setSecret(e.target.value)}>
            <option value="">none</option>
            {(secrets.data?.items ?? []).filter(item => item.set).map(item => <option key={item.name} value={item.name}>{item.name}</option>)}
          </Select>
        </Field>
        <label className="flex items-center gap-2 self-end text-[13px]">
          <input type="checkbox" checked={readOnly} onChange={e => setReadOnly(e.target.checked)} /> read-only
        </label>
        <div className="sm:col-span-2"><Button type="submit"><Plus className="size-4" /> Add destination</Button></div>
      </form>
    </div>
  );
}

function ExportForm({ artifact, onDone }: { artifact: ArtifactManifest; onDone: () => void }) {
  const toast = useToast();
  const connections = useQuery({ queryKey: ["storage"], queryFn: api.storageConnections });
  const [connection, setConnection] = useState("");
  const [target, setTarget] = useState(`${artifact.artifact_id}/${artifact.version}`);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const rows = connections.data?.connections ?? [];
  return (
    <form className="space-y-4" onSubmit={async event => {
      event.preventDefault();
      setBusy(true);
      try {
        const job = await api.exportArtifact(artifact.artifact_id, { connection, target_path: target, version: artifact.version, confirm });
        toast(jobLine(job) ?? "Export started");
        onDone();
      } catch (error) { toast(error instanceof Error ? error.message : "Export failed", "error"); }
      finally { setBusy(false); }
    }}>
      <p className="text-[13px] text-fg-2">{artifact.artifact_id} {artifact.version} · {artifact.files.length} file(s) · {bytes(artifact.files.reduce((sum, f) => sum + f.size, 0))}</p>
      <Field label="Destination">
        <Select required value={connection} onChange={e => setConnection(e.target.value)}>
          <option value="">choose…</option>
          {rows.map(row => <option key={row.name} value={row.name}>{row.name} ({KIND_LABEL[row.kind]})</option>)}
        </Select>
      </Field>
      <Field label="Remote path" hint="Relative to the destination root.">
        <Input required value={target} onChange={e => setTarget(e.target.value)} />
      </Field>
      {!artifact.license && (
        <label className="flex items-start gap-2 text-[13px] text-fg-2">
          <input type="checkbox" checked={confirm} onChange={e => setConfirm(e.target.checked)} className="mt-0.5" />
          This artifact has no license recorded. Publish it anyway.
        </label>
      )}
      <Button type="submit" disabled={busy || !connection}><CloudUpload className="size-4" /> Export</Button>
    </form>
  );
}

function Artifacts() {
  const cache = useQueryClient();
  const toast = useToast();
  const artifacts = useQuery({ queryKey: ["artifacts"], queryFn: api.artifacts });
  const [exporting, setExporting] = useState<ArtifactManifest | null>(null);
  const [verify, setVerify] = useState<Record<string, { ok: boolean; problems: string[]; error?: string }>>({});
  const [deleting, setDeleting] = useState<ArtifactManifest | null>(null);
  const [typed, setTyped] = useState("");
  const rows = artifacts.data?.artifacts ?? [];

  return (
    <div className="panel p-5">
      <h3 className="text-[16px] font-semibold">Artifacts</h3>
      <p className="mt-1 max-w-[65ch] text-[13px] text-fg-2">Versioned bundles with a sha256 per file. Verify before shipping; delete only after a typed confirmation.</p>
      {rows.length === 0 ? (
        <div className="mt-4"><Empty icon={<Boxes className="size-5" />} title="No artifacts yet">Trained adapters, converted GGUFs and restored exports appear here.</Empty></div>
      ) : (
        <ul className="mt-4 divide-y divide-line">
          {rows.map(artifact => {
            const key = `${artifact.artifact_id}@${artifact.version}`;
            const report = verify[key];
            return (
              <li key={key} className="py-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{artifact.artifact_id}</span>
                  <Chip>{artifact.version}</Chip>
                  <Chip tone="info">{artifact.kind}</Chip>
                  {artifact.license ? null : <Chip tone="warn">no license</Chip>}
                  <span className="text-[12.5px] text-fg-3">
                    {artifact.files.length} file(s) · {bytes(artifact.files.reduce((sum, f) => sum + f.size, 0))}
                  </span>
                  <span className="flex-1" />
                  <Button variant="ghost" onClick={async () => {
                    try {
                      const result = await api.verifyArtifact(artifact.artifact_id, artifact.version);
                      setVerify(prev => ({ ...prev, [key]: result }));
                      toast(result.ok ? "All hashes match" : `${result.problems.length} problem(s)`, result.ok ? "ok" : "error");
                    } catch (error) { toast(error instanceof Error ? error.message : "Verify failed", "error"); }
                  }}><ShieldCheck className="size-4" /> Verify</Button>
                  <Button variant="ghost" onClick={() => setExporting(artifact)}><CloudUpload className="size-4" /> Export</Button>
                  <Button variant="ghost" onClick={() => { setDeleting(artifact); setTyped(""); }}><Trash2 className="size-4" /></Button>
                </div>
                {report && (
                  <p className={report.ok ? "mt-1.5 text-[12.5px] text-good" : "mt-1.5 text-[12.5px] text-worst"}>
                    {report.ok ? <><Check className="mr-1 inline size-3.5" />all files match their hashes</> : report.problems.join(", ")}
                  </p>
                )}
                <p className="mt-1.5 truncate font-mono text-[11.5px] text-fg-3">
                  {artifact.files.map(f => `${f.path} (${bytes(f.size)})`).join("  ")}
                </p>
                {deleting && deleting.artifact_id === artifact.artifact_id && deleting.version === artifact.version && (
                  <div className="mt-2 flex flex-wrap items-center gap-2 rounded-lg border border-worst/40 p-2.5">
                    <AlertTriangle className="size-4 text-worst" />
                    <span className="text-[12.5px]">Type <b>{artifact.artifact_id}</b> to delete this version.</span>
                    <Input value={typed} onChange={e => setTyped(e.target.value)} className="max-w-48" />
                    <Button disabled={typed !== artifact.artifact_id} onClick={async () => {
                      try {
                        await api.deleteArtifact(artifact.artifact_id, { version: artifact.version, confirm: true });
                        await cache.invalidateQueries({ queryKey: ["artifacts"] });
                        setDeleting(null);
                        toast(`Deleted ${artifact.artifact_id} ${artifact.version}`);
                      } catch (error) { toast(error instanceof Error ? error.message : "Delete failed", "error"); }
                    }}>Delete</Button>
                    <Button variant="ghost" onClick={() => setDeleting(null)}>Cancel</Button>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
      <Drawer open={!!exporting} onClose={() => setExporting(null)} title="Export artifact">
        {exporting && <ExportForm artifact={exporting} onDone={() => { setExporting(null); cache.invalidateQueries({ queryKey: ["jobs"] }); }} />}
      </Drawer>
    </div>
  );
}

function Restore() {
  const cache = useQueryClient();
  const toast = useToast();
  const connections = useQuery({ queryKey: ["storage"], queryFn: api.storageConnections });
  const [connection, setConnection] = useState("");
  const [remote, setRemote] = useState("");
  const [name, setName] = useState("");
  const [kind, setKind] = useState<ArtifactManifest["kind"]>("model");
  const [busy, setBusy] = useState(false);
  const rows = connections.data?.connections ?? [];
  return (
    <div className="panel p-5">
      <h3 className="text-[16px] font-semibold">Restore</h3>
      <p className="mt-1 max-w-[65ch] text-[13px] text-fg-2">Download an exported bundle back into the artifact store. Files land under the cache first; nothing streams into a registry directory.</p>
      <form className="mt-4 grid gap-3 sm:grid-cols-2" onSubmit={async event => {
        event.preventDefault();
        setBusy(true);
        try {
          const job = await api.restoreArtifact({ connection, remote, name, kind });
          toast(jobLine(job) ?? "Restore started");
          await cache.invalidateQueries({ queryKey: ["jobs"] });
        } catch (error) { toast(error instanceof Error ? error.message : "Restore failed", "error"); }
        finally { setBusy(false); }
      }}>
        <Field label="Destination">
          <Select required value={connection} onChange={e => setConnection(e.target.value)}>
            <option value="">choose…</option>
            {rows.map(row => <option key={row.name} value={row.name}>{row.name}</option>)}
          </Select>
        </Field>
        <Field label="Remote path"><Input required value={remote} onChange={e => setRemote(e.target.value)} placeholder="exports/demo/v1" /></Field>
        <Field label="Artifact name"><Input required value={name} onChange={e => setName(e.target.value)} placeholder="demo" /></Field>
        <Field label="Kind">
          <Select value={kind} onChange={e => setKind(e.target.value as ArtifactManifest["kind"])}>
            {(["model", "adapter", "dataset", "export", "projector"] as const).map(value => <option key={value} value={value}>{value}</option>)}
          </Select>
        </Field>
        <div className="sm:col-span-2"><Button type="submit" disabled={busy}><HardDriveDownload className="size-4" /> Restore</Button></div>
      </form>
    </div>
  );
}

export function StorageSections() {
  return (
    <div className="space-y-6">
      <Remotes />
      <Connections />
      <Artifacts />
      <Restore />
    </div>
  );
}