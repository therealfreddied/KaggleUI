import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Check,
  CloudDownload,
  CloudUpload,
  ExternalLink,
  GitBranch,
  History,
  KeyRound,
  PlusCircle,
  RefreshCw,
  ShieldOff,
} from "lucide-react";
import { useEffect, useState } from "react";

import { Button, Chip, Empty, Field, Input, SectionTitle, Select, Toggle, useToast } from "../components/ui";
import { ApiError, api } from "../lib/api";
import type { ManagementToken, WorkspacePreview, WorkspaceSettings } from "../lib/types";

// GitHub's classic-token form, pre-filled with the one scope workspace sync needs.
const TOKEN_URL = "https://github.com/settings/tokens/new?scopes=repo&description=KaggleUI%20workspace";

function errorText(err: unknown) {
  if (err instanceof ApiError) return err.message;
  return err instanceof Error ? err.message : String(err);
}

function when(ts: number | null | undefined) {
  if (!ts) return "never";
  return new Date(ts * 1000).toLocaleString();
}

function SyncStatus() {
  const status = useQuery({ queryKey: ["workspace-status"], queryFn: api.workspaceStatus, retry: false });
  const data = status.data;
  if (status.error) return <p className="text-[13px] text-worst">{errorText(status.error)}</p>;
  if (!data) return <p className="text-[13px] text-fg-3">Checking the remote…</p>;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <Chip tone={data.configured ? "info" : "plain"}>{data.configured ? "Configured" : "Not configured"}</Chip>
        <Chip tone={data.token_set ? "info" : "warn"}>{data.token_set ? "Token present" : "Token missing"}</Chip>
        <Chip tone={data.remote_available ? "info" : "plain"}>
          {data.remote_available ? "Remote found" : "No remote copy"}
        </Chip>
        {data.local_changed && <Chip tone="warn">Local changes unbacked</Chip>}
        {data.remote_changed && <Chip tone="warn">Remote moved</Chip>}
        {data.conflict && <Chip tone="warn">Conflict</Chip>}
      </div>
      {data.error && (
        <p className="flex items-start gap-2 text-[13px] text-tight">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" /> {data.error}
        </p>
      )}
      <dl className="grid gap-x-6 gap-y-1.5 text-[13px] sm:grid-cols-2">
        <div className="flex justify-between gap-4"><dt className="text-fg-3">Workspace id</dt><dd className="font-mono text-[12px]">{data.workspace_id}</dd></div>
        <div className="flex justify-between gap-4"><dt className="text-fg-3">Name</dt><dd>{data.name}</dd></div>
        <div className="flex justify-between gap-4"><dt className="text-fg-3">Last backup</dt><dd>{when(data.last_backup_at)}</dd></div>
        <div className="flex justify-between gap-4"><dt className="text-fg-3">Last restore</dt><dd>{when(data.last_restore_at)}</dd></div>
        <div className="flex justify-between gap-4"><dt className="text-fg-3">Repository</dt><dd className="truncate font-mono text-[12px]">{data.repo || "—"}@{data.branch}</dd></div>
        <div className="flex justify-between gap-4"><dt className="text-fg-3">Remote revision</dt><dd className="font-mono text-[12px]">{data.remote_sha?.slice(0, 10) ?? "—"}</dd></div>
      </dl>
    </div>
  );
}

function ConfigForm({ initial }: { initial: WorkspaceSettings }) {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<WorkspaceSettings>(initial);
  const [busy, setBusy] = useState(false);
  const [token, setToken] = useState("");
  const [tokenBusy, setTokenBusy] = useState(false);
  const [newRepo, setNewRepo] = useState("kaggleui-workspace");
  const [linkBusy, setLinkBusy] = useState<"" | "create" | "existing">("");
  const status = useQuery({ queryKey: ["workspace-status"], queryFn: api.workspaceStatus, retry: false });
  const set = <K extends keyof WorkspaceSettings>(key: K, value: WorkspaceSettings[K]) =>
    setDraft((old) => ({ ...old, [key]: value }));

  function refresh() {
    queryClient.invalidateQueries({ queryKey: ["workspace-status"] });
    queryClient.invalidateQueries({ queryKey: ["workspace"] });
  }

  async function save() {
    setBusy(true);
    try {
      await api.configureWorkspace(draft);
      refresh();
      toast("Workspace settings saved");
    } catch (err) {
      toast(`Could not save: ${errorText(err)}`, "error");
    } finally {
      setBusy(false);
    }
  }

  async function saveToken() {
    setTokenBusy(true);
    try {
      const res = await api.saveWorkspaceToken(token.trim());
      setToken("");
      refresh();
      if (res.warning) toast(res.warning, "error");
      else toast(`Token accepted${res.login ? ` for ${res.login}` : ""}`);
    } catch (err) {
      toast(`GitHub did not accept that token: ${errorText(err)}`, "error");
    } finally {
      setTokenBusy(false);
    }
  }

  async function link(mode: "create" | "existing") {
    setLinkBusy(mode);
    try {
      const linked = mode === "create"
        ? await api.linkWorkspaceRepo({ mode, name: newRepo.trim(), private: true })
        : await api.linkWorkspaceRepo({ mode, repo: draft.repo.trim() });
      setDraft((old) => ({ ...old, enabled: true, repo: linked.repo, branch: linked.branch }));
      refresh();
      toast(`${linked.created ? "Created" : "Linked"} ${linked.repo} on ${linked.branch}`);
      if (mode === "create") {
        try {
          await api.backupWorkspace({ confirm: true, message: "Initial KaggleUI workspace" });
          toast("Workspace pushed to GitHub");
        } catch (err) {
          toast(`Repository is ready, but the first backup failed: ${errorText(err)}`, "error");
        }
        refresh();
      }
    } catch (err) {
      toast(`Could not link the repository: ${errorText(err)}`, "error");
    } finally {
      setLinkBusy("");
    }
  }

  return (
    <div className="space-y-4">
      <Toggle checked={draft.enabled} onChange={(v) => set("enabled", v)} label="Sync this workspace to GitHub"
        hint="Only portable configuration is synced: preferences, deployment definitions and sets, storage connections, and artifact references. Never model files, pids, ports or GPU claims." />

      <div className="space-y-2.5 rounded-lg border border-line bg-panel p-3.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-[13px] font-medium">1 · GitHub access</span>
          <Chip tone={status.data?.token_set ? "info" : "warn"}>
            {status.data?.token_set ? "Token saved" : "No token yet"}
          </Chip>
        </div>
        <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto_auto]">
          <Input type="password" value={token} onChange={(e) => setToken(e.target.value)}
            placeholder="Paste a classic token here" autoComplete="off" spellCheck={false} />
          <Button busy={tokenBusy} disabled={!token.trim()} onClick={saveToken}>
            <KeyRound className="size-4" /> Save token
          </Button>
          <a href={TOKEN_URL} target="_blank" rel="noreferrer" className="inline-flex">
            <Button><ExternalLink className="size-4" /> Create a classic token</Button>
          </a>
        </div>
        <p className="text-[12px] text-fg-3">
          The link opens GitHub's form with the <code className="font-mono">repo</code> scope pre-checked. The token is
          stored on the server (mode 600) and never sent back to this page.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-2 rounded-lg border border-line bg-panel p-3.5">
          <div className="text-[13px] font-medium">2a · Create a private repository</div>
          <div className="flex gap-2">
            <Input value={newRepo} onChange={(e) => setNewRepo(e.target.value)} placeholder="kaggleui-workspace"
              autoComplete="off" />
            <Button variant="primary" busy={linkBusy === "create"} onClick={() => link("create")}>
              <PlusCircle className="size-4" /> Create
            </Button>
          </div>
          <p className="text-[12px] text-fg-3">
            Made under the token's account with an initial commit, then this workspace is pushed to it.
          </p>
        </div>
        <div className="space-y-2 rounded-lg border border-line bg-panel p-3.5">
          <div className="text-[13px] font-medium">2b · Or use an existing repository</div>
          <div className="flex gap-2">
            <Input value={draft.repo} onChange={(e) => set("repo", e.target.value)} placeholder="owner/name"
              autoComplete="off" />
            <Button busy={linkBusy === "existing"} disabled={!draft.repo.trim()} onClick={() => link("existing")}>
              <GitBranch className="size-4" /> Link
            </Button>
          </div>
          <p className="text-[12px] text-fg-3">
            Verifies the token can push there. The workspace directory is created by the first backup.
          </p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <Field label="Branch">
          <Input value={draft.branch} onChange={(e) => set("branch", e.target.value)} placeholder="main" autoComplete="off" />
        </Field>
        <Field label="Path" hint="Repository-relative directory for this workspace's document.">
          <Input value={draft.path} onChange={(e) => set("path", e.target.value)} placeholder="workspaces/default" autoComplete="off" />
        </Field>
        <Field label="Token secret" hint="Secret name holding the token; GITHUB_TOKEN unless you renamed it.">
          <Input value={draft.token_secret} onChange={(e) => set("token_secret", e.target.value)} placeholder="GITHUB_TOKEN" autoComplete="off" />
        </Field>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <Toggle checked={draft.restore_on_launch} onChange={(v) => set("restore_on_launch", v)}
          label="Restore on launch" hint="A fresh KaggleUI instance pulls this workspace before printing the link. Local unsaved changes win." />
        <Toggle checked={draft.auto_backup} onChange={(v) => set("auto_backup", v)}
          label="Back up after changes" hint="Publishes a snapshot when settings or deployment definitions change." />
        <Toggle checked={draft.auto_start_set} onChange={(v) => set("auto_start_set", v)}
          label="Start the active set after restore" hint="Reconstructs deployments from definitions; this is not a session resume." />
        <Toggle checked={draft.publish_discovery} onChange={(v) => set("publish_discovery", v)}
          label="Publish discovery document" hint="Public metadata only, with no endpoints, tokens or model paths." />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" busy={busy} onClick={save}><Check className="size-4" /> Save workspace settings</Button>
        <span className="text-[12px] text-fg-3">Saved secrets never leave the server.</span>
      </div>
    </div>
  );
}

function PreviewCard({ preview }: { preview: WorkspacePreview | null }) {
  if (!preview) return null;
  if (!preview.exists) {
    return <p className="rounded-lg border border-line bg-panel p-3 text-[13px] text-fg-2">
      Nothing is stored in the repository yet. <span className="font-medium">Back up now</span> writes the
      first document, creating the directory automatically.
    </p>;
  }
  const diff = preview.diff;
  if (!diff) return null;
  const rows = [
    ["Deployments", diff.deployments],
    ["Deployment sets", diff.deployment_sets],
    ["Storage connections", diff.storage_connections],
  ] as const;
  return (
    <div className="space-y-3 rounded-lg border border-line bg-panel p-3.5">
      <div className="flex flex-wrap items-center gap-2 text-[13px]">
        <Chip tone="info">{diff.name.incoming}</Chip>
        <span className="text-fg-3">incoming snapshot</span>
        {diff.name.local !== diff.name.incoming && <>
          <span className="text-fg-3">·</span>
          <span className="text-fg-3">local is {diff.name.local}</span>
        </>}
        {diff.auto_start_set && <Chip tone="warn">will start active set</Chip>}
      </div>
      <div className="space-y-1.5 text-[13px]">
        {rows.map(([label, value]) => (
          <div key={label} className="flex flex-wrap items-baseline justify-between gap-2">
            <span className="text-fg-2">{label}</span>
            <span className="text-fg-3">
              +{value.added.length} · ~{value.changed.length} · −{value.removed.length}
            </span>
          </div>
        ))}
        <div className="flex justify-between"><span className="text-fg-2">Artifact references</span><span className="text-fg-3">{diff.artifacts.incoming} incoming / {diff.artifacts.local} local</span></div>
      </div>
      {diff.warnings.length > 0 && (
        <ul className="space-y-1 text-[12.5px] text-tight">
          {diff.warnings.map((w) => <li key={w} className="flex gap-1.5"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" />{w}</li>)}
        </ul>
      )}
      {diff.secret_requirements.length > 0 && (
        <div className="text-[12.5px] text-fg-2">
          <div className="mb-1 font-medium">Secrets this workspace expects</div>
          <ul className="space-y-0.5">
            {diff.secret_requirements.map((s) => (
              <li key={s.name} className="flex justify-between gap-3">
                <span className="font-mono text-[12px]">{s.name}</span>
                <span className="text-fg-3">{s.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function SyncActions({ settings }: { settings: WorkspaceSettings }) {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [preview, setPreview] = useState<WorkspacePreview | null>(null);
  const [busy, setBusy] = useState<"" | "preview" | "backup" | "restore">("");
  const [history, setHistory] = useState<{ sha: string; message: string; date: string | null; url: string | null }[] | null>(null);

  async function run(kind: "preview" | "backup" | "restore") {
    if (kind === "restore" && !window.confirm("Replace local workspace settings with the remote document? A local backup is written first.")) return;
    if (kind === "backup" && !window.confirm(`Upload this workspace to ${settings.repo}@${settings.branch}?`)) return;
    setBusy(kind);
    try {
      if (kind === "preview") {
        setPreview(await api.workspacePreview());
      } else if (kind === "backup") {
        const result = await api.backupWorkspace({ confirm: true });
        toast(result.status === "unchanged" ? "Already up to date"
          : `Workspace backed up${result.sha ? ` at ${result.sha.slice(0, 10)}` : ""}`);
        queryClient.invalidateQueries({ queryKey: ["workspace-status"] });
        queryClient.invalidateQueries({ queryKey: ["workspace"] });
        setPreview(null);
      } else {
        const result = await api.restoreWorkspace({ confirm: true });
        toast(result.status === "unchanged" ? "No remote changes" : "Workspace restored");
        queryClient.invalidateQueries({ queryKey: ["workspace-status"] });
        queryClient.invalidateQueries({ queryKey: ["workspace"] });
        setPreview(null);
      }
    } catch (err) {
      toast(`Could not ${kind === "preview" ? "preview" : kind}: ${errorText(err)}`, "error");
    } finally {
      setBusy("");
    }
  }

  async function showHistory() {
    try {
      setHistory(await api.workspaceHistory(15));
    } catch (err) {
      toast(`Could not read history: ${errorText(err)}`, "error");
    }
  }

  async function rollback(sha: string) {
    if (!window.confirm("Roll this workspace back to the selected revision? A local backup is written first.")) return;
    try {
      await api.rollbackWorkspace(sha);
      toast(`Workspace rolled back to ${sha.slice(0, 10)}`);
      setHistory(null);
    } catch (err) {
      toast(`Rollback failed: ${errorText(err)}`, "error");
    }
  }

  const disabled = !settings.enabled || !settings.repo;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Button size="sm" busy={busy === "preview"} disabled={disabled} onClick={() => run("preview")}>
          <RefreshCw className="size-3.5" /> Preview remote
        </Button>
        <Button size="sm" busy={busy === "backup"} disabled={disabled} onClick={() => run("backup")}>
          <CloudUpload className="size-3.5" /> Back up now
        </Button>
        <Button size="sm" busy={busy === "restore"} disabled={disabled} onClick={() => run("restore")}>
          <CloudDownload className="size-3.5" /> Restore from GitHub
        </Button>
        <Button size="sm" variant="ghost" onClick={showHistory}><History className="size-3.5" /> History</Button>
      </div>
      <PreviewCard preview={preview} />
      {history && (
        <ul className="divide-y divide-line rounded-lg border border-line bg-panel">
          {history.length === 0 && <li className="px-3 py-2 text-[13px] text-fg-3">No commits at this path yet.</li>}
          {history.map((row) => (
            <li key={row.sha} className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-[13px]">
              <div className="min-w-0">
                <div className="truncate">{row.message}</div>
                <div className="font-mono text-[11.5px] text-fg-3">{row.sha.slice(0, 10)} · {row.date ? new Date(row.date).toLocaleString() : "unknown date"}</div>
              </div>
              <Button size="sm" variant="ghost" onClick={() => rollback(row.sha)}>Roll back</Button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const SCOPES = ["status", "deployments", "jobs", "workspace", "storage", "shutdown"] as const;

function CliKeys() {
  const toast = useToast();
  const keys = useQuery({ queryKey: ["management"], queryFn: api.managementKeys, retry: false });
  const [label, setLabel] = useState("laptop");
  const [scopes, setScopes] = useState<string[]>(["status", "jobs"]);
  const [created, setCreated] = useState<{ token: string; record: ManagementToken } | null>(null);
  const [busy, setBusy] = useState(false);

  async function create() {
    setBusy(true);
    try {
      const result = await api.createManagementKey(label, scopes);
      setCreated({ token: result.token, record: result.record });
      toast("Management key created. Copy it now; it is shown once.");
      await keys.refetch();
    } catch (err) {
      toast(`Could not create key: ${errorText(err)}`, "error");
    } finally {
      setBusy(false);
    }
  }

  async function revoke(id: string) {
    try {
      await api.revokeManagementKey(id);
      toast("Key revoked");
      setCreated(null);
      await keys.refetch();
    } catch (err) {
      toast(`Could not revoke: ${errorText(err)}`, "error");
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-[13px] text-fg-2">
        The <code className="font-mono">kui</code> CLI and TUI use a scoped key instead of the admin token. Keys are hashed at
        rest, work only on the CLI routes, and never open the panel or the gateway.
      </p>
      <div className="grid gap-3 sm:grid-cols-[1fr_2fr_auto] sm:items-end">
        <Field label="Label"><Input value={label} onChange={(e) => setLabel(e.target.value)} /></Field>
        <div className="flex flex-wrap gap-1.5">
          {SCOPES.map((scope) => {
            const on = scopes.includes(scope);
            return (
              <button key={scope} type="button"
                onClick={() => setScopes((old) => on ? old.filter((s) => s !== scope) : [...old, scope])}
                className={`rounded-md border px-2 py-1 text-[12px] ${on ? "border-fg-3 bg-panel-2 text-fg" : "border-line text-fg-3"}`}>
                {scope}
              </button>
            );
          })}
        </div>
        <Button variant="primary" busy={busy} disabled={!label.trim() || scopes.length === 0} onClick={create}>
          <KeyRound className="size-4" /> Create key
        </Button>
      </div>
      {created && (
        <div className="space-y-1 rounded-lg border border-tight/40 bg-tight-soft p-3">
          <div className="text-[12.5px] font-medium">Shown once — store it in <code className="font-mono">kui</code>, not in Git.</div>
          <code className="block overflow-x-auto whitespace-nowrap rounded bg-panel px-2 py-1.5 font-mono text-[12px]">{created.token}</code>
        </div>
      )}
      {keys.data && keys.data.tokens.length > 0 ? (
        <ul className="divide-y divide-line rounded-lg border border-line bg-panel">
          {keys.data.tokens.map((row) => (
            <li key={row.id} className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-[13px]">
              <div>
                <div className="font-medium">{row.label} <span className="font-mono text-[11.5px] text-fg-3">{row.prefix}…</span></div>
                <div className="text-[12px] text-fg-3">{row.scopes.join(", ")} · used {row.last_used_at ? when(row.last_used_at) : "never"}</div>
              </div>
              <Button size="sm" variant="danger" onClick={() => revoke(row.id)}><ShieldOff className="size-3.5" /> Revoke</Button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[13px] text-fg-3">No CLI keys yet.</p>
      )}
    </div>
  );
}


function ModelFiles() {
  const toast = useToast();
  const queryClient = useQueryClient();
  const artifacts = useQuery({ queryKey: ["artifacts"], queryFn: api.artifacts, retry: false });
  const connections = useQuery({ queryKey: ["storage"], queryFn: api.storageConnections, retry: false });
  const [connection, setConnection] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [remoteId, setRemoteId] = useState("");
  const [busy, setBusy] = useState(false);
  const remote = useQuery({
    queryKey: ["remote-model-backups", connection],
    queryFn: () => api.remoteModelBackups(connection),
    enabled: Boolean(connection),
    retry: false,
  });
  const job = useQuery({
    queryKey: ["backup-job", jobId],
    queryFn: () => api.cliJob(jobId as string),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return jobId && !(state && ["succeeded", "failed", "cancelled", "interrupted"].includes(state))
        ? 1200 : false;
    },
  });
  const jobState = job.data?.state;
  useEffect(() => setRemoteId(""), [connection]);
  useEffect(() => {
    if (jobState !== "succeeded") return;
    queryClient.invalidateQueries({ queryKey: ["artifacts"] });
    queryClient.invalidateQueries({ queryKey: ["remote-model-backups"] });
    queryClient.invalidateQueries({ queryKey: ["model-backups"] });
  }, [jobState, queryClient]);
  const usable = (connections.data?.connections ?? []).filter((c) => c.kind === "rclone" || c.kind === "hf" || c.kind === "github");
  const toggle = (id: string) => setSelected((old) => old.includes(id) ? old.filter((x) => x !== id) : [...old, id]);

  async function runBackup() {
    if (!connection) return toast("Choose a non-local connection", "error");
    if (selected.length === 0) return toast("Select at least one artifact", "error");
    setBusy(true);
    try {
      const accepted = await api.backupModels({ connection, artifact_ids: selected });
      setJobId(accepted.id);
      toast(`Backup started as ${accepted.id}`);
    } catch (err) {
      toast(`Backup failed: ${errorText(err)}`, "error");
    } finally {
      setBusy(false);
    }
  }

  async function runRestore() {
    if (!connection || !remoteId.trim()) return toast("Choose a connection and an artifact", "error");
    setBusy(true);
    const [artifact_id, ...rest] = remoteId.trim().split("@");
    try {
      const accepted = await api.restoreModels({ connection, artifact_id, version: rest.join("@") || undefined });
      setJobId(accepted.id);
      toast(`Restore started as ${accepted.id}`);
    } catch (err) {
      toast(`Restore failed: ${errorText(err)}`, "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-[13px] text-fg-2">
        Copy model weights once to an rclone, Hugging Face or GitHub destination, then restore them instead of downloading
        from the Hub again. Uploads use <code className="font-mono">copyto</code>; local connections are never treated as a backup.
      </p>
      <Field label="Destination">
        <Select value={connection} onChange={(e) => setConnection(e.target.value)}>
          <option value="">Choose a connection…</option>
          {usable.map((c) => <option key={c.name} value={c.name}>{c.name} · {c.kind}{c.read_only ? " (read-only)" : ""}</option>)}
        </Select>
      </Field>
      {usable.length === 0 && <p className="text-[12.5px] text-tight">Add an rclone, Hugging Face or GitHub connection in Storage first.</p>}
      <div className="space-y-1.5">
        <div className="text-[12.5px] font-medium text-fg-2">Local artifacts</div>
        {(artifacts.data?.artifacts ?? []).length === 0 && <p className="text-[13px] text-fg-3">No stored artifacts yet.</p>}
        <div className="max-h-48 space-y-1 overflow-y-auto rounded-lg border border-line bg-panel p-2 scroll-thin">
          {(artifacts.data?.artifacts ?? []).map((a) => (
            <label key={`${a.artifact_id}@${a.version}`} className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-[13px] hover:bg-panel-2">
              <input type="checkbox" checked={selected.includes(`${a.artifact_id}@${a.version}`)}
                onChange={() => toggle(`${a.artifact_id}@${a.version}`)} />
              <span className="truncate font-mono text-[12px]">{a.artifact_id}@{a.version}</span>
              <Chip>{a.kind}</Chip>
            </label>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap items-end gap-2">
        <Button size="sm" variant="primary" busy={busy} onClick={runBackup}><CloudUpload className="size-3.5" /> Back up selected</Button>
        <div className="flex items-end gap-2">
          <Field label="Restore artifact">
            <Select value={remoteId} onChange={(e) => setRemoteId(e.target.value)}>
              <option value="">{remote.isFetching ? "Listing remote backups…" : "Choose a backup…"}</option>
              {(remote.data?.artifacts ?? []).map((row) => (
                <option key={`${row.artifact_id}@${row.version}`} value={`${row.artifact_id}@${row.version}`}>
                  {row.artifact_id}@{row.version}
                </option>
              ))}
            </Select>
          </Field>
          <Button size="sm" busy={busy} onClick={runRestore}><CloudDownload className="size-3.5" /> Restore</Button>
        </div>
      </div>
      {connection && remote.isError && (
        <p className="text-[12.5px] text-tight">Could not list remote backups: {errorText(remote.error)}</p>
      )}
      {connection && remote.data?.artifacts.length === 0 && !remote.isFetching && (
        <p className="text-[12.5px] text-fg-3">No backups under this destination yet.</p>
      )}
      {job.data && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-line bg-panel px-3 py-2 text-[13px]">
          <Chip tone={job.data.state === "succeeded" ? "info" : job.data.state === "failed" ? "warn" : "plain"}>{job.data.state}</Chip>
          <span className="font-mono text-[12px]">{job.data.kind}</span>
          {(job.data.error || job.data.message) && <span className="text-fg-2">{job.data.error || job.data.message}</span>}
        </div>
      )}
    </div>
  );
}

export default function Workspace() {
  const workspace = useQuery({ queryKey: ["workspace"], queryFn: api.workspace });
  const [draft, setDraft] = useState<WorkspaceSettings | null>(null);

  useEffect(() => {
    if (workspace.data && !draft) setDraft(workspace.data.settings.workspace);
  }, [workspace.data, draft]);

  if (workspace.error) return <Empty icon={<GitBranch />} title="Workspace unavailable">{errorText(workspace.error)}</Empty>;
  if (!workspace.data) return null;

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <div className="eyebrow">Portability</div>
        <h1 className="display mt-1 text-[30px]">Workspace</h1>
        <p className="mt-1 max-w-2xl text-[13.5px] text-fg-2">
          Restore dashboard preferences, deployment definitions and storage connections on a fresh instance. Model files travel
          separately; runtime state is always rebuilt locally.
        </p>
      </div>

      <section className="panel p-4">
        <SectionTitle title="GitHub sync" description="A versioned workspace document in a repository you control." />
        <div className="mt-3"><SyncStatus /></div>
      </section>

      <section className="panel space-y-4 p-4">
        <SyncActions settings={draft ?? workspace.data.settings.workspace} />
      </section>

      <section className="panel p-4">
        <SectionTitle title="Configuration" />
        <div className="mt-3">
          {draft ? <ConfigForm initial={draft} /> : <p className="text-[13px] text-fg-3">Loading…</p>}
        </div>
      </section>

      <section className="panel p-4">
        <SectionTitle title="Model files" description="Back up weights once; restore instead of re-downloading." />
        <div className="mt-3"><ModelFiles /></div>
      </section>

      <section className="panel p-4">
        <SectionTitle title="CLI access" description="Scoped keys for kui and the TUI." />
        <div className="mt-3"><CliKeys /></div>
      </section>
    </div>
  );
}

