import type {
  ArtifactManifest,
  BenchmarkRecord,
  CacheProfile,
  ArtifactVerify,
  DatasetReport,
  EvalReport,
  FeedEntry,
  ForgeBody,
  HarnessProfileInfo,
  TrainingRun,
  Assessment,
  CatalogItem,
  CatalogMeta,
  Deployment,
  DeploymentSet,
  DeploymentSpec,
  Job,
  Modality,
  ModelImport,
  OptimizationProfile,
  Overview,
  PlacementPreview,
  QuotaStatus,
  RcloneRemote,
  SearchResult,
  SecretDetails,
  Session,
  Settings,
  StorageConnection,
  StudioModel,
  UpdateStatus,
  WorkbenchSnapshot,
  InstanceInfo,
  ManagementToken,
  WorkspaceDocument,
  WorkspacePreview,
  WorkspaceRepoLink,
  WorkspaceSettings,
  WorkspaceStatus,
  WorkspaceTokenSave,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  const res = await fetch(path, { credentials: "same-origin", cache: "no-store", signal: AbortSignal.timeout(30_000), ...init, headers });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      const detail = data?.detail ?? data?.error?.message ?? data?.error ?? msg;
      // Conflict payloads carry {message, shas}; show the message instead of a JSON blob.
      msg = detail && typeof detail === "object" && typeof detail.message === "string" ? detail.message : detail;
      if (typeof msg !== "string") msg = JSON.stringify(msg);
    } catch {
      /* not json */
    }
    if (res.status === 405 || (res.status === 404 && path.startsWith("/_kui/api/") && msg === "Not Found")) {
      // An endpoint this page knows about but the running server doesn't: the server is outdated.
      msg = "The KaggleUI server is older than this page. In the notebook, re-run the KaggleUI cell (it restarts the server).";
    }
    throw new ApiError(res.status, msg);
  }
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

const json = (body: unknown) => JSON.stringify(body);

export const api = {
  session: () => request<Session>("/_kui/api/session"),
  login: (token: string) => request("/_kui/api/login", { method: "POST", body: json({ token }) }),
  accessLogin: (password: string) => request("/_kui/api/access-login", { method: "POST", body: json({ password }) }),
  logout: () => request("/_kui/api/logout", { method: "POST" }),
  overview: () => request<Overview>("/_kui/api/overview"),
  updates: (check = false) => request<UpdateStatus>(`/_kui/api/updates?check=${check}`),
  installUpdate: () => request<UpdateStatus>("/_kui/api/updates/install", { method: "POST" }),
  shutdown: (stopModels = true) =>
    request<{ status: string; note: string }>("/_kui/api/shutdown", {
      method: "POST", body: json({ confirm: true, stop_models: stopModels }),
    }),
  restart: () =>
    request<{ status: string; note: string }>("/_kui/api/restart", {
      method: "POST", body: json({ confirm: true }),
    }),
  updateLogs: () => request<{ lines: string[] }>("/_kui/api/updates/logs"),
  benchmarks: () => request<{ benchmarks: BenchmarkRecord[] }>("/_kui/api/benchmarks"),
  runBenchmark: (body: { deployment_id: string; ctxs?: number[]; prompts?: number[]; runs?: number }) =>
    request<Job>("/_kui/api/benchmarks", { method: "POST", body: json(body) }),
  feed: () => request<{ favorites: string[]; entries: FeedEntry[] }>("/_kui/api/feed"),
  addFeed: (repo: string) =>
    request<{ favorites: string[] }>("/_kui/api/feed", { method: "POST", body: json({ repo }) }),
  removeFeed: (repo: string) =>
    request<{ favorites: string[] }>(`/_kui/api/feed/${encodeURIComponent(repo)}`, { method: "DELETE" }),
  quota: () => request<QuotaStatus>("/_kui/api/system/quota"),
  updateSettings: (patch: Partial<{ [K in keyof Settings]: Partial<Settings[K]> }>) =>
    request<Settings>("/_kui/api/settings", { method: "PUT", body: json(patch) }),
  regenerate: (which: "password" | "api_key") =>
    request<Overview["credentials"]>("/_kui/api/credentials/regenerate", { method: "POST", body: json({ which }) }),
  meta: () => request<CatalogMeta>("/_kui/api/catalog/meta"),
  jobs: (p: { kind?: string; state?: string; limit?: number } = {}) => {
    const qs = new URLSearchParams();
    if (p.kind) qs.set("kind", p.kind);
    if (p.state) qs.set("state", p.state);
    if (p.limit) qs.set("limit", String(p.limit));
    return request<{ jobs: Job[] }>(`/_kui/api/jobs?${qs}`).then((r) => r.jobs);
  },
  job: (id: string) => request<Job>(`/_kui/api/jobs/${encodeURIComponent(id)}`),
  jobLogs: (id: string, offset = 0, limit = 400) =>
    request<{ job: Job; lines: string[] }>(
      `/_kui/api/jobs/${encodeURIComponent(id)}/logs?offset=${offset}&limit=${limit}`),
  cancelJob: (id: string) => request<Job>(`/_kui/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  resumeJob: (id: string) => request<Job>(`/_kui/api/jobs/${encodeURIComponent(id)}/resume`, { method: "POST" }),
  workbench: () => request<WorkbenchSnapshot>("/_kui/api/workbench"),
  deploymentSets: () =>
    request<{ sets: DeploymentSet[]; active: string | null }>("/_kui/api/workbench/deployment-sets"),
  createDeploymentSet: (set: Partial<DeploymentSet> & { name: string }) =>
    request<DeploymentSet>("/_kui/api/workbench/deployment-sets", { method: "POST", body: json(set) }),
  deleteDeploymentSet: (id: string) =>
    request<{ ok: boolean }>(`/_kui/api/workbench/deployment-sets/${encodeURIComponent(id)}`, { method: "DELETE" }),
  activateDeploymentSet: (id: string) =>
    request<{ active: string }>(`/_kui/api/workbench/deployment-sets/${encodeURIComponent(id)}/activate`,
      { method: "POST" }),
  // -- portability ------------------------------------------------------------------------
  workspace: () => request<{
    workspace_id: string; name: string; has_settings_file: boolean; settings: Settings;
    deployments: number; deployment_sets: number; artifacts: number; connections: number;
  }>("/_kui/api/workspace"),
  configureWorkspace: (patch: Partial<WorkspaceSettings>) =>
    request<{ workspace: WorkspaceSettings }>("/_kui/api/workspace/config", { method: "PUT", body: json(patch) }),
  exportWorkspace: () => request<WorkspaceDocument>("/_kui/api/workspace/export"),
  importWorkspace: (document: WorkspaceDocument, replace = true, replaceConnections = true) =>
    request<{ warnings: string[]; deployments: unknown }>("/_kui/api/workspace/import", {
      method: "POST", body: json({ confirm: true, document, replace, replace_connections: replaceConnections }),
    }),
  workspaceStatus: () => request<WorkspaceStatus>("/_kui/api/workspace/status"),
  workspaceGithub: () => request<{
    token_set: boolean; token_url: string; login: string | null; scopes: string[]; verified: boolean;
    error: string | null; repo: string; branch: string; path: string; enabled: boolean;
  }>("/_kui/api/workspace/github"),
  saveWorkspaceToken: (token: string) =>
    request<WorkspaceTokenSave>("/_kui/api/workspace/token", { method: "PUT", body: json({ token }) }),
  linkWorkspaceRepo: (body: { mode: "create" | "existing"; name?: string; repo?: string; private?: boolean }) =>
    request<WorkspaceRepoLink>("/_kui/api/workspace/repo", { method: "POST", body: json(body) }),
  workspacePreview: () => request<WorkspacePreview>("/_kui/api/workspace/preview"),
  backupWorkspace: (body: { force?: boolean; message?: string; confirm?: boolean } = {}) =>
    request<{ status: string; sha?: string | null; document_hash?: string; path?: string; bytes?: number }>(
      "/_kui/api/workspace/backup", { method: "POST", body: json(body) }),
  restoreWorkspace: (body: { confirm?: boolean; resolution?: "replace" | "keep_local" | "fork"; sha?: string } = {}) =>
    request<{ status: string; warnings?: string[]; diff?: unknown }>("/_kui/api/workspace/restore", {
      method: "POST", body: json(body),
    }),
  workspaceHistory: (limit = 20) =>
    request<{ sha: string; message: string; date: string | null; url: string | null }[]>(
      `/_kui/api/workspace/history?limit=${limit}`),
  rollbackWorkspace: (sha: string) =>
    request<{ status: string; warnings?: string[] }>("/_kui/api/workspace/rollback", {
      method: "POST", body: json({ sha, confirm: true }),
    }),
  managementKeys: () => request<{ tokens: ManagementToken[]; scopes: string[] }>("/_kui/api/management"),
  createManagementKey: (label: string, scopes: string[]) =>
    request<{ token: string; record: ManagementToken; notice: string }>("/_kui/api/management", {
      method: "POST", body: json({ label, scopes }),
    }),
  revokeManagementKey: (id: string) =>
    request<{ revoked: string }>(`/_kui/api/management/${encodeURIComponent(id)}`, { method: "DELETE" }),
  instance: () => request<InstanceInfo>("/_kui/api/instance"),
  cliDeployments: () => request<{ deployments: Deployment[] }>("/_kui/api/workbench/deployments"),
  modelBackups: () => request<{ artifacts: Record<string, unknown>[] }>("/_kui/api/workbench/model-backups"),
  remoteModelBackups: (connection: string, target_path = "kaggleui") =>
    request<{ artifacts: { artifact_id: string; version: string; connection: string; size?: number | null }[] }>(
      `/_kui/api/workbench/model-backups/remote?connection=${encodeURIComponent(connection)}&target_path=${encodeURIComponent(target_path)}`,
    ),
  cliJob: (id: string) => request<Job>(`/_kui/api/workbench/jobs/${encodeURIComponent(id)}`),
  backupModels: (body: { connection: string; artifact_ids?: string[]; deployment_ids?: string[]; target_path?: string }) =>
    request<Job>("/_kui/api/workbench/model-backups", { method: "POST", body: json({ ...body, confirm: true }) }),
  restoreModels: (body: { connection: string; artifact_id: string; version?: string; target_path?: string }) =>
    request<Job>("/_kui/api/workbench/model-backups/restore", {
      method: "POST", body: json({ ...body, confirm: true }),
    }),
  storageConnections: () => request<{ connections: StorageConnection[] }>("/_kui/api/workbench/storage"),
  saveStorageConnection: (connection: Partial<StorageConnection> & { name: string; kind: StorageConnection["kind"] }) =>
    request<StorageConnection>("/_kui/api/workbench/storage", { method: "POST", body: json(connection) }),
  deleteStorageConnection: (name: string) =>
    request<{ ok: boolean }>(`/_kui/api/workbench/storage/${encodeURIComponent(name)}`, { method: "DELETE" }),
  testStorageConnection: (name: string) =>
    request<{ ok: boolean; error?: string; [key: string]: unknown }>(
      `/_kui/api/workbench/storage/${encodeURIComponent(name)}/test`, { method: "POST" }),
  rcloneRemotes: () => request<{ remotes: RcloneRemote[] }>("/_kui/api/storage/remotes"),
  createRcloneRemote: (body: { name: string; backend: string; params: Record<string, string> }) =>
    request<RcloneRemote>("/_kui/api/storage/remotes", { method: "POST", body: json(body) }),
  deleteRcloneRemote: (name: string) =>
    request<{ deleted: string }>(`/_kui/api/storage/remotes/${encodeURIComponent(name)}`, { method: "DELETE" }),
  artifacts: () => request<{ artifacts: ArtifactManifest[] }>("/_kui/api/workbench/artifacts"),
  artifact: (id: string, version?: string) =>
    request<ArtifactManifest>(`/_kui/api/workbench/artifacts/${encodeURIComponent(id)}` +
      (version ? `?version=${encodeURIComponent(version)}` : "")),
  verifyArtifact: (id: string, version?: string) =>
    request<ArtifactVerify>(`/_kui/api/workbench/artifacts/${encodeURIComponent(id)}/verify`,
      { method: "POST", body: json({ version }) }),
  exportArtifact: (id: string, body: { connection: string; target_path?: string; version?: string; confirm?: boolean }) =>
    request<Job>(`/_kui/api/workbench/artifacts/${encodeURIComponent(id)}/export`, { method: "POST", body: json(body) }),
  deleteArtifact: (id: string, body: { version: string; confirm: boolean }) =>
    request<{ ok: boolean }>(`/_kui/api/workbench/artifacts/${encodeURIComponent(id)}/delete`,
      { method: "POST", body: json(body) }),
  restoreArtifact: (body: { connection: string; remote: string; name: string; kind?: ArtifactManifest["kind"] }) =>
    request<Job>("/_kui/api/workbench/restore", { method: "POST", body: json(body) }),
  optimizeSettings: () =>
    request<{ optimize: OptimizationProfile; cache: CacheProfile }>("/_kui/api/workbench/optimize"),
  saveOptimizeSettings: (patch: { optimize?: Partial<OptimizationProfile>; cache?: Partial<CacheProfile> }) =>
    request<{ optimize: OptimizationProfile; cache: CacheProfile }>("/_kui/api/workbench/optimize",
      { method: "PUT", body: json(patch) }),
  placementPreview: (body: { spec: unknown; gguf?: string }) =>
    request<PlacementPreview>("/_kui/api/workbench/placement/preview", { method: "POST", body: json(body) }),
  modelImports: () => request<ModelImport[]>("/_kui/api/catalog/imports"),
  importModel: (body: { url: string; kind?: string; name?: string; revision?: string }) =>
    request<ModelImport>("/_kui/api/catalog/imports", { method: "POST", body: json(body) }),
  original: (name: string, tag: string) =>
    request<{ repo: string | null }>(`/_kui/api/catalog/original?name=${encodeURIComponent(name)}&tag=${encodeURIComponent(tag)}`),
  abliterate: (body: { repo: string; method: "quick" | "heretic"; trials?: number; quantize?: boolean; name?: string }) =>
    request<ModelImport>("/_kui/api/catalog/abliterate", { method: "POST", body: json(body) }),
  cancelImport: (id: string) => request<ModelImport>(`/_kui/api/catalog/imports/${id}`, { method: "DELETE" }),
  customizeImport: (id: string, index: number) => request<CatalogItem>(`/_kui/api/catalog/imports/${id}/models/${index}`),
  search: (p: {
    modality: Modality;
    q?: string;
    source?: string;
    sort?: string;
    show_unsupported?: boolean;
    fit?: "all" | "fits";
    limit?: number;
    live?: boolean;
    min_params?: number;
    max_params?: number;
    min_ctx?: number;
    capabilities?: string[];
    capability_match?: "all" | "any";
    brand?: string;
    engine?: string;
    include_unknown?: boolean;
    hide_gated?: boolean;
    hide_experimental?: boolean;
  }) => {
    const qs = new URLSearchParams({
      modality: p.modality,
      q: p.q ?? "",
      source: p.source ?? "all",
      sort: p.sort ?? "recommended",
      show_unsupported: String(Boolean(p.show_unsupported)),
      fit: p.fit ?? "all",
      limit: String(p.limit ?? 40),
      live: String(p.live ?? true),
    });
    for (const key of ["min_params", "max_params", "min_ctx"] as const) if (p[key] !== undefined) qs.set(key, String(p[key]));
    if (p.capabilities?.length) qs.set("capabilities", p.capabilities.join(","));
    if (p.capability_match) qs.set("capability_match", p.capability_match);
    if (p.brand) qs.set("brand", p.brand);
    if (p.engine) qs.set("engine", p.engine);
    for (const key of ["include_unknown", "hide_gated", "hide_experimental"] as const) if (p[key]) qs.set(key, "true");
    return request<SearchResult>(`/_kui/api/catalog/search?${qs}`);
  },
  detail: (key: string) => request<CatalogItem>(`/_kui/api/catalog/detail?key=${encodeURIComponent(key)}`),
  assess: (body: { key: string; variant?: string | null; engine?: string | null; options?: Record<string, unknown> }) =>
    request<Assessment>("/_kui/api/catalog/assess", { method: "POST", body: JSON.stringify(body) }),
  deploy: (spec: DeploymentSpec) => request<Deployment>("/_kui/api/deployments", { method: "POST", body: json(spec) }),
  stop: (id: string) => request(`/_kui/api/deployments/${id}`, { method: "DELETE" }),
  restart: (id: string) => request<Deployment>(`/_kui/api/deployments/${id}/restart`, { method: "POST" }),
  logs: (id: string) => request<{ lines: string[] }>(`/_kui/api/deployments/${id}/logs`),
  restartTunnel: () => request("/_kui/api/tunnels/main/restart", { method: "POST" }),
  preset: () => request<{ preset: unknown; cell: string }>("/_kui/api/preset"),
  studioModels: () => request<StudioModel[]>("/_kui/api/studio/models"),
  secrets: () => request<SecretDetails>("/_kui/api/secrets"),
  setSecret: (name: string, value: string) =>
    request<SecretDetails>(`/_kui/api/secrets/${name}`, { method: "PUT", body: json({ value }) }),
  clearSecret: (name: string) => request<SecretDetails>(`/_kui/api/secrets/${name}`, { method: "DELETE" }),
  reloadSecrets: () => request<SecretDetails>("/_kui/api/secrets/reload", { method: "POST" }),
  mlDatasets: () => request<{ datasets: ArtifactManifest[] }>("/_kui/api/workbench/ml/datasets"),
  uploadDataset: async (file: File, name: string): Promise<{ artifact: string; report: DatasetReport }> => {
    const form = new FormData();
    form.append("file", file);
    return request(`/_kui/api/workbench/ml/datasets?name=${encodeURIComponent(name)}`, { method: "POST", body: form });
  },
  validateDataset: async (file: File): Promise<DatasetReport> => {
    const form = new FormData();
    form.append("file", file);
    return request("/_kui/api/workbench/ml/datasets/validate", { method: "POST", body: form });
  },
  startTraining: (run: Partial<TrainingRun>) =>
    request<Job>("/_kui/api/workbench/ml/train", { method: "POST", body: json(run) }),
  cancelTraining: (id: string) =>
    request<Job>(`/_kui/api/workbench/ml/train/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  resumeTraining: (id: string) =>
    request<Job>(`/_kui/api/workbench/ml/train/${encodeURIComponent(id)}/resume`, { method: "POST" }),
  conversions: () => request<{ jobs: Job[]; artifacts: ArtifactManifest[] }>("/_kui/api/workbench/ml/convert"),
  convertArtifact: (body: { artifact_id: string; quant: string; out_name?: string }) =>
    request<Job>("/_kui/api/workbench/ml/convert", { method: "POST", body: json(body) }),
  edits: () => request<{ jobs: Job[] }>("/_kui/api/workbench/ml/edit"),
  studioEdit: (body: { source_repo: string; method: "quick" | "heretic"; trials?: number; quantize?: boolean; name?: string }) =>
    request<Job>("/_kui/api/workbench/ml/edit", { method: "POST", body: json(body) }),
  forges: () => request<{ jobs: Job[] }>("/_kui/api/workbench/ml/forge"),
  startForge: (body: ForgeBody) =>
    request<Job>("/_kui/api/workbench/ml/forge", { method: "POST", body: json(body) }),
  cancelForge: (id: string) =>
    request<Job>(`/_kui/api/workbench/ml/forge/${id}/cancel`, { method: "POST" }),
  resumeForge: (id: string) =>
    request<Job>(`/_kui/api/workbench/ml/forge/${encodeURIComponent(id)}/resume`, { method: "POST" }),
  harnessProfiles: () => request<{ profiles: HarnessProfileInfo[] }>("/_kui/api/workbench/ml/harness"),
  runHarness: (deployment: string, profile: string) =>
    request<EvalReport>("/_kui/api/workbench/ml/harness", { method: "POST", body: json({ deployment, profile }) }),
  teacherDatasets: () =>
    request<{ jobs: Job[]; datasets: ArtifactManifest[] }>("/_kui/api/workbench/ml/teacher"),
  startTeacher: (body: { deployment: string; prompts: string[]; name?: string; max_new_tokens?: number; budget_tokens?: number }) =>
    request<Job>("/_kui/api/workbench/ml/teacher", { method: "POST", body: json(body) }),
};

/** Calls to the public gateway from the Studio (cookie auth + studio header). */
export const gateway = {
  async post(path: string, body: unknown): Promise<Response> {
    const res = await fetch(path, {
      method: "POST",
      credentials: "same-origin",
      headers: { "content-type": "application/json", "x-kui-studio": "1" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new ApiError(res.status, await errorText(res));
    return res;
  },
  async form(path: string, form: FormData): Promise<Response> {
    const res = await fetch(path, { method: "POST", credentials: "same-origin", headers: { "x-kui-studio": "1" }, body: form });
    if (!res.ok) throw new ApiError(res.status, await errorText(res));
    return res;
  },
  async get(path: string): Promise<Response> {
    const res = await fetch(path, { credentials: "same-origin", headers: { "x-kui-studio": "1" } });
    if (!res.ok) throw new ApiError(res.status, await errorText(res));
    return res;
  },
};

async function errorText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data?.error?.message ?? data?.detail ?? data?.error ?? res.statusText;
  } catch {
    return res.statusText;
  }
}
