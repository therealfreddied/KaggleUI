export type Modality = "text" | "image" | "video" | "tts" | "audio" | "stt" | "world" | "mesh";
export type Verdict = "best" | "good" | "tight" | "worst";

export interface Gpu {
  index: number;
  name: string;
  vram_total: number;
  vram_used: number;
  vram_free: number;
  util: number | null;
  temp: number | null;
  compute_capability: string | null;
  sm: number | null;
  kind?: "gpu" | "tpu";
}

export interface Disk {
  key: string;
  label: string;
  path: string;
  total: number;
  used: number;
  free: number;
  note: string;
}

export interface System {
  platform: string;
  accelerator: string;
  accelerator_label: string;
  gpus: Gpu[];
  tpus: Gpu[];
  cpu_count: number;
  ram_total: number;
  ram_available: number;
  disks: Disk[];
  run_type: string | null;
  session_started: number;
  session_limit: number;
  session_elapsed: number;
  session_remaining: number;
  internet: boolean | null;
  fake: string | null;
  notes: string[];
  vram_total_all: number;
  vram_free_all: number;
  hbm_total_all: number;
  hbm_free_all: number;
}

export interface Settings {
  access: { password_enabled: boolean };
  api: { enabled: boolean; openai: boolean; anthropic: boolean; ollama: boolean; require_key: boolean };
  exposure: { provider: TunnelProvider };
  storage: StorageSettings;
  optimize: OptimizationProfile;
  cache: CacheProfile;
  ml: MLSettings;
  workspace: WorkspaceSettings;
  quota: QuotaSettings;
  feed: FeedSettings;
}

export interface BenchmarkRecord {
  deployment: string;
  model_rev: string;
  engine: string;
  created_at: number;
  runs_per_point: number;
  best_tokens_per_s: number;
  best_ttft_s: number;
  rows: number;
  artifact_id: string;
  version: string;
}

export interface FeedSettings {
  favorites: string[];
}

export interface FeedEntry {
  repo: string;
  id: string;
  pipeline_tag?: string | null;
  downloads?: number | null;
  likes?: number | null;
  last_modified?: number | null;
  gated?: boolean;
  private?: boolean;
  error?: string | null;
}

export interface QuotaSettings {
  weekly_gpu_hours: number;
  weekly_tpu_hours: number;
  week_resets: "monday";
}

/** Measured locally by the server ledger; budgets come from settings. */
export interface QuotaStatus {
  used_gpu_hours: number;
  budget_gpu_hours: number;
  used_tpu_hours: number;
  budget_tpu_hours: number;
  session_elapsed_s: number;
  session_limit_s: number | null;
  resets_on: string;
}

export interface MLSettings {
  teacher_remote: boolean;
  max_teacher_tokens: number;
}

/** GitHub workspace sync. The token is a secret *name*, never the value. */
export interface WorkspaceSettings {
  enabled: boolean;
  repo: string;
  branch: string;
  path: string;
  token_secret: string;
  restore_on_launch: boolean;
  auto_backup: boolean;
  auto_start_set: boolean;
  publish_discovery: boolean;
}

export interface WorkspaceMeta {
  schema_version: number;
  workspace_id: string;
  name: string;
  app_version: string;
  exported_at: number;
  source_host: string | null;
}

export interface SecretRequirement {
  name: string;
  reason: string;
  required: boolean;
}

export interface WorkspaceArtifactRef {
  artifact_id: string;
  version: string;
  kind: string;
  files: number;
  size: number;
}

/** Portable configuration only; runtime state (pids, ports, GPU claims) is rebuilt locally. */
export interface WorkspaceDocument {
  meta: WorkspaceMeta;
  settings: Settings;
  deployments: DeploymentSpec[];
  deployment_sets: DeploymentSet[];
  active_set: string | null;
  storage_connections: StorageConnection[];
  artifacts: WorkspaceArtifactRef[];
  secret_requirements: SecretRequirement[];
}

export interface WorkspaceStatus {
  configured: boolean;
  repo: string;
  branch: string;
  path: string;
  token_secret: string;
  token_set: boolean;
  token_url: string;
  document_path: string;
  workspace_id: string;
  name: string;
  local_hash: string;
  local_changed: boolean;
  remote_sha: string | null;
  remote_available: boolean;
  remote_changed: boolean;
  conflict: boolean;
  remote_workspace_id?: string;
  remote_name?: string;
  last_backup_at: number | null;
  last_restore_at: number | null;
  error: string | null;
}

/** Result of linking sync to a repository (created by KaggleUI or picked from the account). */
export interface WorkspaceRepoLink {
  repo: string;
  branch: string;
  created: boolean;
  private: boolean;
  url: string;
  path: string;
  login: string | null;
  token_url: string;
}

/** Result of storing a pasted GitHub token; never contains the value. */
export interface WorkspaceTokenSave {
  ok: boolean;
  token_set: boolean;
  login: string | null;
  scopes: string[];
  token_secret: string;
  token_url: string;
  warning: string | null;
}

export interface ManagementToken {
  id: string;
  label: string;
  prefix: string;
  scopes: string[];
  created_at: number;
  last_used_at: number | null;
  revoked_at: number | null;
}

export interface InstanceInfo {
  instance_id: string;
  workspace_id: string;
  workspace_name: string;
  version: string;
  started_at: number;
  url: string;
  host: string;
  mock: boolean;
  fake_hw: string | null;
  deployments: Deployment[];
  jobs: Job[];
  uptime_s: number;
}

export interface WorkspaceDiff {
  same: boolean;
  workspace_id: { local: string; incoming: string };
  name: { local: string; incoming: string };
  settings: string[];
  deployments: { added: string[]; removed: string[]; changed: string[] };
  deployment_sets: { added: string[]; removed: string[]; changed: string[] };
  storage_connections: { added: string[]; removed: string[]; changed: string[] };
  artifacts: { incoming: number; local: number };
  secret_requirements: SecretRequirement[];
  warnings: string[];
  auto_start_set: boolean;
}

export interface WorkspacePreview {
  exists: boolean;
  sha?: string | null;
  incoming?: WorkspaceDocument;
  diff?: WorkspaceDiff;
  document_hash?: string;
}

export interface StorageSettings {
  active_connection: string | null;
  restore_on_boot: boolean;
}

/** Request-shaping stages. Everything is off unless `enabled` and the individual stage say so. */
export interface OptimizationProfile {
  enabled: boolean;
  max_prompt_tokens: number | null;
  context_strategy: "head" | "tail" | "middle" | "balanced";
  output_headroom_margin: number;
  terse_replies: boolean;
  system_prompt: string;
}

export interface CacheProfile {
  exact: boolean;
  exact_ttl_s: number;
  semantic: boolean;
  semantic_threshold: number;
  embedding_model: string | null;
}

export interface StorageConnection {
  name: string;
  kind: "local" | "rclone" | "hf" | "github";
  config: Record<string, string>;
  /** Name of the secret holding the credential, never the value itself. */
  secret: string | null;
  read_only: boolean;
}

export interface RcloneRemote {
  name: string;
  /** Configured options with secret values masked. */
  options: Record<string, string>;
}

export interface ArtifactFile {
  path: string;
  size: number;
  sha256: string;
}

/** A versioned bundle of files with provenance (model, adapter, dataset, export, projector). */
export interface ArtifactManifest {
  schema_version: number;
  artifact_id: string;
  version: string;
  kind: "model" | "adapter" | "dataset" | "export" | "projector";
  parent: string | null;
  base_revision: string | null;
  files: ArtifactFile[];
  companions: string[];
  recipe: Record<string, unknown>;
  backend: Record<string, unknown>;
  dataset: Record<string, unknown>;
  eval_summary: Record<string, unknown>;
  license: string | null;
  provenance: Record<string, unknown>;
  created_at: number;
}

export interface ArtifactVerify {
  ok: boolean;
  problems: string[];
  artifact?: ArtifactManifest;
  error?: string;
}

export type TunnelProvider = "cloudflare" | "cloudflare_named" | "ngrok" | "none";

export interface Tunnel {
  name: string;
  provider: TunnelProvider;
  label: string;
  port: number;
  url: string | null;
  status: string;
  error: string | null;
  restarts: number;
  supports_sse: boolean;
}

export interface ModelRef {
  source: "hf" | "ollama" | "attached" | "curated" | "artifact";
  id: string;
  variant: string | null;
  engine: string;
}

/** Where a text model's layers live. Engine-agnostic; llama.cpp compiles it to argv. */
export interface Placement {
  /** Physical GPU indices; null uses every GPU the deployment got. */
  devices: number[] | null;
  split_mode: "none" | "layer" | "row" | "tensor";
  tensor_split: number[] | null;
  main_gpu: number | null;
  gpu_layers: number | "auto";
  cpu_moe_layers: number | null;
  cpu_moe: boolean;
  tensor_overrides: string[];
  flash_attn: "auto" | "on" | "off";
  kv_cache_type_k: string | null;
  kv_cache_type_v: string | null;
  kv_offload: boolean;
  mmap: boolean;
  mlock: boolean;
  threads: number | null;
  batch_threads: number | null;
  parallel: number;
}

export interface DeploymentSpec {
  modality: Modality;
  model: ModelRef;
  name?: string | null;
  interface?: "studio" | "gradio" | "openwebui" | "comfyui" | "overtchat" | "none";
  options?: Record<string, unknown>;
  gpus?: number[] | null;
  /** direct = one process per deployment; llama_swap = one shared llama-swap proxy. */
  lifecycle?: "direct" | "llama_swap";
  /** manual starts now; pinned is never evicted; on_demand cold-starts per request. */
  residency?: "manual" | "pinned" | "on_demand";
  idle_timeout_s?: number;
  placement?: Placement | null;
}

export interface ResolvedPlacement {
  gpus: number[];
  gpu_layers: number;
  split_mode: string;
  tensor_split: number[] | null;
  main_gpu: number | null;
  cpu_moe_layers: number | null;
  cpu_moe: boolean;
  flash_attn: string;
  kv_k: string | null;
  kv_v: string | null;
  kv_offload: boolean;
  mmap: boolean;
  mlock: boolean;
  threads: number;
  threads_batch: number | null;
  parallel: number;
  tensor_overrides: string[];
  notes: string[];
}

export interface PlacementPreview {
  ok: boolean;
  errors: string[];
  resolved: ResolvedPlacement;
}

export type JobState = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";

export interface ForgeBody {
  base: ModelRef;
  output_name?: string;
  dataset_artifact?: string;
  stages: ("train" | "eval" | "edit" | "export" | "deploy_test")[];
  method?: "lora" | "qlora";
  eval_deployment?: string;
  edit_method?: "quick" | "heretic";
  edit_trials?: number;
  quant?: string;
  allow_quick_quant?: boolean;
  allow_blind_edit?: boolean;
}

export interface Job {
  id: string;
  kind: string;
  state: JobState;
  spec: Record<string, unknown>;
  progress: number | null;
  message: string;
  error: string | null;
  created_at: number;
  started_at: number | null;
  ended_at: number | null;
  pid: number | null;
  worker: string | null;
  artifacts: string[];
  checkpoint: string | null;
  resume_supported: boolean;
  cancel_requested: boolean;
  logs_path: string | null;
}

export interface DeploymentSet {
  id: string;
  name: string;
  deployments: DeploymentSpec[];
  created_at: number;
}

export interface WorkbenchSnapshot {
  active_set: string | null;
  sets: DeploymentSet[];
  connections: StorageConnection[];
  deployments: { id: string; spec: DeploymentSpec; served_name: string; created_at: number; runtime: Record<string, unknown> }[];
  storage: StorageSettings;
  jobs: Job[];
}

export interface DatasetReport {
  ok: boolean;
  samples: number;
  detected_format: string;
  token_estimate: number;
  errors: string[];
  warnings: string[];
}

export interface HarnessProfileInfo {
  name: string;
  version: string;
  description: string;
  cases: number;
  buildable: boolean;
}

export interface EvalCaseResult {
  id: string;
  passed: boolean;
  detail: string;
  answer: string;
  at: number;
}

export interface EvalReport {
  profile: string;
  version: string;
  score: number;
  passed: number;
  total: number;
  cases: EvalCaseResult[];
  at: number;
  artifact?: string;
  report_path?: string;
}

export interface TrainingRun {
  base: ModelRef;
  dataset_artifact: string;
  method: "lora" | "qlora";
  output_name?: string;
  r: number;
  alpha: number;
  epochs: number;
  seq_length: number;
  lr: number;
  batch_size: number;
  grad_accum: number;
  target_modules: string[];
}

export interface Deployment {
  id: string;
  spec: DeploymentSpec;
  name: string;
  modality: Modality;
  status: "queued" | "installing" | "downloading" | "loading" | "ready" | "error" | "stopping" | "stopped";
  progress: number | null;
  message: string;
  error: string | null;
  gpus: number[];
  vram_estimate: number;
  created_at: number;
  ready_at: number | null;
  info: Record<string, any>;
  interface: { kind?: string; status?: string; url?: string | null; error?: string | null; share_url?: string; message?: string };
}

export interface Overview {
  version: string;
  mock: boolean;
  fake_hw: string | null;
  system: System;
  settings: Settings;
  credentials: { admin_token: string; password: string; api_key: string };
  secrets: Record<string, boolean>;
  tunnels: Tunnel[];
  main_url: string;
  streaming_ok: boolean;
  deployments: Deployment[];
  started_at: number;
}

export interface Fit {
  verdict: Verdict;
  vram_needed: number | null;
  capacity: number | null;
  placement: "single" | "split" | "offload" | "cpu" | "none";
  /** Which hardware this verdict assumes: one card, offload, CPU only, or TPU. */
  mode: "gpu" | "gpu_cpu" | "cpu" | "tpu";
  reasons: string[];
  warnings: string[];
  disk_needed: number | null;
  score: number;
  suggest: Record<string, unknown>;
}

/** Size and fit for one exact configuration of a catalog model. */
export interface Assessment {
  key: string;
  variant: string | null;
  engine: string | null;
  size: number | null;
  fit: Fit;
}

export interface Variant {
  id: string;
  ref: string;
  label: string;
  size: number | null;
  params: number | null;
  context?: string | null;
  context_length?: number | null;
  input?: string | null;
  fit?: { verdict: Verdict; reasons: string[]; warnings?: string[] };
}

export interface CatalogBrand {
  id: string;
  label: string;
  logo: string | null;
  count?: number;
}

export interface ModelPreset {
  name: string;
  engine?: string;
  quant?: string;
  ctx?: number;
  lifecycle?: string;
  placement?: string;
  note?: string;
  options?: Record<string, unknown>;
}

export interface CatalogItem {
  brand?: CatalogBrand;
  custom_code?: boolean;
  reasoning?: import("./capabilities").ReasoningSupport;
  runtime_repos?: string[];
  extra_size?: number;
  key: string;
  source: "hf" | "ollama" | "attached";
  id: string;
  name: string;
  modality: Modality;
  engine: string | null;
  engine_label?: string;
  description: string;
  params: number | null;
  size: number | null;
  /** Bytes saved by storing the published fp32 weights as BF16 (already subtracted from size). */
  bf16_saved?: number;
  /** Full-precision Hugging Face repo to abliterate for this model (guessed for Ollama). */
  abliterate_source?: string | null;
  abliterate_guessed?: boolean;
  variant: string | null;
  variants: Variant[];
  downloads: number;
  likes: number;
  updated: number | null;
  gated: boolean;
  capabilities: string[];
  experimental: boolean;
  curated: boolean;
  selftest?: boolean;
  finetune?: boolean;
  presets?: ModelPreset[];
  /** MoE signature from the repo name; the engine still re-checks expert_count in the GGUF. */
  moe?: boolean;
  options: Record<string, unknown>;
  url: string | null;
  fit: Fit;
  spec: DeploymentSpec | null;
  library?: string | null;
  files?: number;
  context_length?: number | null;
  /** Published output cap; not the context window. */
  max_output_tokens?: number | null;
  /** Extra facts joined from models.dev when the model's own metadata has no value. */
  models_dev?: {
    id?: string;
    name?: string;
    family?: string;
    knowledge?: string;
    release_date?: string;
    last_updated?: string;
    attachment?: boolean;
    reasoning?: boolean;
    tool_call?: boolean;
    open_weights?: boolean;
    context?: number;
    output?: number;
    modalities?: { input?: string[]; output?: string[] };
  };
}

export interface SearchResult {
  filters_active?: boolean;
  facets?: { brands: CatalogBrand[]; engines: { id: string; label: string; count: number }[]; capabilities: Record<string, number> };
  modality: Modality;
  items: CatalogItem[];
  sources: Record<string, string>;
  hardware: { accelerator: string; gpus: number; tpus: number; vram_free: number; hbm_free: number; ram_available: number; disk_free: number | null; fake: string | null };
  counts?: { total: number; fits: number; shown: number; launchable?: number; hidden_unsupported?: number };
  /** Most-downloaded matches hidden because nothing here can run them (searches only). */
  hidden_examples?: string[];
  has_more?: boolean;
  /** Exact sizes are still being looked up; the same search will return more detail shortly. */
  pending?: boolean;
}

export interface CatalogMeta {
  modalities: Modality[];
  engines: Record<Modality, string[]>;
  engine_labels: Record<string, string>;
  recipes: { name: string; title: string; repo: string; controls: string | null; unverified: boolean }[];
  tunnel_providers: Record<string, string>;
}

export interface Session {
  compatibility?: { blocked: boolean; message: string };
  admin: boolean;
  access: boolean;
  password_enabled: boolean;
  version: string;
  mock: boolean;
  fake_hw: string | null;
}

export interface UpdateStatus {
  current: string;
  latest: string | null;
  available: boolean;
  blocked: boolean;
  minimum_supported: string | null;
  message: string;
  release_url: string;
  checked_at: number | null;
  error: string | null;
  progress: { status: string; message: string } | null;
}

export interface StudioModel {
  adapter?: string;
  reasoning?: import("./capabilities").ReasoningSupport;
  model_id?: string;
  source?: string;
  text_input?: boolean;
  result_kind?: "video" | "scene";
  needs_vlm?: boolean;
  prompt_conditioning?: boolean;
  id: string;
  name: string;
  modality: Modality;
  status: Deployment["status"];
  message: string;
  progress: number | null;
  capabilities: string[];
  controls: string | null;
  start_frame?: { size: string; prompt_prefix?: string } | null;
  options: Record<string, any>;
  interface: Deployment["interface"];
  engine: string;
  runtime?: string;
}

export interface SecretItem {
  name: string;
  set: boolean;
  source: "kaggle" | "colab" | "attached" | "env" | "kaggleui" | null;
  editable: boolean;
}

export interface SecretDetails {
  items: SecretItem[];
  kaggle: { state: string; error: string | null; source?: string };
  hf?: { ok: boolean; user: string | null; error: string | null };
  added?: string[];
}
export interface ModelImport {
  id: string;
  name: string;
  url: string;
  kind: string;
  created: number;
  status: string;
  progress: number | null;
  bytes: number;
  message: string;
  models: { path: string; name: string; modality: Modality; engine: string | null }[];
}
