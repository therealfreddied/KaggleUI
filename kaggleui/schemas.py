"""Shared data models: deployment specs, runtime settings and launch presets."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .catalog.categories import audio_family

Modality = Literal["text", "image", "video", "tts", "audio", "stt", "world", "mesh"]
MODALITIES: tuple[str, ...] = ("text", "image", "video", "tts", "audio", "stt", "world", "mesh")

InterfaceKind = Literal["studio", "gradio", "openwebui", "comfyui", "none"]
TunnelProvider = Literal["cloudflare", "cloudflare_named", "ngrok", "none"]


class Placement(BaseModel):
    """Where a text model's layers live. Engine-agnostic; llama.cpp compiles it to argv."""

    devices: list[int] | None = None
    """Physical GPU indices. ``None`` lets the engine use every GPU on the machine."""
    split_mode: Literal["none", "layer", "row", "tensor"] = "layer"
    tensor_split: list[float] | None = None
    main_gpu: int | None = None
    gpu_layers: int | Literal["auto"] = "auto"
    cpu_moe_layers: int | None = None
    """Offload the first N expert layers to the CPU (``--n-cpu-moe``)."""
    cpu_moe: bool = False
    """Offload every expert tensor to the CPU (``--cpu-moe``)."""
    tensor_overrides: list[str] = Field(default_factory=list)
    """Repeated ``--override-tensor`` values for the attention/KV tensors."""
    flash_attn: Literal["auto", "on", "off"] = "auto"
    kv_cache_type_k: str | None = None
    kv_cache_type_v: str | None = None
    kv_offload: bool = True
    mmap: bool = True
    mlock: bool = False
    threads: int | None = None
    batch_threads: int | None = None
    parallel: int = 1


class StorageSettings(BaseModel):
    active_connection: str | None = None
    restore_on_boot: bool = False


class OptimizationProfile(BaseModel):
    """Optional request-shaping stages. Every stage is off unless explicitly enabled."""

    enabled: bool = False
    max_prompt_tokens: int | None = None
    context_strategy: Literal["head", "tail", "middle", "balanced"] = "balanced"
    output_headroom_margin: int = 256
    terse_replies: bool = False


class CacheProfile(BaseModel):
    exact: bool = False
    exact_ttl_s: int = 3600
    semantic: bool = False
    semantic_threshold: float = 0.97
    embedding_model: str | None = None


class ArtifactFile(BaseModel):
    path: str
    size: int
    sha256: str


class ArtifactManifest(BaseModel):
    """A versioned bundle of files (model, adapter, dataset, export) with provenance."""

    schema_version: int = 1
    artifact_id: str
    version: str
    kind: Literal["model", "adapter", "dataset", "export", "projector"]
    parent: str | None = None
    base_revision: str | None = None
    files: list[ArtifactFile] = Field(default_factory=list)
    companions: list[str] = Field(default_factory=list)
    recipe: dict[str, Any] = Field(default_factory=dict)
    backend: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    eval_summary: dict[str, Any] = Field(default_factory=dict)
    license: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: float = 0.0


class StorageConnection(BaseModel):
    """A remote destination. Secret values are never stored, only the secret's name."""

    name: str
    kind: Literal["local", "rclone", "hf", "github"]
    config: dict[str, str] = Field(default_factory=dict)
    secret: str | None = None
    read_only: bool = False


class DeploymentSet(BaseModel):
    id: str = ""
    """Assigned by the server when omitted."""
    name: str
    deployments: list[DeploymentSpec] = Field(default_factory=list)
    created_at: float = 0.0


class ModelRef(BaseModel):
    source: Literal["hf", "ollama", "attached", "curated", "artifact"] = "hf"
    id: str
    """HF repo id, Ollama model name (``qwen3:8b``), or a local path for attached models."""
    variant: str | None = None
    """GGUF quant (``Q4_K_M``), Ollama tag, or diffusers dtype variant."""
    engine: str
    """Engine that runs it: ollama, vllm, sglang, transformers, diffusers, diffusers_video, diffusers_world,
    kokoro, chatterbox, transformers_tts, faster_whisper, transformers_asr, world_recipe, mock."""


class DeploymentSpec(BaseModel):
    modality: Modality
    model: ModelRef
    name: str | None = None
    """Name clients use in the ``model`` field of API requests. Defaults to a short model name."""
    interface: InterfaceKind = "studio"
    options: dict[str, Any] = Field(default_factory=dict)
    """Engine options, e.g. ``{"ctx": 8192}``, ``{"offload": "model"}``, ``{"dtype": "float16"}``."""
    gpus: list[int] | None = None
    """Pin to specific GPU indices; normally chosen automatically from free VRAM."""
    lifecycle: Literal["direct", "llama_swap"] = "direct"
    """``direct`` runs one process per deployment; ``llama_swap`` shares one llama-swap process."""
    residency: Literal["manual", "pinned", "on_demand"] = "manual"
    """``manual`` starts now and stays; ``pinned`` never gets evicted; ``on_demand`` cold-starts per request."""
    idle_timeout_s: int = 300
    placement: Placement | None = None
    """Explicit CPU/GPU placement for llama.cpp. ``None`` uses the engine's default."""

    @model_validator(mode="after")
    def check_lifecycle(self):
        if self.lifecycle == "llama_swap" and self.model.engine != "llamacpp":
            raise ValueError("llama_swap lifecycle requires the llamacpp engine")
        if self.residency == "on_demand" and self.idle_timeout_s < 30:
            raise ValueError("on_demand residency requires idle_timeout_s >= 30")
        return self

    @model_validator(mode="after")
    def classify_audio(self):
        if self.modality == "tts" and audio_family(self.model.id):
            self.modality = "audio"
            if self.model.engine == "transformers_tts":
                self.model.engine = "transformers_audio"
        return self


class AccessSettings(BaseModel):
    password_enabled: bool = True


class ApiSettings(BaseModel):
    enabled: bool = True
    openai: bool = True
    anthropic: bool = True
    ollama: bool = False
    require_key: bool = True


class ExposureSettings(BaseModel):
    provider: TunnelProvider = "cloudflare"


class MLSettings(BaseModel):
    """Model Studio switches; remote teachers are opt-in and cost nothing but tokens."""

    teacher_remote: bool = False
    max_teacher_tokens: int = 200_000

class Settings(BaseModel):
    access: AccessSettings = Field(default_factory=AccessSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    exposure: ExposureSettings = Field(default_factory=ExposureSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    optimize: OptimizationProfile = Field(default_factory=OptimizationProfile)
    cache: CacheProfile = Field(default_factory=CacheProfile)
    ml: MLSettings = Field(default_factory=MLSettings)


class Preset(BaseModel):
    """Everything needed to reproduce a setup without clicking: settings + deployments."""

    version: int = 1
    settings: Settings = Field(default_factory=Settings)
    deployments: list[DeploymentSpec] = Field(default_factory=list)


class SettingsPatch(BaseModel):
    access: dict[str, Any] | None = None
    api: dict[str, Any] | None = None
    exposure: dict[str, Any] | None = None
    storage: dict[str, Any] | None = None
    optimize: dict[str, Any] | None = None
    cache: dict[str, Any] | None = None
    ml: dict[str, Any] | None = None
