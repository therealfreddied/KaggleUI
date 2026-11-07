"""llama.cpp engine: run a GGUF behind ``llama-server`` with explicit CPU/GPU placement.

The installer and every model download run as persisted jobs, so a cancelled or restarted deployment
leaves a job record and a log file instead of a half-installed directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

from .. import llamacpp as runtime
from ..config import Paths
from ..gguf import GGUFError, read_metadata
from ..proc import ManagedProcess
from ..schemas import DeploymentSpec
from ..util import free_port, human_bytes, wait_for
from ..workbench import placement as place
from ..workbench.registry import ArtifactRegistry
from .base import OpenAIUpstream
from .context import DeployError, EngineContext
from .download import repo_files
from .jobs import run_stage
from .llamacpp_launch import format_cmd, server_argv
from .llamacpp_models import file_list, pick

HEALTH_TIMEOUT = 900.0
LOCAL_SOURCES = ("attached", "artifact")


def _engine_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "kaggleui.llamacpp", *args]


def _cuda(ctx: EngineContext) -> bool:
    return runtime.needs_cuda(ctx.paths, ctx.hw)


async def ensure_runtime(ctx: EngineContext) -> None:
    if runtime.is_installed(ctx.paths):
        return
    if ctx.hw.accelerator == "tpu":
        raise DeployError("llama.cpp needs a CUDA GPU or a CPU; TPU sessions cannot run it")
    await run_stage(ctx, stage="installing", kind="llamacpp.install",
                    spec={"release": runtime.release.LLAMACPP_RELEASE, "cuda": _cuda(ctx)},
                    cmd=_engine_cmd("--home", str(ctx.paths.home), "install", "--cuda" if _cuda(ctx) else "--no-cuda"),
                    log_name="llamacpp-install")


async def _ensure_swap(ctx: EngineContext) -> Path:
    existing = runtime.swap_bin(ctx.paths)
    if existing is not None:
        return existing
    if sys.platform != "linux":
        raise DeployError("llama-swap prebuilt binaries are Linux-only; use a direct llama.cpp deployment")
    await run_stage(ctx, stage="installing", kind="llamacpp.swap-install",
                    spec={"release": runtime.release.LLAMA_SWAP_RELEASE},
                    cmd=_engine_cmd("--home", str(ctx.paths.home), "install-swap"), log_name="llama-swap-install")
    found = runtime.swap_bin(ctx.paths)
    if found is None:
        raise DeployError("llama-swap install finished but the binary is missing")
    return found


def local_gguf(spec: DeploymentSpec, paths: Paths) -> Path:
    if spec.model.source == "artifact":
        return _artifact_gguf(spec.model.id, spec.model.variant, paths)
    raw = Path(spec.model.id).expanduser()
    if not raw.exists():
        raise DeployError(f"{spec.model.id} is not a file on this machine")
    if raw.is_dir():
        ggups = sorted(p for p in raw.iterdir() if p.suffix == ".gguf" and "mmproj" not in p.name.lower())
        if not ggups:
            raise DeployError(f"No .gguf file inside {raw}")
        raw = ggups[0]
    if raw.suffix.lower() != ".gguf":
        raise DeployError(f"{raw.name} is not a GGUF file")
    return raw


def _artifact_gguf(artifact_id: str, version: str | None, paths: Paths) -> Path:
    """Resolve a registered artifact to its GGUF entry point (first shard when split)."""
    registry = ArtifactRegistry(paths.models / "artifacts")
    manifest = registry.get(artifact_id, version)
    if manifest is None:
        raise DeployError(f"no artifact {artifact_id}{'' if version is None else ' ' + version} is registered")
    base = registry.dir_for(manifest.artifact_id, manifest.version)
    ggups = sorted(base / f.path for f in manifest.files
                   if f.path.lower().endswith(".gguf") and "mmproj" not in f.path.lower())
    if not ggups:
        raise DeployError(f"artifact {artifact_id} contains no GGUF file; convert it first")
    return ggups[0]

async def _download_gguf(ctx: EngineContext, spec: DeploymentSpec) -> Path:
    quant = spec.options.get("quant")
    revision = spec.options.get("revision")
    ctx.progress("downloading", 0.0, f"Listing files in {spec.model.id}")
    try:
        files = await repo_files(spec.model.id, revision)
    except Exception as exc:  # network / auth / missing repo
        raise DeployError(f"Couldn't list {spec.model.id}: {exc}") from exc
    try:
        chosen = pick(files, quant=quant)
    except ValueError as exc:
        raise DeployError(str(exc)) from exc
    wanted = file_list(files, quant=quant)
    total = sum(int(f["size"] or 0) for f in chosen)
    ctx.log(f"Selected {chosen[0]['rfilename']} ({human_bytes(total)})")
    cmd = _engine_cmd("--home", str(ctx.paths.home), "fetch", "--repo", spec.model.id)
    for entry in wanted:
        cmd += ["--file", entry["path"], "--size", entry["size"] or "0"]
    if revision:
        cmd += ["--revision", str(revision)]
    await run_stage(ctx, stage="downloading", kind="llamacpp.download",
                    spec={"repo": spec.model.id, "files": [e["path"] for e in wanted], "quant": quant},
                    cmd=cmd, log_name="llamacpp-download")
    target = runtime.model_dir(ctx.paths, spec.model.id) / Path(wanted[0]["path"]).name
    if not target.exists():
        raise DeployError(f"Download finished but {target} is missing")
    return target


async def model_path(ctx: EngineContext, spec: DeploymentSpec) -> Path:
    if spec.model.source in LOCAL_SOURCES:
        return local_gguf(spec, ctx.paths)
    return await _download_gguf(ctx, spec)


def _metadata(path: Path) -> dict:
    try:
        return read_metadata(path)
    except GGUFError as exc:
        raise DeployError(f"{path.name} is not a readable GGUF: {exc}") from exc


async def _health(port: int) -> bool:
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2, read=3, write=3, pool=2)) as client:
        try:
            r = await client.get(f"http://127.0.0.1:{port}/health")
        except httpx.HTTPError:
            return False
        return r.status_code == 200


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[OpenAIUpstream, dict]:
    if spec.lifecycle == "llama_swap":
        from . import llamaswap

        return await llamaswap.deploy(spec, ctx)
    await ensure_runtime(ctx)
    gguf = await model_path(ctx, spec)
    meta = _metadata(gguf)
    errors = place.validate(spec, ctx.hw, arch=meta)
    if errors:
        raise DeployError("; ".join(errors))
    placed = place.resolve(spec, ctx.hw, arch=meta, assigned=ctx.gpus)
    for note in placed.notes:
        ctx.log(note)

    bin_path = runtime.require_bin(ctx.paths, "llama-server")
    port = free_port()
    argv = server_argv(bin_path=bin_path, model_path=gguf, host="127.0.0.1", port=port,
                       served_name=ctx.served_name, spec=spec, placed=placed)
    ctx.log(format_cmd(argv))
    ctx.progress("loading", 0.0, f"Loading {gguf.name}")
    proc = ManagedProcess(f"llama-server-{ctx.dep_id}", argv, env=ctx.env(),
                          log_path=ctx.log_path("llama-server"), on_line=ctx.log)
    ctx.processes.append(proc)
    await proc.start()
    ready = await wait_for(lambda: _health(port), timeout=HEALTH_TIMEOUT, interval=1.0)
    if not ready:
        if not proc.running:
            raise DeployError("llama-server exited while loading; see the deployment log")
        await proc.stop()
        raise DeployError(f"llama-server did not become healthy within {int(HEALTH_TIMEOUT)}s")
    ctx.progress("loading", 1.0, "Ready")
    info = {
        "runtime": "llamacpp", "release": runtime.release.LLAMACPP_RELEASE, "port": port,
        "gguf": str(gguf), "architecture": meta.get("general.architecture"),
        "placement": placed.public(), "context": spec.options.get("ctx"),
    }
    backend = OpenAIUpstream(f"http://127.0.0.1:{port}", ctx.served_name, profile=None)
    return backend, info
