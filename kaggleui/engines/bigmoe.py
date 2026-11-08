"""BigMoeOnEdge engine: a MoE GGUF whose routed experts stream from disk instead of sitting in VRAM.

This is the option for a Mixture-of-Experts model too large for the card: ``bmoe-server`` keeps the dense
weights and a small expert cache resident and reads expert rows per token, which is slower than llama.cpp
but runs models that otherwise would not fit at all. The server speaks OpenAI chat-completions and
Anthropic messages natively, so the gateway can proxy both without translation.

Params the engine adds to every launch: ``--moe-stream --overlap`` (the expert streamer), and
``--cache-mb`` when the deployment sets one. Context comes from ``options.ctx``.
"""

from __future__ import annotations

import asyncio
import hashlib
import platform
import shutil
import sys
import tarfile
from pathlib import Path

import httpx

from ..config import Paths
from ..gguf import GGUFError, expert_count, is_moe, read_metadata
from ..proc import ManagedProcess
from ..schemas import DeploymentSpec
from ..util import free_port, human_bytes, wait_for
from .base import OpenAIUpstream
from .bigmoe_release import ASSETS, BMOE_RELEASE, RELEASE_BASE
from .context import DeployError, EngineContext
from .llamacpp import model_path

HEALTH_TIMEOUT = 900.0
SERVER = "bmoe-server"

def _arch() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)

def _glibc() -> tuple[int, int] | None:
    try:
        raw = platform.libc_ver()[1]
        major, _, minor = raw.partition(".")
        return int(major), int(minor)
    except (ValueError, IndexError):
        return None

def runtime_dir(paths: Paths) -> Path:
    return paths.home / "bmoe" / BMOE_RELEASE

def installed(paths: Paths) -> Path | None:
    binary = runtime_dir(paths) / SERVER
    return binary if binary.is_file() else None

def _safe_extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve()) + "/"):
                raise DeployError(
                    f"{archive.name} carries an entry outside the install folder: {member.name}"
                )
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # Python 3.10/3.11: the filter argument is 3.12+
            tar.extractall(dest)

async def _download(ctx: EngineContext, name: str, sha256: str, dest: Path) -> None:
    url = f"{RELEASE_BASE}/{name}"
    digest = hashlib.sha256()
    ctx.progress("installing", 0.0, f"Downloading {name}")
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=15, read=120, write=60, pool=10),
                                 follow_redirects=True) as client:
        async with client.stream("GET", url) as r:
            if r.status_code != 200:
                raise DeployError(f"BigMoeOnEdge download failed ({r.status_code}) for {url}")
            total = int(r.headers.get("content-length") or 0) or None
            done = 0
            with dest.open("wb") as fh:
                async for chunk in r.aiter_bytes(1 << 20):
                    digest.update(chunk)
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        ctx.progress("installing", 0.6 * done / total,
                                     f"Downloading {name} · {human_bytes(done)} of {human_bytes(total)}")
    if digest.hexdigest() != sha256:
        dest.unlink(missing_ok=True)
        raise DeployError(f"{name} does not match the pinned sha256; refusing to run it")

async def ensure_runtime(ctx: EngineContext) -> Path:
    existing = installed(ctx.paths)
    if existing is not None:
        return existing
    if sys.platform != "linux":
        raise DeployError("BigMoeOnEdge ships Linux builds only; use llama.cpp on this host")
    arch = _arch()
    try:
        name, sha256, min_glibc, platforms = ASSETS[arch]
    except KeyError:
        raise DeployError(f"BigMoeOnEdge publishes no Linux build for {arch}") from None
    libc = _glibc()
    if libc is not None and libc < min_glibc:
        raise DeployError(
            f"BigMoeOnEdge {BMOE_RELEASE} needs glibc {min_glibc[0]}.{min_glibc[1]} on {arch} ({platforms}); "
            f"this host has {libc[0]}.{libc[1]}. Run it on a newer image or build from source."
        )
    target = runtime_dir(ctx.paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / name
    if not archive.is_file():
        await _download(ctx, name, sha256, archive)
    staging = target.parent / f"{BMOE_RELEASE}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    ctx.progress("installing", 0.9, f"Unpacking {name}")
    await asyncio.to_thread(_safe_extract, archive, staging)
    inner = next((p for p in staging.iterdir() if p.is_dir()), staging)
    shutil.rmtree(target, ignore_errors=True)
    inner.rename(target)
    shutil.rmtree(staging, ignore_errors=True)
    binary = installed(ctx.paths)
    if binary is None:
        raise DeployError(f"{name} carries no {SERVER} binary")
    binary.chmod(0o755)
    archive.unlink(missing_ok=True)
    ctx.progress("installing", 1.0, f"BigMoeOnEdge {BMOE_RELEASE} installed")
    return binary

def _metadata(path: Path) -> dict:
    try:
        return read_metadata(path)
    except GGUFError as exc:
        raise DeployError(f"{path.name} is not a readable GGUF: {exc}") from exc

async def _models(port: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2, read=5, write=3, pool=2)) as client:
        r = await client.get(f"http://127.0.0.1:{port}/v1/models")
        if r.status_code != 200:
            return []
        data = r.json().get("data")
        return data if isinstance(data, list) else []

async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[OpenAIUpstream, dict]:
    binary = await ensure_runtime(ctx)
    gguf = await model_path(ctx, spec)
    meta = _metadata(gguf)
    if not is_moe(meta):
        raise DeployError(
            f"{gguf.name} declares no experts, so streaming them would save nothing; "
            "deploy it with llama.cpp instead"
        )
    port = free_port()
    data_dir = ctx.paths.home / "bmoe" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    argv = [str(binary), "--host", "127.0.0.1", "--port", str(port), "--data-dir", str(data_dir),
            "-m", str(gguf), "--moe-stream", "--overlap"]
    if spec.options.get("ctx"):
        argv += ["--ctx-size", str(int(spec.options["ctx"]))]
    if spec.options.get("cache_mb"):
        argv += ["--cache-mb", str(int(spec.options["cache_mb"]))]
    env = ctx.env()
    libs = str(binary.parent)
    env["LD_LIBRARY_PATH"] = f"{libs}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else libs
    ctx.log(f"Starting {SERVER} {BMOE_RELEASE}: {gguf.name}, {expert_count(meta)} experts streamed from disk")
    proc = ManagedProcess("bmoe-server", argv, env=env, cwd=binary.parent, log_path=ctx.log_path("bmoe"))
    ctx.processes.append(proc)
    await proc.start()
    ready = await wait_for(lambda: _models(port), timeout=HEALTH_TIMEOUT, interval=1.0,
                           stop=lambda: proc.proc is None or proc.proc.returncode is not None)
    if not ready:
        tail = "\n".join(list(proc.tail)[-8:])
        raise DeployError(f"bmoe-server did not load {gguf.name} in {int(HEALTH_TIMEOUT)}s\n{tail}")
    model_id = str(ready[0].get("id") or gguf.stem)
    info = {"runtime": "bigmoe", "release": BMOE_RELEASE, "port": port, "gguf": str(gguf),
            "architecture": meta.get("general.architecture"), "experts": expert_count(meta),
            "context": spec.options.get("ctx"), "engine_params": argv[argv.index("--moe-stream"):]}
    return OpenAIUpstream(f"http://127.0.0.1:{port}", model_id, native=frozenset({"anthropic"})), info
