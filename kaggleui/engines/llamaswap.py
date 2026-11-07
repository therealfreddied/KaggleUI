"""llama-swap lifecycle: one shared process serving many GGUF deployments on demand.

llama-swap owns the GPU arbitrage (its ``ttl`` unloads idle models), so this module only keeps the
generated config in sync with the durable deployment records and points the gateway at the proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .. import llamacpp as runtime
from ..config import Paths
from ..proc import ManagedProcess
from ..schemas import DeploymentSpec
from ..server.state import AppState
from ..util import free_port, wait_for
from ..workbench import placement as place
from .base import OpenAIUpstream
from .context import DeployError, EngineContext
from .llamacpp_launch import server_argv, swap_model_entry, swap_yaml

HEALTH_TIMEOUT = 120.0


@dataclass
class SwapInstance:
    root: Path
    port: int
    config_path: Path
    proc: ManagedProcess | None = None
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


_INSTANCES: dict[str, SwapInstance] = {}


def instance_for(paths: Paths) -> SwapInstance:
    key = str(paths.home)
    found = _INSTANCES.get(key)
    if found is None:
        workbench = paths.home / "workbench"
        found = SwapInstance(root=workbench, port=free_port(), config_path=workbench / "llama-swap.yaml")
        _INSTANCES[key] = found
    return found


def _env_for(placed: place.ResolvedPlacement) -> list[str]:
    env = ["PYTHONUNBUFFERED=1"]
    if placed.gpus:
        env.append("CUDA_VISIBLE_DEVICES=" + ",".join(str(g) for g in placed.gpus))
    return env


def _write_config(inst: SwapInstance) -> None:
    inst.config_path.parent.mkdir(parents=True, exist_ok=True)
    inst.config_path.write_text(swap_yaml(inst.models))


async def _models_online(port: int) -> list[str] | None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2, read=5, write=5, pool=2)) as client:
        try:
            r = await client.get(f"http://127.0.0.1:{port}/v1/models")
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        try:
            return [str(m.get("id")) for m in r.json().get("data", [])]
        except (ValueError, AttributeError):
            return None


async def _ensure_running(inst: SwapInstance, ctx: EngineContext) -> None:
    binary = runtime.swap_bin(ctx.paths)
    if binary is None:
        from .llamacpp import _ensure_swap

        binary = await _ensure_swap(ctx)
    _write_config(inst)
    if inst.proc is not None and inst.proc.running:
        return
    # The proxy is shared by every llama-swap deployment: it is stopped by llamaswap.shutdown(), never
    # by an individual deployment's stop.
    inst.proc = ManagedProcess(f"llama-swap-{inst.port}", [str(binary), "-config", str(inst.config_path),
                                                         "-listen", f"127.0.0.1:{inst.port}"],
                               env=ctx.env(), log_path=ctx.log_path("llama-swap"), on_line=ctx.log)
    await inst.proc.start()
    if not await wait_for(lambda: _models_online(inst.port) is not None, timeout=HEALTH_TIMEOUT, interval=0.5):
        raise DeployError("llama-swap did not answer /v1/models in time")


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[OpenAIUpstream, dict]:
    from .llamacpp import _download_gguf, _metadata, ensure_runtime, local_gguf

    await ensure_runtime(ctx)
    if spec.model.source in ("attached", "artifact"):
        gguf = local_gguf(spec, ctx.paths)
    else:
        gguf = await _download_gguf(ctx, spec)
    meta = _metadata(gguf)
    errors = place.validate(spec, ctx.hw, arch=meta)
    if errors:
        raise DeployError("; ".join(errors))
    placed = place.resolve(spec, ctx.hw, arch=meta, assigned=ctx.gpus)
    argv = server_argv(bin_path=runtime.require_bin(ctx.paths, "llama-server"), model_path=gguf,
                       host="127.0.0.1", port="${PORT}", served_name=ctx.served_name, spec=spec, placed=placed)
    ttl = 0 if spec.residency == "pinned" else max(0, spec.idle_timeout_s)
    inst = instance_for(ctx.paths)
    inst.models[ctx.served_name] = swap_model_entry(served_name=ctx.served_name, cmd=argv, ttl=ttl,
                                                    env=_env_for(placed))
    if ctx.workbench is not None:
        record = ctx.workbench.deployment_record(ctx.dep_id)
        ctx.workbench.put_deployment(ctx.dep_id, spec, ctx.served_name,
                                     created_at=record.created_at if record else 0.0,
                                     runtime={"gguf": str(gguf), "argv": argv, "served_name": ctx.served_name,
                                              "gpus": list(ctx.gpus)})
    ctx.progress("loading", None, "Registering with llama-swap")
    try:
        await _ensure_running(inst, ctx)
    except DeployError:
        inst.models.pop(ctx.served_name, None)
        _write_config(inst)
        raise
    names = await wait_for(lambda: _names_include(inst.port, ctx.served_name), timeout=30.0, interval=0.5)
    if not names:
        raise DeployError("llama-swap did not list the new model")
    info = {"runtime": "llama-swap", "swap_port": inst.port, "gguf": str(gguf), "ttl": ttl,
            "placement": placed.public(), "loads_on_demand": True}
    return OpenAIUpstream(inst.base_url, ctx.served_name), info


async def _names_include(port: int, name: str) -> bool:
    names = await _models_online(port)
    return names is not None and name in names


def forget(paths: Paths, served_name: str) -> None:
    """Drop a model from the generated config (the process exits when nothing is left)."""
    inst = _INSTANCES.get(str(paths.home))
    if inst is None:
        return
    inst.models.pop(served_name, None)
    _write_config(inst)


async def remove(state: AppState, dep_id: str) -> None:
    record = state.workbench.deployment_record(dep_id)
    if record is None:
        return
    inst = _INSTANCES.get(str(state.paths.home))
    if inst is not None:
        inst.models.pop(record.served_name, None)
        _write_config(inst)
        if not inst.models and inst.proc is not None:
            await inst.proc.stop()
            inst.proc = None


async def shutdown() -> None:
    for inst in list(_INSTANCES.values()):
        if inst.proc is not None:
            await inst.proc.stop()
            inst.proc = None
    _INSTANCES.clear()


def sync_from_records(state: AppState) -> None:
    """Rebuild the config from durable records (used after a restart before the first request)."""
    inst = instance_for(state.paths)
    inst.models = {}
    for record in state.workbench.deployment_records():
        if record.spec.get("lifecycle") != "llama_swap":
            continue
        argv = (record.runtime or {}).get("argv")
        if not argv:
            continue
        try:
            spec = DeploymentSpec.model_validate(record.spec)
        except ValueError:
            continue
        ttl = 0 if spec.residency == "pinned" else max(0, spec.idle_timeout_s)
        placed = place.resolve(spec, state.probe.snapshot(), arch=None, assigned=record.runtime.get("gpus") or None)
        inst.models[record.served_name] = swap_model_entry(served_name=record.served_name, cmd=list(argv), ttl=ttl,
                                                           env=_env_for(placed))
    _write_config(inst)


async def start_if_needed(state: AppState) -> None:
    """Restart llama-swap after a server restart when records describe live models."""
    sync_from_records(state)
    inst = instance_for(state.paths)
    if not inst.models or (inst.proc is not None and inst.proc.running):
        return
    ctx = EngineContext(paths=state.paths, dep_id="swap", served_name="llama-swap", gpus=[],
                        hw=state.probe.snapshot(), log=lambda _line: None,
                        progress=lambda *_a: None, internal_key=state.auth.internal_key,
                        main_port=state.port, jobs=state.jobs, workbench=state.workbench)
    try:
        await _ensure_running(inst, ctx)
    except DeployError:
        pass
