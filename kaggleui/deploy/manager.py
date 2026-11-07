"""Deployments: lifecycle (installing → downloading → loading → ready), GPU placement, logs, interfaces."""

from __future__ import annotations

import asyncio
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from ..engines import registry
from ..engines.base import Backend
from ..engines.context import DeployError, EngineContext
from ..proc import ManagedProcess
from ..schemas import DeploymentSpec
from ..util import short_id

if TYPE_CHECKING:
    from ..server.state import AppState

ACTIVE = {"queued", "installing", "downloading", "loading", "ready"}


@dataclass
class Deployment:
    id: str
    spec: DeploymentSpec
    served_name: str
    status: str = "queued"
    progress: float | None = None
    message: str = "Queued"
    error: str | None = None
    gpus: list[int] = field(default_factory=list)
    vram_estimate: int = 0
    vram_plan: int = 0
    last_used: float = 0.0
    _start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: float = field(default_factory=time.time)
    ready_at: float | None = None
    info: dict[str, Any] = field(default_factory=dict)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=1500))
    backend: Backend | None = None
    processes: list[ManagedProcess] = field(default_factory=list)
    task: asyncio.Task | None = None
    interface: dict[str, Any] = field(default_factory=dict)
    _last_emit: float = 0.0
    _stop_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def public(self) -> dict:
        return {
            "id": self.id, "spec": self.spec.model_dump(), "name": self.served_name, "modality": self.spec.modality,
            "status": self.status, "progress": self.progress, "message": self.message, "error": self.error,
            "gpus": self.gpus, "vram_estimate": self.vram_estimate, "created_at": self.created_at,
            "ready_at": self.ready_at, "info": self.info, "interface": self.interface,
        }


def default_name(spec: DeploymentSpec) -> str:
    m = spec.model
    if spec.name:
        return spec.name.strip()
    if m.source == "ollama":
        base = m.id if ":" in m.id else (f"{m.id}:{m.variant}" if m.variant else m.id)
        return base
    if m.source == "attached":
        base = PurePath(m.id).name
        base = re.sub(r"\.gguf$", "", base, flags=re.I)
    else:
        base = m.id.split("/")[-1]
        base = re.sub(r"[-_]gguf$", "", base, flags=re.I)
        if m.engine == "ollama" and m.variant:
            base = f"{base}:{m.variant.lower()}"
    return base.lower()


class DeploymentManager:
    def __init__(self, state: AppState):
        self.state = state
        self.deployments: dict[str, Deployment] = {}

    # -- queries ------------------------------------------------------------------------------
    def list(self) -> list[dict]:
        return [d.public() for d in self.deployments.values()]

    def get(self, dep_id: str) -> Deployment | None:
        return self.deployments.get(dep_id)

    def ready(self, modality: str | None = None) -> list[Deployment]:
        return [d for d in self.deployments.values()
                if d.status == "ready" and d.backend is not None and (modality is None or d.spec.modality == modality)]

    def resolve(self, model: str | None, modality: str | None = None) -> Deployment | None:
        """Find the deployment a request's ``model`` field refers to.

        Exact name match first; otherwise, if only one model of that modality is running, use it
        (so clients with a hard-coded model name like ``gpt-4o`` still work).
        """
        candidates = self.ready(modality)
        if model:
            for d in candidates:
                if model in (d.served_name, d.id, d.spec.model.id):
                    return d
            for d in self.deployments.values():
                if d.status == "cold" and (modality is None or d.spec.modality == modality) and \
                        model in (d.served_name, d.id, d.spec.model.id):
                    return d
        if len(candidates) == 1:
            return candidates[0]
        if candidates and not model:
            return candidates[0]
        return None

    # -- lifecycle ----------------------------------------------------------------------------
    def create(self, spec: DeploymentSpec) -> Deployment:
        if self.state.updates.blocked:
            raise DeployError(self.state.updates.status()["message"])
        name = default_name(spec)
        taken = {d.served_name for d in self.deployments.values() if d.status in ACTIVE}
        base, i = name, 2
        while name in taken:
            name = f"{base}-{i}"
            i += 1
        dep = Deployment(id=short_id(6), spec=spec, served_name=name)
        dep.vram_estimate = int(spec.options.get("vram_estimate") or 0)
        dep.vram_plan = dep.vram_estimate
        self.deployments[dep.id] = dep
        self.state.workbench.put_deployment(dep.id, spec, dep.served_name, created_at=dep.created_at)
        if spec.residency == "on_demand":
            dep.status = "cold"
            dep.message = "Loads on first request"
            self._emit(dep, force=True)
            return dep
        self._start(dep)
        return dep

    def _start(self, dep: Deployment) -> None:
        dep.gpus = self._choose_gpus(dep.spec, dep.vram_plan)
        free = sum(g.vram_free for g in self.state.probe.snapshot().gpus if g.index in dep.gpus)
        if dep.vram_plan and dep.gpus and dep.vram_plan > free:
            dep.logs.append(f"WARNING: needs ≈{dep.vram_plan / 1e9:.1f} GB VRAM but only {free / 1e9:.1f} GB is free "
                            "on its GPUs. Stop another deployment if loading fails or runs slowly.")
        dep.vram_estimate = dep.vram_plan
        self._reserve(dep)
        dep.status = "queued"
        dep.error = None
        dep.progress = None
        dep.message = "Queued"
        dep.task = asyncio.create_task(self._run(dep))
        self._emit(dep, force=True)

    async def ensure_ready(self, dep: Deployment, timeout: float = 900.0) -> None:
        """Start an on-demand deployment if needed and wait until it can serve (single flight)."""
        if dep.status == "ready" and dep.backend is not None:
            return
        async with dep._start_lock:
            if dep.status == "ready" and dep.backend is not None:
                return
            if dep.status not in ("cold", "stopped", "error"):
                raise DeployError(f"{dep.served_name} is {dep.status}: {dep.message}")
            if self.state.updates.blocked:
                raise DeployError(self.state.updates.status()["message"])
            await self.state.residency.make_room(dep)
            self._start(dep)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if dep.status == "ready" and dep.backend is not None:
                return
            if dep.status in ("error", "stopped"):
                raise DeployError(f"{dep.served_name} failed to start: {dep.message}")
            await asyncio.sleep(0.25)
        raise DeployError(f"{dep.served_name} did not become ready in {int(timeout)}s")

    def touch(self, dep: Deployment) -> None:
        dep.last_used = time.time()

    def _choose_gpus(self, spec: DeploymentSpec, need: int) -> list[int]:
        hw = self.state.probe.snapshot()
        if spec.gpus is not None:
            return [g for g in spec.gpus if any(x.index == g for x in hw.gpus)]
        if not hw.gpus:
            return []
        engine = spec.model.engine
        if spec.lifecycle == "llama_swap":
            # The shared proxy owns per-model GPU pinning, so it must see every card.
            return [g.index for g in hw.gpus]
        if spec.placement is not None and spec.placement.devices:
            return [g for g in spec.placement.devices if any(x.index == g for x in hw.gpus)]
        if engine in ("ollama", "llamacpp") or spec.interface == "comfyui":
            return [g.index for g in hw.gpus]
        pending: dict[int, int] = {}
        if not hw.fake:  # fake probes already count reservations as used VRAM
            for d in self.deployments.values():
                if d.status in ("queued", "installing", "downloading", "loading"):
                    for g in d.gpus:
                        pending[g] = pending.get(g, 0) + d.vram_estimate // max(len(d.gpus), 1)
        free = sorted(((g.vram_free - pending.get(g.index, 0), g.index) for g in hw.gpus), reverse=True)
        if engine in ("vllm", "transformers") and need and need > free[0][0] * 0.97 and len(free) > 1:
            return sorted(i for _, i in free)
        return [free[0][1]]

    def _reserve(self, dep: Deployment) -> None:
        if not dep.gpus or not dep.vram_estimate:
            return
        share = dep.vram_estimate // len(dep.gpus)
        for g in dep.gpus:
            self.state.probe.reserved[g] = self.state.probe.reserved.get(g, 0) + share

    def _release(self, dep: Deployment) -> None:
        if not dep.gpus or not dep.vram_estimate:
            return
        share = dep.vram_estimate // len(dep.gpus)
        for g in dep.gpus:
            self.state.probe.reserved[g] = max(self.state.probe.reserved.get(g, 0) - share, 0)
        dep.vram_estimate = 0

    def _emit(self, dep: Deployment, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - dep._last_emit < 0.25:
            return
        dep._last_emit = now
        self.state.bus.publish("deployment", dep.public())

    def _log(self, dep: Deployment, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        dep.logs.append(line)
        self.state.bus.publish("log", {"id": dep.id, "line": line})

    async def _run(self, dep: Deployment) -> None:
        st = self.state

        def progress(stage: str, frac: float | None, message: str) -> None:
            if dep.status in ("stopping", "stopped"):
                return
            dep.status = stage
            dep.progress = frac
            dep.message = message or dep.message
            self._emit(dep, force=frac is None or frac >= 1.0)

        ctx = EngineContext(paths=st.paths, dep_id=dep.id, served_name=dep.served_name, gpus=dep.gpus,
                            hw=st.probe.snapshot(), log=lambda line: self._log(dep, line), progress=progress,
                            internal_key=st.auth.internal_key, main_port=st.port, processes=dep.processes,
                            jobs=st.jobs, workbench=st.workbench)
        try:
            hw = st.probe.snapshot()
            where = f"TPU ({len(hw.tpus)} chips)" if hw.accelerator == "tpu" and hw.tpus else f"GPUs {dep.gpus or 'none'}"
            self._log(dep, f"Deploying {dep.spec.model.id} with {dep.spec.model.engine} on {where}")
            if dep.spec.interface == "comfyui":
                await st.interfaces.start(dep, ctx)
                dep.info["runtime"] = "comfyui"
            else:
                backend, info = await registry.deploy(dep.spec, ctx, mock=st.mock)
                dep.backend = backend
                dep.info.update(info)
            dep.status = "ready"
            dep.progress = 1.0
            dep.message = "Ready"
            dep.ready_at = time.time()
            self._log(dep, f"Ready as '{dep.served_name}'")
            self._emit(dep, force=True)
            if dep.spec.interface in ("gradio", "openwebui"):
                await st.interfaces.start(dep, ctx)
                self._emit(dep, force=True)
        except asyncio.CancelledError:
            raise
        except DeployError as e:
            self._fail(dep, str(e))
        except Exception as e:  # noqa: BLE001
            self._log(dep, traceback.format_exc())
            self._fail(dep, f"{type(e).__name__}: {e}")

    def _fail(self, dep: Deployment, message: str) -> None:
        dep.status = "error"
        dep.error = message
        dep.message = "Failed"
        self._log(dep, f"ERROR: {message}")
        self._release(dep)
        self._emit(dep, force=True)

    async def stop(self, dep_id: str, remove: bool = True) -> None:
        dep = self.deployments.get(dep_id)
        if not dep:
            return
        async with dep._stop_lock:
            if dep.status == "stopped":
                return
            await self._stop(dep, remove)

    async def _stop(self, dep: Deployment, remove: bool) -> None:
        dep_id = dep.id
        dep.status = "stopping"
        dep.message = "Stopping"
        self._emit(dep, force=True)
        if remove:
            self.state.workbench.drop_deployment(dep_id)
        if dep.task and not dep.task.done():
            dep.task.cancel()
            try:
                await dep.task
            except BaseException:
                pass
        try:
            await self.state.interfaces.stop(dep)
        except Exception as exc:
            self._log(dep, f"Interface cleanup: {exc}")
        if dep.backend:
            try:
                await dep.backend.close()
            except Exception:
                pass
            dep.backend = None
        for proc in dep.processes:
            await proc.stop()
        try:
            await registry.undeploy(dep.spec, dep.id, mock=self.state.mock, state=self.state)
        except Exception as exc:
            self._log(dep, f"Model cleanup: {exc}")
        self._release(dep)
        if not remove and dep.spec.residency == "on_demand":
            dep.status = "cold"
            dep.message = "Loads on first request"
        else:
            dep.status = "stopped"
            dep.message = "Stopped"
        self._emit(dep, force=True)
        if remove:
            self.deployments.pop(dep_id, None)
            self.state.bus.publish("deployment_removed", {"id": dep_id})

    def restore(self) -> None:
        """Re-register durable deployments as cold/stopped so their names resolve after a restart."""
        for record in self.state.workbench.deployment_records():
            if record.id in self.deployments:
                continue
            try:
                spec = DeploymentSpec.model_validate(record.spec)
            except ValueError:
                continue
            dep = Deployment(id=record.id, spec=spec, served_name=record.served_name,
                             created_at=record.created_at or time.time())
            dep.vram_plan = dep.vram_estimate = int(spec.options.get("vram_estimate") or 0)
            if spec.residency == "on_demand":
                dep.status, dep.message = "cold", "Loads on first request"
            else:
                dep.status, dep.message = "stopped", "Stopped when the server last shut down"
            self.deployments[dep.id] = dep

    async def restart(self, dep_id: str) -> Deployment | None:
        dep = self.deployments.get(dep_id)
        if not dep:
            return None
        spec = dep.spec
        await self.stop(dep_id)
        return self.create(spec)

    async def shutdown(self) -> None:
        # remove=False keeps the workbench records, so the next server start can rebuild the set.
        results = await asyncio.gather(*(self.stop(dep_id, remove=False) for dep_id in list(self.deployments)),
                                       return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                print(f"[kaggleui] deployment cleanup: {result}", flush=True)
