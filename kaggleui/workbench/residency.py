"""Residency: keep several models registered while only the hot ones hold VRAM.

Three policies, per :class:`~kaggleui.schemas.DeploymentSpec`:

* ``pinned`` — never evicted automatically.
* ``on_demand`` — starts on the first request and is unloaded after ``idle_timeout_s``.
* ``manual`` — loads when created, stopped only by the user; admission control may still evict it
  when a new deployment needs the VRAM.

Every eviction goes through ``DeploymentManager.stop``, so process ownership stays in one place and
``pkill``-style shortcuts are never needed.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from ..engines.context import DeployError

if TYPE_CHECKING:
    from ..deploy.manager import Deployment
    from ..server.state import AppState

SWEEP_SECONDS = 15.0


class Residency:
    def __init__(self, state: AppState):
        self.state = state
        self.busy = 0
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None

    @staticmethod
    def touch(dep: Deployment) -> None:
        dep.last_used = time.time()

    def _idle(self, dep: Deployment) -> float:
        return time.time() - max(dep.last_used, dep.ready_at or 0.0)

    def _evictable(self, dep: Deployment, *, for_admission: bool) -> bool:
        if dep.spec.residency == "pinned":
            return False
        if dep.status != "ready":
            return False
        if not for_admission and dep.spec.residency != "on_demand":
            return False
        timeout = max(30, dep.spec.idle_timeout_s)
        return self._idle(dep) >= timeout

    async def make_room(self, dep: Deployment) -> None:
        """Free VRAM for ``dep`` by stopping idle deployments; raise when even that is not enough."""
        hw = self.state.probe.snapshot()
        if not hw.gpus or not dep.vram_plan:
            return
        candidates = [g for g in (dep.spec.placement.devices if dep.spec.placement and dep.spec.placement.devices
                                  else range(len(hw.gpus)))]
        need = dep.vram_plan
        if self._free(candidates) >= need:
            return
        victims = sorted((d for d in self.state.manager.deployments.values() if self._evictable(d, for_admission=True)),
                         key=lambda d: d.last_used or d.created_at)
        for victim in victims:
            if self._free(candidates) >= need:
                return
            await self.state.manager.stop(victim.id, remove=False)
        if self._free(candidates) < need:
            raise DeployError(f"Not enough free VRAM for {dep.served_name} "
                              f"(needs ≈{need / 1e9:.1f} GB). Stop a deployment and try again.")

    def _free(self, gpus) -> int:
        wanted = set(gpus)
        return sum(g.vram_free for g in self.state.probe.snapshot().gpus if g.index in wanted)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(SWEEP_SECONDS)
            if not self.busy:
                await self.sweep()

    async def sweep(self) -> int:
        """Unload one round of idle on-demand models; returns how many were stopped."""
        stopped = 0
        for dep in list(self.state.manager.deployments.values()):
            if not self._evictable(dep, for_admission=False):
                continue
            try:
                await self.state.manager.stop(dep.id, remove=False)
                stopped += 1
            except Exception:  # noqa: BLE001 - one bad stop must not kill the sweep
                continue
        return stopped
