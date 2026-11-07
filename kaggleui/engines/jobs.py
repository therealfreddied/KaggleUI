"""Run one deployment stage as a persisted, cancellable job (install, download, convert…).

The child reports progress with ``KUI_EVENT`` lines; those become both the job record and the
deployment's own stage progress, so the panel shows one live thing, not two.
"""

from __future__ import annotations

from typing import Any

from ..proc import ManagedProcess
from ..workbench.jobs import TERMINAL, JobRecord
from ..workbench.runner import JobRunner, parse_event
from .context import DeployError, EngineContext


def _bind(ctx: EngineContext, stage: str) -> Any:
    def on_line(line: str) -> None:
        event = parse_event(line)
        if not event:
            return
        if isinstance(event.get("progress"), (int, float)):
            ctx.progress(stage, float(event["progress"]), str(event.get("message") or ""))
        elif event.get("message"):
            ctx.progress(stage, None, str(event["message"]))

    return on_line


async def run_stage(ctx: EngineContext, *, stage: str, kind: str, spec: dict[str, Any], cmd: list[str],
                    env: dict[str, str] | None = None, log_name: str = "step") -> JobRecord | None:
    """Run ``cmd`` to completion. Raises :class:`DeployError` unless it exits 0."""
    log_path = ctx.log_path(log_name)
    ctx.log("$ " + " ".join(cmd))
    ctx.progress(stage, 0.0, f"{kind} started")
    if ctx.jobs is None:
        proc = ManagedProcess(f"{kind}-{ctx.dep_id}", cmd, env=env or ctx.env(), log_path=log_path,
                              on_line=_bind(ctx, stage))
        ctx.processes.append(proc)
        await proc.start()
        rc = await proc.wait()
        if rc != 0:
            raise DeployError(f"{kind} failed (exit {rc}); see the deployment log")
        return None

    job = ctx.jobs.create(kind, {**spec, "deployment": ctx.dep_id}, logs_path=log_path)
    runner = JobRunner(ctx.jobs, job, cmd, env=env or ctx.env(), log_path=log_path, name=f"{kind}-{ctx.dep_id}",
                       on_event=_bind(ctx, stage))
    record = await runner.run()
    if record.state == "cancelled":
        raise DeployError(f"{kind} cancelled")
    if record.state in TERMINAL and record.state != "succeeded":
        raise DeployError(f"{kind} failed: {record.error or 'unknown error'}")
    return record
