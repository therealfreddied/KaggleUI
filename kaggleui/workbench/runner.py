"""Run a job's child process and fold its ``KUI_EVENT`` progress lines into the job record."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..proc import ManagedProcess
from .jobs import JobRecord, JobStore, read_log

EVENT_TAG = "KUI_EVENT "
EventParser = Callable[[dict[str, Any]], None]


def parse_event(line: str) -> dict[str, Any] | None:
    """Extract the JSON payload of a ``KUI_EVENT {...}`` line, or ``None``."""
    idx = line.find(EVENT_TAG)
    if idx < 0:
        return None
    try:
        payload = json.loads(line[idx + len(EVENT_TAG):].strip())
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


class JobRunner:
    """Owns one child process for one job: start, stream, cancel, final state."""

    def __init__(self, store: JobStore, job: JobRecord, cmd: list[str], *, env: dict[str, str] | None = None,
                 cwd: str | Path | None = None, log_path: str | Path | None = None, name: str | None = None,
                 on_event: EventParser | None = None):
        self.store = store
        self.job = job
        self.cmd = cmd
        self.on_event = on_event
        self._artifacts = list(job.artifacts)
        self.log_path = Path(log_path) if log_path else None
        self.proc = ManagedProcess(name or f"job-{job.id}", cmd, env=env, cwd=cwd, log_path=self.log_path,
                                   on_line=self._on_line)

    def _on_line(self, line: str) -> None:
        event = parse_event(line)
        if event is None:
            return
        fields: dict[str, Any] = {}
        if isinstance(event.get("progress"), (int, float)):
            fields["progress"] = max(0.0, min(1.0, float(event["progress"])))
        for key in ("message", "checkpoint", "error"):
            if event.get(key):
                fields[key] = str(event[key])
        if event.get("artifact"):
            self._artifacts.append(str(event["artifact"]))
            fields["artifacts"] = list(self._artifacts)
        if fields:
            try:
                self.store.update(self.job.id, **fields)
            except KeyError:
                pass
        if self.on_event is not None:
            self.on_event(event)

    async def run(self) -> JobRecord:
        self.store.attach(self)
        self.store.update(self.job.id, state="running", started_at=time.time(), message="Starting")
        error: str | None = None
        rc: int | None = None
        try:
            await self.proc.start()
            pid = self.proc.proc.pid if self.proc.proc else None
            self.store.update(self.job.id, pid=pid)
            rc = await self.proc.wait()
        except Exception as exc:  # missing binary, permission error, ...
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self.store.detach(self.job.id)
        record = self.store.get(self.job.id) or self.job
        if record.cancel_requested:
            return self.store.update(self.job.id, state="cancelled", ended_at=time.time(), message="Cancelled")
        if rc == 0:
            return self.store.update(self.job.id, state="succeeded", ended_at=time.time(),
                                     progress=record.progress if record.progress is not None else 1.0,
                                     message=record.message or "Finished")
        lines = read_log(self.log_path)
        detail = error or "\n".join(lines[-5:]) or f"exit code {rc}"
        return self.store.update(self.job.id, state="failed", ended_at=time.time(),
                                 error=record.error or detail, message="Failed")

    async def stop(self) -> None:
        await self.proc.stop()


def spawn(store: JobStore, job: JobRecord, cmd: list[str], *, env: dict[str, str] | None = None,
          cwd: str | Path | None = None, log_path: str | Path | None = None, name: str | None = None,
          on_event: EventParser | None = None) -> JobRunner:
    """Build a runner. Call ``await runner.run()`` from a task (the caller owns task lifetime)."""
    return JobRunner(store, job, cmd, env=env, cwd=cwd, log_path=log_path, name=name, on_event=on_event)


class JobContext:
    """Progress/log handle for work that runs inside the server (exports, restores, verification)."""

    def __init__(self, store: JobStore, job: JobRecord):
        self.store = store
        self.job_id = job.id
        self.logs_path = job.logs_path

    def log(self, line: str) -> None:
        if not self.logs_path:
            return
        try:
            Path(self.logs_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(self.logs_path).open("a", encoding="utf-8") as f:
                f.write(line.rstrip("\n") + "\n")
        except OSError:
            pass

    def progress(self, fraction: float, message: str = "") -> None:
        self.store.update(self.job_id, progress=max(0.0, min(1.0, float(fraction))), message=message)
        if message:
            self.log(message)

    def artifact(self, manifest: dict) -> None:
        """Record a produced artifact as ``artifact_id@version`` (the record field is a string list)."""
        record = self.store.get(self.job_id)
        if record is None:
            return
        ref = f"{manifest.get('artifact_id')}@{manifest.get('version')}"
        if ref not in record.artifacts:
            self.store.update(self.job_id, artifacts=[*record.artifacts, ref])


async def start_local(store: JobStore, job: JobRecord, work) -> asyncio.Task:
    """Run an in-process coroutine as a job: running -> succeeded/failed, with logs and progress."""
    ctx = JobContext(store, job)
    store.update(job.id, state="running", started_at=time.time(), pid=os.getpid())

    async def _run() -> None:
        try:
            await work(ctx)
        except asyncio.CancelledError:
            store.update(job.id, state="cancelled", ended_at=time.time(), message="Cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - every failure belongs to the job record
            ctx.log(f"error: {exc}")
            store.update(job.id, state="failed", ended_at=time.time(), error=str(exc))
        else:
            store.update(job.id, state="succeeded", ended_at=time.time(), progress=1.0)

    task = asyncio.create_task(_run())
    _background.add(task)
    task.add_done_callback(_background.discard)
    return task

_background: set[asyncio.Task] = set()


def start_in_background(store: JobStore, job: JobRecord, cmd: list[str], **kwargs: Any) -> asyncio.Task:
    runner = spawn(store, job, cmd, **kwargs)
    task = asyncio.create_task(runner.run())
    _background.add(task)
    task.add_done_callback(_background.discard)
    return task
