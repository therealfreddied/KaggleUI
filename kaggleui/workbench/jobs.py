"""Persisted long-running jobs: sqlite records, log files, cancel and crash recovery.

A job is the only state shared with its child process. Children report progress on stdout as
``KUI_EVENT {"progress": 0.4, ...}`` lines, which :class:`~kaggleui.workbench.runner.JobRunner` parses.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..util import short_id

if TYPE_CHECKING:
    from .runner import JobRunner

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]

TERMINAL: frozenset[str] = frozenset({"succeeded", "failed", "cancelled", "interrupted"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    state TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    progress REAL,
    message TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    ended_at REAL,
    pid INTEGER,
    worker TEXT,
    artifacts_json TEXT,
    checkpoint TEXT,
    resume_supported INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    logs_path TEXT
)
"""


class JobRecord(BaseModel):
    id: str
    kind: str
    state: JobState = "queued"
    spec: dict[str, Any] = Field(default_factory=dict)
    progress: float | None = None
    message: str = ""
    error: str | None = None
    created_at: float = 0.0
    started_at: float | None = None
    ended_at: float | None = None
    pid: int | None = None
    worker: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    checkpoint: str | None = None
    resume_supported: bool = False
    cancel_requested: bool = False
    logs_path: str | None = None

    def public(self) -> dict:
        return self.model_dump()

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"], kind=row["kind"], state=row["state"],
        spec=json.loads(row["spec_json"] or "{}"), progress=row["progress"], message=row["message"] or "",
        error=row["error"], created_at=row["created_at"], started_at=row["started_at"],
        ended_at=row["ended_at"], pid=row["pid"], worker=row["worker"],
        artifacts=json.loads(row["artifacts_json"] or "[]"), checkpoint=row["checkpoint"],
        resume_supported=bool(row["resume_supported"]), cancel_requested=bool(row["cancel_requested"]),
        logs_path=row["logs_path"],
    )


class JobStore:
    """sqlite-backed job table plus the live runners that own the child processes."""

    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(SCHEMA)
            self._conn.commit()
        self._runners: dict[str, JobRunner] = {}
        # In-process jobs (start_local) have no child to stop; their task is what cancel interrupts.
        self._locals: dict[str, asyncio.Task] = {}

    # -- records -------------------------------------------------------------------------------
    def create(self, kind: str, spec: dict, *, resume_supported: bool = False, logs_path: str | None = None,
               worker: str | None = None) -> JobRecord:
        rec = JobRecord(id=short_id(), kind=kind, spec=spec, resume_supported=resume_supported,
                        logs_path=logs_path, worker=worker, created_at=time.time())
        self._insert(rec)
        return rec

    def _insert(self, rec: JobRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, state, spec_json, progress, message, error, created_at, started_at,"
                " ended_at, pid, worker, artifacts_json, checkpoint, resume_supported, cancel_requested, logs_path)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec.id, rec.kind, rec.state, json.dumps(rec.spec), rec.progress, rec.message, rec.error,
                 rec.created_at, rec.started_at, rec.ended_at, rec.pid, rec.worker,
                 json.dumps(rec.artifacts), rec.checkpoint, int(rec.resume_supported), int(rec.cancel_requested),
                 rec.logs_path))
            self._conn.commit()

    def update(self, job_id: str, **fields: Any) -> JobRecord:
        # Read and write under one lock: update() is read-modify-write over the whole row, so a
        # concurrent update (progress vs cancel) would otherwise restore the fields it read stale.
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            data = _row_to_record(row).model_dump()
            data.update(fields)
            rec = JobRecord.model_validate(data)
            self._conn.execute(
                "UPDATE jobs SET state=?, progress=?, message=?, error=?, started_at=?, ended_at=?, pid=?, worker=?,"
                " artifacts_json=?, checkpoint=?, cancel_requested=?, logs_path=?, spec_json=? WHERE id=?",
                (rec.state, rec.progress, rec.message, rec.error, rec.started_at, rec.ended_at, rec.pid,
                 rec.worker, json.dumps(rec.artifacts), rec.checkpoint, int(rec.cancel_requested), rec.logs_path,
                 json.dumps(rec.spec), rec.id))
            self._conn.commit()
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_record(row) if row else None

    def list(self, limit: int = 50, kind: str | None = None) -> list[JobRecord]:
        sql = "SELECT * FROM jobs"
        args: tuple = ()
        if kind:
            sql += " WHERE kind=?"
            args = (kind,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (*args, max(1, min(limit, 500)))).fetchall()
        return [_row_to_record(r) for r in rows]

    # -- lifecycle -----------------------------------------------------------------------------
    def request_cancel(self, job_id: str) -> JobRecord:
        rec = self.update(job_id, cancel_requested=True, message="Cancel requested")
        if rec.state == "queued":
            rec = self.update(job_id, state="cancelled", ended_at=time.time(), message="Cancelled before start")
        return rec

    async def cancel(self, job_id: str) -> JobRecord:
        rec = self.request_cancel(job_id)
        runner = self._runners.get(job_id)
        if runner is not None:
            await runner.stop()
        task = self._locals.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self.get(job_id) or rec

    def attach(self, runner: JobRunner) -> None:
        self._runners[runner.job.id] = runner

    def detach(self, job_id: str) -> None:
        self._runners.pop(job_id, None)

    def attach_local(self, job_id: str, task: asyncio.Task) -> None:
        self._locals[job_id] = task

    def detach_local(self, job_id: str) -> None:
        self._locals.pop(job_id, None)

    def runner(self, job_id: str) -> JobRunner | None:
        return self._runners.get(job_id)

    def recover_interrupted(self) -> int:
        """Mark jobs that were running when the process died; their child is gone with the session."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET state='interrupted', ended_at=?, cancel_requested=0,"
                " error=COALESCE(error, 'Server restarted while this job was running')"
                " WHERE state IN ('queued','running')", (time.time(),))
            self._conn.commit()
        return cur.rowcount or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def read_log(path: str | Path | None, offset: int = 0, limit: int = 400) -> list[str]:
    """Return log lines starting at ``offset`` (line index), at most ``limit`` of them.

    Streams the file and stops after the requested window; never loads the whole log into memory.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    start = max(0, offset)
    stop = start + max(1, limit)
    out: list[str] = []
    try:
        with p.open("r", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index >= stop:
                    break
                if index >= start:
                    out.append(line.rstrip("\r\n"))
    except OSError:
        return []
    return out


def tail_log(path: str | Path | None, count: int = 5) -> list[str]:
    """The last ``count`` lines of a log, read with constant memory."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    tail: deque[str] = deque(maxlen=max(1, count))
    try:
        with p.open("r", errors="replace") as handle:
            for line in handle:
                tail.append(line.rstrip("\n"))
    except OSError:
        return []
    return list(tail)
