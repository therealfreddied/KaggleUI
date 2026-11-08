"""A main tunnel whose process and log file survive replacement of the app server.

Only the main tunnel is retained. Its PID and creation time are recorded together
so a later launch can adopt it, and stop() can safely find it after its parent exits.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from collections import deque
from collections.abc import Callable

import psutil

from ..config import Paths


def binding(provider: str, port: int) -> str:
    names = {"ngrok": ("NGROK_AUTHTOKEN", "NGROK_DOMAIN"),
             "cloudflare_named": ("CF_TUNNEL_TOKEN", "CF_TUNNEL_HOSTNAME")}.get(provider, ())
    value = [provider, port, *(os.environ.get(name, "") for name in names)]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def read_record(paths: Paths) -> dict:
    try:
        value = json.loads((paths.home / "tunnel-main.json").read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def recorded_process(record: dict) -> psutil.Process | None:
    try:
        pid = record["pid"]
        if not isinstance(pid, int) or pid <= 0:
            return None
        proc = psutil.Process(pid)
        started, current = record["process_started_at"], _create_time(proc)
        if started is not None and current is not None and abs(current - started) >= .01:
            return None
        return proc if proc.status() != psutil.STATUS_ZOMBIE else None
    except (KeyError, TypeError, ValueError, psutil.Error):
        pass
    return None


def saved_processes(paths: Paths) -> list[psutil.Process]:
    proc = recorded_process(read_record(paths))
    if proc is None:
        return []
    try:
        return [proc, *proc.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []
    except psutil.AccessDenied:  # a sandbox that hides other processes still lets us stop the leader
        return [proc]


def forget_record(paths: Paths, record: dict) -> None:
    current = read_record(paths)
    if (current.get("pid"), current.get("process_started_at")) == (record.get("pid"), record.get("process_started_at")):
        (paths.home / "tunnel-main.json").unlink(missing_ok=True)


def _create_time(proc: psutil.Process) -> float | None:
    """Some sandboxes deny /proc/stat; the handoff then falls back to the PID alone."""
    try:
        return proc.create_time()
    except psutil.Error:
        return None

def request_handoff(paths: Paths, server: psutil.Process) -> None:
    """Ask this specific server to retain its tunnel for an update restart."""
    path = paths.home / "tunnel-restart.json"
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"pid": server.pid, "process_started_at": _create_time(server),
                   "expires": time.time() + 120}, stream)
    temp.replace(path)


def take_handoff(paths: Paths) -> bool:
    path = paths.home / "tunnel-restart.json"
    try:
        record = json.loads(path.read_text())
        if record.get("pid") != os.getpid() or record.get("expires", 0) <= time.time():
            return False
        recorded = record.get("process_started_at")
        current = _create_time(psutil.Process())
        return recorded is None or current is None or abs(recorded - current) < .01
    except (OSError, ValueError, TypeError, KeyError):
        return False
    finally:
        path.unlink(missing_ok=True)


def stop_saved_tunnel(paths: Paths, timeout: float = 5) -> bool:
    record = read_record(paths)
    processes = saved_processes(paths)
    for proc in processes:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            pass
    _, alive = psutil.wait_procs(processes, timeout=timeout)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            pass
    psutil.wait_procs(alive, timeout=2)
    remaining = []
    for proc in processes:
        try:
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                remaining.append(proc.pid)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            remaining.append(proc.pid)
    if remaining:
        raise RuntimeError(f"Could not stop the public tunnel processes: {remaining}")
    forget_record(paths, record)
    return bool(processes)


class PersistentProcess:
    """Write directly to a file, never to a pipe owned by the replaceable server."""

    def __init__(self, paths: Paths, cmd: list[str], *, signature: str,
                 on_line: Callable[[str], None], env: dict[str, str] | None = None):
        self.paths = paths
        self.cmd = cmd
        self.signature = signature
        self.env = env
        self.on_line = on_line
        self.log_path = paths.logs / "tunnel-main.log"
        self.tail: deque[str] = deque(maxlen=400)
        self.record: dict = {}
        self.process: subprocess.Popen | None = None
        self.offset = 0

    def adopt(self) -> dict | None:
        record = read_record(self.paths)
        if record.get("binding") != self.signature or recorded_process(record) is None:
            return None
        self.record = record
        # Replay startup if the server exited before recording registration.
        self.offset = (self.log_path.stat().st_size if record.get("status") == "up" and self.log_path.exists()
                       else int(record.get("log_start", 0)))
        return record

    async def start(self) -> None:
        if self.record:
            if self.running:
                return
            raise RuntimeError("The retained public tunnel has exited; reconnecting")
        with self.log_path.open("ab") as log:
            self.offset = log.tell()
            self.process = subprocess.Popen(self.cmd, env={**os.environ, **(self.env or {})},
                                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
        try:
            started = psutil.Process(self.process.pid).create_time()
        except psutil.NoSuchProcess:
            self.process.poll()
            raise RuntimeError("The public tunnel exited during startup") from None
        except psutil.AccessDenied:  # /proc/stat is unreadable here; the PID is all we can record
            started = None
        self.record = {"pid": self.process.pid, "process_started_at": started,
                       "binding": self.signature, "log_start": self.offset}

    def save(self, info: dict) -> None:
        if not self.record:
            return
        self.record.update(info, updated=time.time())
        path = self.paths.home / "tunnel-main.json"
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(self.record, stream)
        temp.replace(path)

    @property
    def running(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        return recorded_process(self.record) is not None

    async def wait(self) -> int:
        with self.log_path.open(errors="replace") as log:
            log.seek(self.offset)
            pending = ""
            while True:
                for _ in range(500):
                    raw = log.readline()
                    if not raw:
                        break
                    pending += raw
                    if not pending.endswith("\n"):
                        continue
                    line = pending.rstrip("\r\n").rsplit("\r", 1)[-1]
                    pending = ""
                    self.tail.append(line)
                    self.on_line(line)
                if not self.running:
                    forget_record(self.paths, self.record)
                    return self.process.returncode if self.process is not None else -1
                await asyncio.sleep(.2)

    async def stop(self, timeout: float = 5) -> None:
        # A stale instance must never terminate a replacement tunnel.
        current = read_record(self.paths)
        if (current.get("pid"), current.get("process_started_at")) == (self.record.get("pid"), self.record.get("process_started_at")):
            await asyncio.to_thread(stop_saved_tunnel, self.paths, timeout)
        if self.process is not None:
            self.process.poll()
