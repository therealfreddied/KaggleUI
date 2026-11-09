"""Async subprocess helpers: supervised long-running processes and one-shot commands with live output."""

from __future__ import annotations

import asyncio
import os
import signal
from collections import deque
from collections.abc import Callable
from pathlib import Path

LineHandler = Callable[[str], None]


class ManagedProcess:
    """A long-running child process in its own process group, with line-by-line log capture."""

    def __init__(self, name: str, cmd: list[str], *, env: dict[str, str] | None = None,
                 cwd: str | Path | None = None, log_path: Path | None = None,
                 on_line: LineHandler | None = None):
        self.name = name
        self.cmd = cmd
        self.env = env
        self.cwd = str(cwd) if cwd else None
        self.log_path = log_path
        self.on_line = on_line
        self.proc: asyncio.subprocess.Process | None = None
        self.tail: deque[str] = deque(maxlen=400)
        self._pumps: list[asyncio.Task] = []

    async def start(self) -> None:
        env = {**os.environ, **(self.env or {})}
        starting = asyncio.create_task(asyncio.create_subprocess_exec(
            *self.cmd, env=env, cwd=self.cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        ))
        try:
            self.proc = await asyncio.shield(starting)
        except asyncio.CancelledError:
            # Cancellation can arrive while the OS is creating the child. Capture
            # it before returning so it cannot escape deployment cleanup.
            self.proc = await starting
            await self.stop(timeout=2)
            raise
        self._pumps = [asyncio.create_task(self._pump())]

    async def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        logf = open(self.log_path, "a", buffering=1) if self.log_path else None
        # Chunked reads, never readline(): a single line over the 64 KiB stream limit makes
        # readline() raise LimitOverrunError, which would kill this pump and block the child.
        pending = b""
        try:
            while True:
                chunk = await self.proc.stdout.read(1 << 16)
                if not chunk:
                    break
                pending += chunk
                *ready, pending = pending.split(b"\n")
                for raw in ready:
                    self._emit_line(raw, logf)
            if pending:
                self._emit_line(pending, logf)
        finally:
            if logf:
                logf.close()

    def _emit_line(self, raw: bytes, logf) -> None:
        line = raw.decode(errors="replace").rstrip("\r\n")
        # tqdm-style progress bars use \r; keep only the latest segment.
        if "\r" in line:
            line = line.split("\r")[-1]
        self.tail.append(line)
        if logf:
            logf.write(line + "\n")
        if self.on_line:
            try:
                self.on_line(line)
            except Exception:
                pass

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    @property
    def returncode(self) -> int | None:
        return self.proc.returncode if self.proc else None

    async def wait(self) -> int:
        assert self.proc
        rc = await self.proc.wait()
        for t in self._pumps:
            try:
                await asyncio.wait_for(t, timeout=2)
            except Exception:
                pass
        return rc

    async def stop(self, timeout: float = 10) -> None:
        if not self.proc:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            # A group leader may exit before its worker children. They still own
            # the same group even though proc.returncode is already set.
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await self.proc.wait()
        for task in self._pumps:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._pumps, return_exceptions=True)


async def run_cmd(cmd: list[str] | str, *, on_line: LineHandler | None = None,
                  env: dict[str, str] | None = None, cwd: str | Path | None = None,
                  timeout: float | None = None) -> int:
    """Run a command to completion, streaming output lines. ``cmd`` as str runs through bash."""
    args = ["bash", "-lc", cmd] if isinstance(cmd, str) else cmd
    proc = ManagedProcess("command", args, env=env, cwd=cwd, on_line=on_line)
    try:
        await proc.start()
        return await asyncio.wait_for(proc.wait(), timeout=timeout)
    except BaseException:
        # Includes task cancellation during install, clone and model creation.
        await proc.stop(timeout=2)
        raise
