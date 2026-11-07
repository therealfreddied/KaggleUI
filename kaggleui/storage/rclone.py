"""rclone destinations (Drive, S3, R2, B2, SFTP...).

Only ``copyto`` is ever used: it copies one file to one path and never deletes or prunes on the remote,
so a typo cannot wipe a bucket. ``rclone sync`` is deliberately absent.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from ..proc import run_cmd
from .base import RemoteEntry, StorageError, check_remote, stage_copy, transfer_scratch

_STATS = re.compile(r"(\d+(?:\.\d+)?)\s*([KMGTP]?i?B)\s*/\s*(\d+(?:\.\d+)?)\s*([KMGTP]?i?B)", re.IGNORECASE)
_UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12,
          "KIB": 1024, "MIB": 1024 ** 2, "GIB": 1024 ** 3, "TIB": 1024 ** 4}


def _bytes(value: str, unit: str) -> int:
    return int(float(value) * _UNITS.get(unit.upper(), 1))


def _mtime(value) -> float | None:
    """rclone reports RFC 3339 timestamps; keep the wire type numeric."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class RcloneProvider:
    name = "rclone"

    def __init__(self, remote: str, *, config_file: str | None = None, binary: str | None = None, cache: Path | None = None):
        if not remote or ":" not in remote:
            raise StorageError("rclone remote must look like 'gdrive:kaggleui'")
        self.remote = remote.rstrip("/")
        self.config_file = config_file or None
        self.binary = binary or "rclone"
        self.cache = Path(cache or "/tmp")

    def _argv(self, *args: str) -> list[str]:
        argv = [self.binary, *args]
        if self.config_file:
            argv += ["--config", self.config_file]
        return argv

    def target(self, remote: str) -> str:
        return f"{self.remote}/{check_remote(remote)}"

    async def _run(self, *args: str, on_line=None) -> int:
        try:
            return await run_cmd(self._argv(*args), on_line=on_line)
        except FileNotFoundError as exc:
            raise StorageError("rclone is not installed in this image. Install a static rclone build "
                               "(https://rclone.org/install/) or use a different connection.") from exc

    async def test(self) -> dict:
        lines: list[str] = []
        code = await self._run("lsjson", self.remote, on_line=lines.append)
        if code != 0:
            raise StorageError("rclone could not list " + self.remote + ": " + " ".join(lines[-3:]))
        try:
            entries = json.loads("".join(lines) or "[]")
        except ValueError:
            entries = []
        return {"ok": True, "remote": self.remote, "entries": len(entries)}

    async def list(self, prefix: str) -> list[RemoteEntry]:
        target = self.target(prefix) if prefix else self.remote
        lines: list[str] = []
        args = ["lsjson", target] + ([] if prefix else ["--files-only"])
        code = await self._run(*args, on_line=lines.append)
        if code != 0:
            raise StorageError(f"rclone could not list {target}: {' '.join(lines[-3:])}")
        try:
            rows = json.loads("".join(lines) or "[]")
        except ValueError as exc:
            raise StorageError(f"unexpected rclone output: {lines[-2:] if lines else ''}") from exc
        return [RemoteEntry(path=f"{prefix.rstrip('/')}/{r.get('Path', r.get('Name', ''))}".lstrip("/"),
                            size=r.get("Size"), mtime=_mtime(r.get("ModTime")),
                            is_dir=bool(r.get("IsDir"))) for r in rows]

    async def upload(self, local: Path, remote: str, *, progress=None) -> None:
        dest = self.target(remote)
        buf: list[str] = []

        def saw(line: str) -> None:
            if progress and (m := _STATS.search(line)):
                total = _bytes(m.group(3), m.group(4))
                if total:
                    progress(min(1.0, _bytes(m.group(1), m.group(2)) / total), line.strip())
            buf.append(line)

        code = await self._run("copyto", "--checksum", "--stats", "1s", "--stats-one-line",
                               str(local), dest, on_line=saw)
        if code != 0:
            raise StorageError(f"rclone copyto failed: {' '.join(buf[-3:])}")

    async def download(self, remote: str, local: Path, *, progress=None) -> None:
        src = self.target(remote)
        scratch = transfer_scratch(self.cache, Path(remote).name)
        staged = scratch / Path(remote).name
        buf: list[str] = []

        def saw(line: str) -> None:
            if progress and (m := _STATS.search(line)):
                total = _bytes(m.group(3), m.group(4))
                if total:
                    progress(min(1.0, _bytes(m.group(1), m.group(2)) / total), line.strip())
            buf.append(line)

        code = await self._run("copyto", "--checksum", "--stats", "1s", "--stats-one-line",
                               src, str(staged), on_line=saw)
        if code != 0:
            raise StorageError(f"rclone copyto failed: {' '.join(buf[-3:])}")
        stage_copy(staged, local)

    async def stat(self, remote: str) -> RemoteEntry | None:
        lines: list[str] = []
        code = await self._run("lsjson", "--stat", self.target(remote), on_line=lines.append)
        if code != 0:
            return None
        try:
            row = json.loads("".join(lines))
        except ValueError:
            return None
        return RemoteEntry(path=remote, size=row.get("Size"), mtime=_mtime(row.get("ModTime")),
                           is_dir=bool(row.get("IsDir")))

    async def delete(self, remote: str) -> None:
        """Only ever deletes the exact path named; directories need an explicit empty prefix."""
        code = await self._run("deletefile", self.target(remote))
        if code != 0:
            raise StorageError(f"rclone deletefile failed for {remote}")
