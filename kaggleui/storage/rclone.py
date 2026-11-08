"""rclone destinations (Drive, S3, R2, B2, SFTP...).

Only ``copyto`` is ever used: it copies one file to one path and never deletes or prunes on the remote,
so a typo cannot wipe a bucket. ``rclone sync`` is deliberately absent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
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


_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_BACKEND = re.compile(r"^[a-z0-9]{1,20}$")
_PARAM = re.compile(r"^[a-z0-9_]{1,40}$")
_SECRET_HINT = re.compile(r"pass|secret|token|key|credential", re.IGNORECASE)

def rclone_installed() -> bool:
    return shutil.which("rclone") is not None

def rclone_env(config_file: Path | None = None) -> dict[str, str]:
    """Environment for one-shot rclone config calls; ``RCLONE_CONFIG`` selects the file we manage."""
    env = dict(os.environ)
    if config_file is not None:
        env["RCLONE_CONFIG"] = str(config_file)
    return env

async def _config_run(args: list[str], env: dict[str, str] | None) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = await run_cmd(["rclone", *args], env=env, on_line=lines.append, timeout=60)
    return code, lines

def _require_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME.match(name):
        raise StorageError("Remote names are 1-40 letters, digits, '-' or '_'")
    return name

async def list_remotes(env: dict[str, str] | None = None) -> list[str]:
    code, lines = await _config_run(["listremotes"], env)
    if code != 0:
        raise StorageError("rclone listremotes failed: " + " ".join(lines[-3:]))
    return sorted(line.strip().rstrip(":") for line in lines if line.strip().endswith(":"))

async def create_remote(env: dict[str, str] | None, name: str, backend: str, params: dict[str, str]) -> str:
    """Create a remote non-interactively. ``rclone config create`` never prompts when values are given."""
    name = _require_name(name)
    if not _BACKEND.match(backend or ""):
        raise StorageError("Unsupported rclone backend")
    args = ["config", "create", name, backend]
    for key, value in params.items():
        if not _PARAM.match(str(key)):
            raise StorageError(f"Invalid rclone option {key!r}")
        text = str(value or "")
        if "\n" in text or "\r" in text:
            raise StorageError(f"{key} must be a single line")
        if text:
            args.append(f"{key}={text}")
    code, lines = await _config_run(args, env)
    if code != 0:
        raise StorageError("rclone config create failed: " + " ".join(lines[-3:]))
    return name

async def delete_remote(env: dict[str, str] | None, name: str) -> None:
    name = _require_name(name)
    code, lines = await _config_run(["config", "delete", name], env)
    if code != 0:
        raise StorageError("rclone config delete failed: " + " ".join(lines[-3:]))

async def reveal_remote(env: dict[str, str] | None, name: str) -> dict[str, str]:
    """Options for one remote with secrets masked, so the panel can show what is configured."""
    name = _require_name(name)
    code, lines = await _config_run(["config", "dump"], env)
    if code != 0:
        raise StorageError("rclone config dump failed: " + " ".join(lines[-3:]))
    try:
        dump = json.loads("".join(lines) or "{}")
    except ValueError as exc:
        raise StorageError("rclone config dump returned unreadable JSON") from exc
    remote = dump.get(name)
    if not isinstance(remote, dict):
        raise StorageError(f"No rclone remote named {name}")
    shown: dict[str, str] = {}
    for key, value in remote.items():
        text = str(value)
        if _SECRET_HINT.search(key) and text:
            shown[key] = "…" + text[-4:] if len(text) > 4 else "…"
        else:
            shown[key] = text
    return shown

class RcloneProvider:

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
