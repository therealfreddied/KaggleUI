"""Shared storage contract: every destination speaks the same six calls.

Uploads and downloads are always temp-local-first: the provider writes into
``paths.cache/transfers`` and only the caller moves the result into its final home, so an interrupted
transfer never leaves a half-written artifact behind.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from ..schemas import StorageConnection


class RemoteEntry(BaseModel):
    path: str
    size: int | None = None
    mtime: float | None = None
    is_dir: bool = False


class StorageError(Exception):
    """A destination refused or failed the operation; the message is user-facing."""


class StorageProvider(Protocol):
    name: str

    async def test(self) -> dict: ...
    async def list(self, prefix: str) -> list[RemoteEntry]: ...
    async def upload(self, local: Path, remote: str, *, progress=None) -> None: ...
    async def download(self, remote: str, local: Path, *, progress=None) -> None: ...
    async def stat(self, remote: str) -> RemoteEntry | None: ...
    async def delete(self, remote: str) -> None: ...


def check_remote(remote: str) -> str:
    """Reject traversal and absolute surprises before any provider sees the path."""
    value = (remote or "").strip()
    if not value:
        raise StorageError("destination path is empty")
    if value.startswith(("/", "~")) or ".." in Path(value).parts:
        raise StorageError(f"destination must be relative, got {value!r}")
    return value


def transfer_scratch(cache: Path, name: str) -> Path:
    """Per-transfer work directory under the cache root (never inside an artifact directory)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(cache) / "transfers" / f"{stamp}-{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_copy(src: Path, dest: Path) -> Path:
    """Copy into ``dest`` atomically and verify the size; returns ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    shutil.copy2(src, tmp)
    if tmp.stat().st_size != Path(src).stat().st_size:
        tmp.unlink(missing_ok=True)
        raise StorageError(f"size mismatch copying {src.name}")
    os.replace(tmp, dest)
    return dest


def connection_of(name: str, kind: str, config: dict[str, str], secret: str | None = None) -> StorageConnection:
    return StorageConnection(name=name, kind=kind, config=config, secret=secret)
