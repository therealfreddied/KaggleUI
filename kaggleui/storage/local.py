"""A directory on mounted storage (``/kaggle/working/...``, a network mount, an attached volume)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from .base import RemoteEntry, StorageError, check_remote, stage_copy


class LocalProvider:
    name = "local"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def _resolve(self, remote: str) -> Path:
        target = (self.root / check_remote(remote)).resolve()
        if self.root != target and self.root not in target.parents:
            raise StorageError(f"{remote} escapes {self.root}")
        return target

    def _require_root(self) -> None:
        if not self.root.is_dir():
            raise StorageError(f"{self.root} is not a directory; mount it or point the connection elsewhere")

    async def test(self) -> dict:
        self._require_root()
        probe = self.root / ".kaggleui-probe"
        writable = True
        try:
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            writable = False
            detail = str(exc)
        else:
            detail = "writable"
        return {"ok": writable, "root": str(self.root), "detail": detail,
                "free_gb": round(shutil.disk_usage(self.root).free / 1e9, 1)}

    async def list(self, prefix: str) -> list[RemoteEntry]:
        self._require_root()
        base = self._resolve(prefix) if prefix else self.root
        if not base.is_dir():
            return []
        out = []
        for child in sorted(base.iterdir()):
            info = child.stat()
            out.append(RemoteEntry(path=str(child.relative_to(self.root)), size=info.st_size,
                                   mtime=info.st_mtime, is_dir=child.is_dir()))
        return out

    async def upload(self, local: Path, remote: str, *, progress=None) -> None:
        self._require_root()
        dest = self._resolve(remote)
        stage_copy(local, dest)
        if progress:
            progress(1.0, dest.name)

    async def download(self, remote: str, local: Path, *, progress=None) -> None:
        self._require_root()
        src = self._resolve(remote)
        if not src.is_file():
            raise StorageError(f"{remote} is not a file")
        stage_copy(src, local)
        if progress:
            progress(1.0, local.name)

    async def stat(self, remote: str) -> RemoteEntry | None:
        self._require_root()
        target = self._resolve(remote)
        if not target.exists():
            return None
        info = target.stat()
        return RemoteEntry(path=str(target.relative_to(self.root)), size=info.st_size,
                           mtime=info.st_mtime, is_dir=target.is_dir())

    async def delete(self, remote: str) -> None:
        self._require_root()
        target = self._resolve(remote)
        if not target.exists():
            raise StorageError(f"{remote} does not exist")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        self._touch()

    def _touch(self) -> None:
        # Keeps the mtime ordering of the directory listing meaningful after a delete.
        try:
            self.root.touch()
        except OSError:
            pass

    def describe(self) -> dict:
        return {"kind": "local", "root": str(self.root), "exists": self.root.is_dir(),
                "checked_at": time.time()}
