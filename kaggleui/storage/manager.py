"""Storage destinations: connections from the workbench store turned into live providers.

Secrets are resolved by name at call time, never stored in the connection or echoed to the panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .. import secretstore
from ..schemas import StorageConnection
from .base import RemoteEntry, StorageError, StorageProvider, check_remote
from .local import LocalProvider
from .rclone import RcloneProvider
from .remotes import GitHubProvider, HFProvider

if TYPE_CHECKING:
    from ..config import Paths

__all__ = ["StorageError", "StorageManager", "RemoteEntry", "StorageProvider", "build_provider"]


def _config(connection: StorageConnection) -> dict[str, str]:
    return {k: str(v) for k, v in connection.config.items() if v is not None}


def build_provider(connection: StorageConnection, paths: Paths) -> StorageProvider:
    """Instantiate the provider for a stored connection."""
    cfg = _config(connection)
    token = secretstore.get(connection.secret) if connection.secret else None
    cache = Path(paths.cache)
    if connection.kind == "local":
        root = cfg.get("root")
        if not root:
            raise StorageError("local connection needs config.root")
        return LocalProvider(root)
    if connection.kind == "rclone":
        remote = cfg.get("remote")
        if not remote:
            raise StorageError("rclone connection needs config.remote, e.g. 'gdrive:kaggleui'")
        return RcloneProvider(remote, config_file=cfg.get("config_file"), binary=cfg.get("binary"), cache=cache)
    if connection.kind == "hf":
        return HFProvider(cfg.get("repo", ""), repo_type=cfg.get("repo_type", "model"), token=token, cache=cache)
    if connection.kind == "github":
        return GitHubProvider(cfg.get("repo", ""), path=cfg.get("path", "kaggleui"),
                              branch=cfg.get("branch", "main"), token=token, cache=cache)
    raise StorageError(f"unknown connection kind {connection.kind}")


class StorageManager:
    """Connections from the workbench store, turned into live providers on demand."""

    def __init__(self, paths: Paths, workbench):
        self.paths = paths
        self.workbench = workbench

    def connection(self, name: str) -> StorageConnection:
        connection = self.workbench.connection(name)
        if connection is None:
            raise StorageError(f"no storage connection named {name!r}")
        return connection

    def provider(self, name: str) -> StorageProvider:
        return build_provider(self.connection(name), self.paths)

    def save(self, connection: StorageConnection) -> StorageConnection:
        return self.workbench.put_connection(connection)

    def delete(self, name: str, artifacts: list) -> None:
        """Refuse to drop a connection an artifact's provenance still names."""
        if self.workbench.connection(name) is None:
            raise StorageError(f"no storage connection named {name!r}")
        if any((m.provenance or {}).get("connection") == name for m in artifacts):
            raise StorageError(f"{name} is referenced by exported artifacts; export elsewhere first")
        self.workbench.delete_connection(name)

    def public(self) -> list[dict]:
        return [redact(c) for c in self.workbench.connections()]

    async def test(self, name: str) -> dict:
        return await self.provider(name).test()

    async def list(self, name: str, prefix: str = "") -> list[RemoteEntry]:
        return await self.provider(name).list(prefix)

    async def upload(self, name: str, local: Path, remote: str, *, progress=None) -> None:
        connection = self.connection(name)
        if connection.read_only:
            raise StorageError(f"{name} is marked read-only")
        await self.provider(name).upload(local, remote, progress=progress)

    async def download(self, name: str, remote: str, local: Path, *, progress=None) -> None:
        await self.provider(name).download(remote, local, progress=progress)


def redact(connection: StorageConnection) -> dict:
    """Connection shape safe for the panel: config values that look like secrets are masked."""
    hidden = {"token", "password", "key", "secret"}
    cfg = {}
    for key, value in _config(connection).items():
        cfg[key] = "***" if any(word in key.lower() for word in hidden) else value
    return {"name": connection.name, "kind": connection.kind, "config": cfg,
            "secret": connection.secret, "read_only": connection.read_only}


def require_remote(remote: str) -> str:
    return check_remote(remote)
