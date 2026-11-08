"""JSON stores for workbench state: deployment sets, storage connections, deployment records.

Files live under ``$KAGGLEUI_HOME/workbench`` and are written atomically (temp file + ``Path.replace``)
so a crash never leaves a half-written document behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..schemas import DeploymentSet, DeploymentSpec, StorageConnection
from ..util import short_id
from .models import DeploymentRecord

SCHEMA_VERSION = 1


def _read(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return default
    return data if isinstance(data, dict) else default


def _write(path: Path, data: dict, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class WorkbenchStore:
    """One object fronts every workbench JSON file; call :meth:`load` once at startup."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._sets: dict[str, DeploymentSet] = {}
        self._connections: dict[str, StorageConnection] = {}
        self._deployments: dict[str, DeploymentRecord] = {}
        self.active_set: str | None = None

    # -- files ---------------------------------------------------------------------------------
    @property
    def sets_file(self) -> Path:
        return self.root / "sets.json"

    @property
    def storage_file(self) -> Path:
        return self.root / "storage.json"

    @property
    def deployments_file(self) -> Path:
        return self.root / "deployments.json"

    @property
    def settings_file(self) -> Path:
        return self.root / "settings.json"

    @property
    def workspace_file(self) -> Path:
        return self.root / "workspace.json"

    # -- portable settings --------------------------------------------------------------------
    def load_settings(self) -> dict[str, Any] | None:
        """Last saved ``Settings``, or ``None`` when this install has never persisted one."""
        data = _read(self.settings_file, {})
        settings = data.get("settings")
        return settings if isinstance(settings, dict) else None

    def save_settings(self, settings: dict[str, Any]) -> None:
        _write(self.settings_file, {"schema_version": SCHEMA_VERSION, "settings": settings})

    def workspace_state(self) -> dict[str, Any]:
        return _read(self.workspace_file, {"schema_version": SCHEMA_VERSION})

    def save_workspace_state(self, state: dict[str, Any]) -> None:
        payload = {"schema_version": SCHEMA_VERSION, **state}
        _write(self.workspace_file, payload)

    def load(self) -> None:
        sets = _read(self.sets_file, {"schema_version": SCHEMA_VERSION, "sets": []})
        self.active_set = sets.get("active_set")
        self._sets = {}
        for raw in sets.get("sets") or []:
            try:
                s = DeploymentSet.model_validate(raw)
            except ValueError:
                continue
            self._sets[s.id] = s
        storage = _read(self.storage_file, {"schema_version": SCHEMA_VERSION, "connections": []})
        self._connections = {}
        for raw in storage.get("connections") or []:
            try:
                c = StorageConnection.model_validate(raw)
            except ValueError:
                continue
            self._connections[c.name] = c
        deps = _read(self.deployments_file, {"schema_version": SCHEMA_VERSION, "deployments": []})
        self._deployments = {}
        for raw in deps.get("deployments") or []:
            try:
                r = DeploymentRecord.model_validate(raw)
            except ValueError:
                continue
            self._deployments[r.id] = r

    def save(self) -> None:
        _write(self.sets_file, {"schema_version": SCHEMA_VERSION, "active_set": self.active_set,
                                "sets": [s.model_dump() for s in self._sets.values()]})
        _write(self.storage_file, {"schema_version": SCHEMA_VERSION,
                                   "connections": [c.model_dump() for c in self._connections.values()]})
        _write(self.deployments_file, {"schema_version": SCHEMA_VERSION,
                                       "deployments": [r.model_dump() for r in self._deployments.values()]})

    # -- deployment sets -----------------------------------------------------------------------
    def deployment_sets(self) -> list[DeploymentSet]:
        return list(self._sets.values())

    def deployment_set(self, set_id: str) -> DeploymentSet | None:
        return self._sets.get(set_id)

    def put_deployment_set(self, s: DeploymentSet) -> DeploymentSet:
        if not s.id:
            s = s.model_copy(update={"id": short_id()})
        self._sets[s.id] = s
        self.save()
        return s

    def delete_deployment_set(self, set_id: str) -> None:
        self._sets.pop(set_id, None)
        if self.active_set == set_id:
            self.active_set = None
        self.save()

    def mark_active(self, set_id: str | None) -> None:
        self.active_set = set_id
        self.save()

    # -- storage connections -------------------------------------------------------------------
    def connections(self) -> list[StorageConnection]:
        return list(self._connections.values())

    def connection(self, name: str) -> StorageConnection | None:
        return self._connections.get(name)

    def put_connection(self, c: StorageConnection) -> StorageConnection:
        self._connections[c.name] = c
        self.save()
        return c

    def delete_connection(self, name: str) -> None:
        self._connections.pop(name, None)
        self.save()

    # -- deployment records (read by the llama-swap launch helper) -----------------------------
    def deployment_records(self) -> list[DeploymentRecord]:
        return list(self._deployments.values())

    def deployment_record(self, dep_id: str) -> DeploymentRecord | None:
        return self._deployments.get(dep_id)

    def put_deployment(self, dep_id: str, spec: DeploymentSpec, served_name: str, *, created_at: float,
                       runtime: dict[str, Any] | None = None) -> None:
        self._deployments[dep_id] = DeploymentRecord(id=dep_id, spec=spec.model_dump(), served_name=served_name,
                                                     created_at=created_at, runtime=runtime or {})
        self.save()

    def drop_deployment(self, dep_id: str) -> None:
        if self._deployments.pop(dep_id, None) is not None:
            self.save()

    def raw(self) -> dict[str, Any]:
        return {"sets": len(self._sets), "connections": len(self._connections),
                "deployments": len(self._deployments), "active_set": self.active_set}
