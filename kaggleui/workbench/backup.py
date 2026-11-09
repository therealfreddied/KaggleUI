"""Artifact backup and restore on storage destinations.

Remote layout::

    artifacts/<artifact>/<version>/<file>       model weights and companions
    workspace-references/<workspace>/<id>@<v>.json
    manifests/<artifact>@<version>.json         written last, the commit marker

Only ``StorageManager`` is used, so rclone still runs ``copyto`` and never deletes. A restore verifies
each downloaded file against the manifest before registering it locally; remote-only entries never
pretend to be local artifacts.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from ..schemas import ArtifactManifest
from ..storage.base import StorageError
from .registry import RegistryError, sha256_of

if TYPE_CHECKING:
    from ..server.state import AppState

MANIFEST_PREFIX = "manifests"
ARTIFACT_PREFIX = "artifacts"
REFERENCE_PREFIX = "workspace-references"

class BackupError(Exception):
    pass

def _root(target: str) -> str:
    root = (target or "kaggleui").strip("/")
    if any(part in {".", ".."} for part in root.split("/") if part):
        raise BackupError("backup path must be a relative directory")
    return root or "kaggleui"

def _progress(ctx, index: int, total: int, label: str):
    return lambda frac, line: ctx.progress((index + frac) / max(total, 1), line or label)

def _state(state: AppState) -> dict[str, Any]:
    return state.workbench.workspace_state()

def _save_records(state: AppState, records: dict[str, dict]) -> None:
    info = _state(state)
    state.workbench.save_workspace_state({**info, "artifact_backups": records})

def _record_key(artifact_id: str, version: str) -> str:
    return f"{artifact_id}@{version}"

def _resolve(state: AppState, artifact_ids: list[str], deployment_ids: list[str]) -> list[ArtifactManifest]:
    wanted: list[tuple[str, str | None]] = []
    for raw in artifact_ids:
        artifact_id, _, version = str(raw).partition("@")
        if artifact_id:
            wanted.append((artifact_id, version or None))
    for dep_id in deployment_ids:
        record = state.workbench.deployment_record(dep_id)
        if record is None:
            raise BackupError(f"no deployment {dep_id}")
        from ..schemas import DeploymentSpec

        spec = DeploymentSpec.model_validate(record.spec)
        if spec.model.source != "artifact":
            raise BackupError(f"deployment {dep_id} does not use a stored artifact")
        wanted.append((spec.model.id, spec.model.variant or None))
    out: list[ArtifactManifest] = []
    for artifact_id, version in wanted:
        manifest = state.artifacts.get(artifact_id, version)
        if manifest is None:
            raise BackupError(f"no local artifact {artifact_id}{'@' + version if version else ''}")
        if all(m.artifact_id != manifest.artifact_id or m.version != manifest.version for m in out):
            out.append(manifest)
    return out

async def backup(state: AppState, ctx, *, connection_name: str, artifact_ids: list[str] | None = None,
                 deployment_ids: list[str] | None = None, target: str = "kaggleui") -> dict[str, Any]:
    connection = state.workbench.connection(connection_name)
    if connection is None:
        raise BackupError(f"no storage connection named {connection_name}")
    if connection.read_only:
        raise BackupError(f"{connection_name} is marked read-only")
    if connection.kind == "local":
        raise BackupError("local connections are not a backup; choose rclone, hf or github")
    manifests = _resolve(state, artifact_ids or [], deployment_ids or [])
    if not manifests:
        raise BackupError("choose at least one artifact or deployment to back up")
    root = _root(target)
    records = dict(_state(state).get("artifact_backups") or {})
    info = _state(state)
    for manifest in manifests:
        key = _record_key(manifest.artifact_id, manifest.version)
        records[key] = {**records.get(key, {}), "status": "restore_pending", "connection": connection_name,
                        "target": root, "at": time.time(), "error": None}
    _save_records(state, records)
    base = state.artifacts.dir_for(manifest.artifact_id, manifest.version) if manifests else None
    done = 0
    for manifest in manifests:
        base = state.artifacts.dir_for(manifest.artifact_id, manifest.version)
        if not base.exists():
            raise BackupError(f"artifact directory is missing for {manifest.artifact_id}")
        prefix = f"{root}/{ARTIFACT_PREFIX}/{manifest.artifact_id}/{manifest.version}".strip("/")
        total = len(manifest.files) + 1
        for index, entry in enumerate(manifest.files):
            source = base / entry.path
            if not source.exists():
                raise BackupError(f"artifact file is missing: {entry.path}")
            if source.stat().st_size != entry.size or sha256_of(source) != entry.sha256:
                raise BackupError(f"artifact file no longer matches its manifest: {entry.path}")
            await state.storage.upload(connection_name, source, f"{prefix}/{entry.path}",
                                       progress=_progress(ctx, index, total, f"uploading {entry.path}"))
        reference = {"schema_version": 1, "workspace_id": info.get("workspace_id"),
                     "artifact_id": manifest.artifact_id, "version": manifest.version, "kind": manifest.kind,
                     "files": [f.model_dump() for f in manifest.files], "manifest": manifest.model_dump(),
                     "at": time.time()}
        ref_path = Path(tempfile.mkstemp(prefix="kui-ref-", suffix=".json")[1])
        try:
            ref_path.write_text(json.dumps(reference, indent=1))
            await state.storage.upload(connection_name, ref_path,
                                       f"{root}/{REFERENCE_PREFIX}/{info.get('workspace_id') or 'default'}/"
                                       f"{manifest.artifact_id}@{manifest.version}.json")
        finally:
            ref_path.unlink(missing_ok=True)
        manifest_path = Path(tempfile.mkstemp(prefix="kui-manifest-", suffix=".json")[1])
        try:
            manifest_path.write_text(json.dumps(manifest.model_dump(), indent=1))
            await state.storage.upload(connection_name, manifest_path,
                                       f"{root}/{MANIFEST_PREFIX}/{manifest.artifact_id}@{manifest.version}.json")
        finally:
            manifest_path.unlink(missing_ok=True)
        records[key] = {**records.get(key, {}), "status": "backed_up", "connection": connection_name,
                        "target": root, "at": time.time(), "error": None}
        _save_records(state, records)
        done += 1
        ctx.progress((done / len(manifests)), f"backed up {manifest.artifact_id}@{manifest.version}")
        ctx.artifact({"artifact_id": manifest.artifact_id, "version": manifest.version,
                      "connection": connection_name, "path": f"{root}/{MANIFEST_PREFIX}"})
    return {"status": "backed_up", "artifacts": [m.artifact_id for m in manifests],
            "connection": connection_name, "target": root}

async def remote_backups(state: AppState, connection_name: str, target: str = "kaggleui") -> list[dict[str, Any]]:
    """List manifests already present on a destination.

    A fresh install has no local backup records, so discovery has to read the remote commit
    markers; otherwise the restore UI can only accept an identifier typed from memory.
    """
    connection = state.workbench.connection(connection_name)
    if connection is None:
        raise BackupError(f"no storage connection named {connection_name}")
    if connection.kind == "local":
        raise BackupError("local connections are not a backup; choose rclone, hf or github")
    root = _root(target)
    prefix = f"{root}/{MANIFEST_PREFIX}"
    try:
        entries = await state.storage.list(connection_name, prefix)
    except StorageError as exc:
        message = str(exc).casefold()
        if any(marker in message for marker in ("not found", "does not exist", "no such file", "404")):
            return []
        raise
    rows: list[dict[str, Any]] = []
    for entry in entries:
        name = PurePosixPath(entry.path).name
        if not name.endswith(".json"):
            continue
        artifact_id, _, version = name[:-len(".json")].rpartition("@")
        if not artifact_id or not version:
            continue
        rows.append({"artifact_id": artifact_id, "version": version, "connection": connection_name,
                     "size": entry.size, "path": entry.path})
    return sorted(rows, key=lambda row: (row["artifact_id"], row["version"]))

async def restore(state: AppState, ctx, *, connection_name: str, artifact_id: str, version: str | None = None,
                  target: str = "kaggleui") -> dict[str, Any]:
    connection = state.workbench.connection(connection_name)
    if connection is None:
        raise BackupError(f"no storage connection named {connection_name}")
    if connection.kind == "local":
        raise BackupError("local connections are not a backup source")
    if not artifact_id or any(ch in artifact_id for ch in "/\\") or ".." in artifact_id:
        raise BackupError("artifact id is not safe")
    root = _root(target)
    records = dict(_state(state).get("artifact_backups") or {})
    if not version:
        known = [k for k in records if k.split("@", 1)[0] == artifact_id and records[k].get("status") == "backed_up"]
        if not known:
            raise BackupError("version is required when the artifact has not been backed up by this install")
        version = known[-1].split("@", 1)[1]
    key = _record_key(artifact_id, version)
    records[key] = {**records.get(key, {}), "status": "restore_pending", "connection": connection_name,
                    "target": root, "at": time.time(), "error": None}
    _save_records(state, records)
    scratch = Path(tempfile.mkdtemp(prefix="kui-restore-", dir=str(state.paths.cache)))
    try:
        remote_manifest = f"{root}/{MANIFEST_PREFIX}/{artifact_id}@{version}.json"
        local_manifest = scratch / "manifest.json"
        await state.storage.download(connection_name, remote_manifest, local_manifest)
        try:
            manifest = ArtifactManifest.model_validate_json(local_manifest.read_text())
        except ValueError as exc:
            raise BackupError("downloaded manifest is not a valid artifact manifest") from exc
        if manifest.artifact_id != artifact_id or manifest.version != version:
            raise BackupError("manifest identity does not match the requested artifact")
        files: list[Path] = []
        for index, entry in enumerate(manifest.files):
            if entry.path.startswith("/") or ".." in Path(entry.path).parts:
                raise BackupError(f"manifest contains an unsafe path: {entry.path}")
            local = scratch / entry.path
            remote = f"{root}/{ARTIFACT_PREFIX}/{artifact_id}/{version}/{entry.path}"
            await state.storage.download(connection_name, remote, local,
                                         progress=_progress(ctx, index, len(manifest.files) + 1,
                                                            f"downloading {entry.path}"))
            if local.stat().st_size != entry.size or sha256_of(local) != entry.sha256:
                raise BackupError(f"downloaded file does not match the manifest: {entry.path}")
            files.append(local)
        ctx.progress(max(0.0, len(manifest.files) / (len(manifest.files) + 1)), "registering artifact")
        saved = state.artifacts.add(manifest, files, overwrite=True)
        records[key] = {**records.get(key, {}), "status": "backed_up", "connection": connection_name,
                        "target": root, "at": time.time(), "error": None}
        _save_records(state, records)
        return {"status": "restored", "artifact_id": saved.artifact_id, "version": saved.version,
                "connection": connection_name, "target": root}
    except (BackupError, RegistryError, OSError, StorageError) as exc:
        # "failed", not "missing_dependency": hash mismatches and network errors are not a
        # missing remote object, and every failure must move the record out of restore_pending.
        records[key] = {**records.get(key, {}), "status": "failed", "error": str(exc),
                        "connection": connection_name, "target": root, "at": time.time()}
        _save_records(state, records)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

def status(state: AppState, connection_name: str | None = None) -> list[dict[str, Any]]:
    records = dict(_state(state).get("artifact_backups") or {})
    local = {_record_key(m.artifact_id, m.version): m for m in state.artifacts.list()}
    rows: list[dict[str, Any]] = []
    for key, manifest in local.items():
        record = records.get(key) or {}
        if connection_name and record.get("connection") not in (None, connection_name):
            continue
        rows.append({"artifact_id": manifest.artifact_id, "version": manifest.version, "kind": manifest.kind,
                     "size": sum(f.size for f in manifest.files),
                     "status": record.get("status") if record.get("status") in {
                         "backed_up", "restore_pending", "missing_dependency", "failed",
                         "unverified"} else "local_only",
                     "connection": record.get("connection"), "target": record.get("target"),
                     "at": record.get("at"), "error": record.get("error")})
    for key, record in records.items():
        if key in local:
            continue
        artifact_id, _, version = key.partition("@")
        if record.get("status") == "backed_up":
            if connection_name and record.get("connection") != connection_name:
                continue
            rows.append({"artifact_id": artifact_id, "version": version, "kind": "unknown", "size": 0,
                         "status": "remote_only", "connection": record.get("connection"),
                         "target": record.get("target"), "at": record.get("at"), "error": None})
    return sorted(rows, key=lambda r: (r["artifact_id"], r["version"]))

