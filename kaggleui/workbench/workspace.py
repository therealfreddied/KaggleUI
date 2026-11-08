"""Portable workspace snapshots: export, validate, preview and apply.

A snapshot is configuration, never runtime state. It contains settings, deployment definitions,
deployment sets, storage destinations and artifact *references*. PIDs, ports, tunnel URLs, GPU
claims, loaded weights and secret values are deliberately excluded; a fresh session reconstructs
those. Applying a snapshot is transactional: the current document is backed up first, references
are validated, and nothing is started automatically unless the caller asks for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__
from ..schemas import (
    DeploymentSpec,
    SecretRequirement,
    WorkspaceArtifactRef,
    WorkspaceDocument,
    WorkspaceMeta,
)
from ..util import short_id
from .store import _write

if TYPE_CHECKING:
    from ..server.state import AppState

WORKSPACE_SCHEMA = 1
RUNTIME_OPTION_KEYS = frozenset({
    "pid", "port", "gpus", "vram_estimate", "vram_plan", "loaded_at", "last_used", "served_name",
    "runtime", "process", "tunnel_url", "hf_home", "resolved_path",
})
BANNED_OPTION_KEYS = frozenset({"trust_remote_code", "allow_remote_code"})

class WorkspaceError(Exception):
    """Snapshot invalid, newer than this build, or references something unavailable."""

def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

def _portable_spec(spec: DeploymentSpec) -> DeploymentSpec:
    data = spec.model_dump()
    options = {k: v for k, v in (data.get("options") or {}).items() if k not in RUNTIME_OPTION_KEYS}
    data["options"] = options
    return DeploymentSpec.model_validate(data)

def _safe_spec(spec: DeploymentSpec) -> tuple[DeploymentSpec, list[str]]:
    """Import-time safety: a restored definition never turns on remote model code by itself."""
    data = spec.model_dump()
    warnings: list[str] = []
    options = dict(data.get("options") or {})
    for key in sorted(BANNED_OPTION_KEYS):
        if options.pop(key, None):
            warnings.append(f"{key} was removed from {data.get('name') or data['model']['id']} on import")
    data["options"] = options
    return DeploymentSpec.model_validate(data), warnings

def _ensure_identity(state: AppState) -> tuple[dict[str, Any], str]:
    info = state.workbench.workspace_state()
    changed = False
    if not info.get("workspace_id"):
        info["workspace_id"] = short_id(10)
        changed = True
    if not info.get("name"):
        info["name"] = "Default"
        changed = True
    if changed:
        state.workbench.save_workspace_state(info)
    return info, str(info["workspace_id"])

def secret_requirements(state: AppState, specs: list[DeploymentSpec], connections) -> list[SecretRequirement]:
    out: list[SecretRequirement] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.model.source == "hf" and "HF_TOKEN" not in seen:
            out.append(SecretRequirement(name="HF_TOKEN", reason="gated or private Hugging Face models",
                                         required=False))
            seen.add("HF_TOKEN")
    for connection in connections:
        if connection.secret and connection.secret not in seen:
            out.append(SecretRequirement(name=connection.secret,
                                         reason=f"storage connection {connection.name}", required=False))
            seen.add(connection.secret)
    token_secret = state.settings.workspace.token_secret or "GITHUB_TOKEN"
    if token_secret not in seen:
        out.append(SecretRequirement(name=token_secret, reason="workspace GitHub sync", required=True))
        seen.add(token_secret)
    return out

def export_document(state: AppState) -> WorkspaceDocument:
    """Build the current portable document from live state, stripping runtime fields."""
    info, workspace_id = _ensure_identity(state)
    records = state.workbench.deployment_records()
    specs = [_portable_spec(DeploymentSpec.model_validate(record.spec)) for record in records]
    connections = state.workbench.connections()
    sets = state.workbench.deployment_sets()
    artifacts = [
        WorkspaceArtifactRef(artifact_id=m.artifact_id, version=m.version, kind=m.kind,
                             files=len(m.files), size=sum(f.size for f in m.files))
        for m in state.artifacts.list()
    ]
    meta = WorkspaceMeta(workspace_id=workspace_id, name=str(info.get("name") or "Default"),
                         app_version=__version__, exported_at=time.time(), source_host=state.host.name)
    from . import bench

    return WorkspaceDocument(meta=meta, settings=state.settings.model_copy(deep=True), deployments=specs,
                             deployment_sets=sets, active_set=state.workbench.active_set,
                             storage_connections=connections, artifacts=artifacts,
                             secret_requirements=secret_requirements(state, specs, connections),
                             benchmarks=bench.records(state))

def validate_document(raw: dict | WorkspaceDocument) -> WorkspaceDocument:
    """Parse and reject documents this build cannot apply without data loss."""
    if isinstance(raw, WorkspaceDocument):
        doc = raw
    else:
        try:
            doc = WorkspaceDocument.model_validate(raw)
        except ValueError as exc:
            raise WorkspaceError(f"Invalid workspace document: {exc}") from exc
    version = int(doc.meta.schema_version)
    if version > WORKSPACE_SCHEMA:
        raise WorkspaceError(f"Workspace schema {version} is newer than this build supports "
                             f"({WORKSPACE_SCHEMA}); upgrade KaggleUI instead of importing partially.")
    if version < 1:
        raise WorkspaceError(f"Unsupported workspace schema {version}")
    if not doc.meta.workspace_id.strip():
        raise WorkspaceError("Workspace document has no workspace_id")
    names = {c.name for c in doc.storage_connections}
    if len(names) != len(doc.storage_connections):
        raise WorkspaceError("Workspace document has duplicate storage connection names")
    set_ids = {s.id for s in doc.deployment_sets if s.id}
    if len(set_ids) != len([s for s in doc.deployment_sets if s.id]):
        raise WorkspaceError("Workspace document has duplicate deployment set ids")
    if doc.active_set and doc.active_set not in set_ids:
        raise WorkspaceError(f"active_set {doc.active_set!r} is not present in this document")
    for spec in doc.deployments:
        if not spec.model.id.strip():
            raise WorkspaceError("Deployment definition has an empty model id")
    return doc

def _definition_key(spec: DeploymentSpec) -> str:
    return "|".join((spec.modality, spec.model.source, spec.model.id, spec.model.variant or "",
                     spec.model.engine, spec.name or ""))

def _record_keys(state: AppState) -> dict[str, str]:
    return {_definition_key(DeploymentSpec.model_validate(record.spec)): record.id
            for record in state.workbench.deployment_records()}

def diff_documents(current: WorkspaceDocument, incoming: WorkspaceDocument) -> dict[str, Any]:
    """Human-readable, sectioned comparison used by preview and the UI."""
    cur_keys = {_definition_key(s): s for s in current.deployments}
    inc_keys = {_definition_key(s): s for s in incoming.deployments}
    cur_sets = {s.id: s for s in current.deployment_sets}
    inc_sets = {s.id: s for s in incoming.deployment_sets}
    cur_conn = {c.name: c for c in current.storage_connections}
    inc_conn = {c.name: c for c in incoming.storage_connections}
    cur_settings = current.settings.model_dump()
    inc_settings = incoming.settings.model_dump()
    changed_settings = sorted(section for section in inc_settings
                              if _canonical(cur_settings.get(section)) != _canonical(inc_settings[section]))
    warnings: list[str] = []
    if current.meta.workspace_id != incoming.meta.workspace_id:
        warnings.append(f"Workspace id changes from {current.meta.workspace_id} to "
                        f"{incoming.meta.workspace_id}; this may be a different setup.")
    if incoming.meta.app_version and incoming.meta.app_version != __version__:
        warnings.append(f"Snapshot was exported by KaggleUI {incoming.meta.app_version}; "
                        f"this instance runs {__version__}.")
    missing = [r.name for r in incoming.secret_requirements if r.required and not os.environ.get(r.name)]
    if missing:
        warnings.append("Missing required secrets: " + ", ".join(missing))
    hardware = incoming.settings.workspace
    if any(s.placement.devices for s in incoming.deployments if s.placement):
        warnings.append("Some deployment definitions pin GPU indices; review them on this host.")
    return {
        "same": _canonical(current.model_dump()) == _canonical(incoming.model_dump()),
        "workspace_id": {"local": current.meta.workspace_id, "incoming": incoming.meta.workspace_id},
        "name": {"local": current.meta.name, "incoming": incoming.meta.name},
        "settings": changed_settings,
        "deployments": {
            "added": sorted(set(inc_keys) - set(cur_keys)),
            "removed": sorted(set(cur_keys) - set(inc_keys)),
            "changed": sorted(k for k in set(cur_keys) & set(inc_keys)
                              if _canonical(cur_keys[k].model_dump()) != _canonical(inc_keys[k].model_dump())),
        },
        "deployment_sets": {
            "added": sorted(set(inc_sets) - set(cur_sets)),
            "removed": sorted(set(cur_sets) - set(inc_sets)),
            "changed": sorted(k for k in set(cur_sets) & set(inc_sets)
                              if _canonical(cur_sets[k].model_dump()) != _canonical(inc_sets[k].model_dump())),
        },
        "storage_connections": {
            "added": sorted(set(inc_conn) - set(cur_conn)),
            "removed": sorted(set(cur_conn) - set(inc_conn)),
            "changed": sorted(k for k in set(cur_conn) & set(inc_conn)
                              if _canonical(cur_conn[k].model_dump()) != _canonical(inc_conn[k].model_dump())),
        },
        "artifacts": {"incoming": len(incoming.artifacts), "local": len(current.artifacts)},
        "secret_requirements": [r.model_dump() for r in incoming.secret_requirements],
        "warnings": warnings,
        "auto_start_set": hardware.auto_start_set,
    }

def local_backup(state: AppState, doc: WorkspaceDocument, *, reason: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    path = state.workbench.root / "backups" / f"{stamp}-{reason}.json"
    _write(path, doc.model_dump(), mode=0o600)
    return path

def apply_document(state: AppState, incoming: WorkspaceDocument, *, replace: bool = True,
                   replace_connections: bool = True) -> dict[str, Any]:
    """Apply a validated snapshot; never starts deployments and never overwrites secret values."""
    doc = validate_document(incoming)
    current = export_document(state)
    backup = local_backup(state, current, reason="pre-restore")
    warnings: list[str] = []

    settings = doc.settings.model_copy(deep=True)
    # The active workspace identity and the local view of runtime capabilities belong to this install.
    state.settings = settings
    state.workbench.save_settings(settings.model_dump())

    if replace_connections:
        for existing in state.workbench.connections():
            if existing.name not in {c.name for c in doc.storage_connections}:
                state.workbench.delete_connection(existing.name)
    for connection in doc.storage_connections:
        state.workbench.put_connection(connection)

    incoming_set_ids = {s.id for s in doc.deployment_sets if s.id}
    if replace:
        for existing in state.workbench.deployment_sets():
            if existing.id and existing.id not in incoming_set_ids:
                state.workbench.delete_deployment_set(existing.id)
    for deployment_set in doc.deployment_sets:
        if not deployment_set.id:
            deployment_set.id = short_id(8)
        state.workbench.put_deployment_set(deployment_set)
    if doc.active_set:
        state.workbench.mark_active(doc.active_set)

    existing = _record_keys(state)
    incoming_keys = {_definition_key(spec) for spec in doc.deployments}
    if replace:
        for record in state.workbench.deployment_records():
            if _definition_key(DeploymentSpec.model_validate(record.spec)) not in incoming_keys:
                deployment = state.manager.deployments.get(record.id)
                if deployment is not None and deployment.status in {"queued", "installing", "downloading",
                                                                    "loading", "ready"}:
                    warnings.append(f"Kept running deployment {deployment.served_name} not present in snapshot")
                else:
                    state.workbench.drop_deployment(record.id)
    for spec in doc.deployments:
        safe, dropped = _safe_spec(spec)
        warnings.extend(dropped)
        dep_id = existing.get(_definition_key(safe)) or short_id(6)
        record = state.workbench.deployment_record(dep_id)
        state.workbench.put_deployment(dep_id, safe, safe.name or safe.model.id,
                                       created_at=(record.created_at if record else time.time()),
                                       runtime=(record.runtime if record else {}))
    state.manager.reload_definitions()

    for requirement in doc.secret_requirements:
        if requirement.required and not os.environ.get(requirement.name):
            warnings.append(f"Secret {requirement.name} is required ({requirement.reason}); "
                            "existing secrets were not overwritten or removed")
    state.workbench.save_workspace_state({
        **state.workbench.workspace_state(),
        "workspace_id": doc.meta.workspace_id,
        "name": doc.meta.name,
        "last_applied_at": time.time(),
    })
    state.bus.publish("workspace", {"action": "restored", "workspace_id": doc.meta.workspace_id,
                                    "backup": str(backup), "warnings": warnings})
    return {"backup": str(backup), "workspace_id": doc.meta.workspace_id, "warnings": warnings,
            "settings": state.settings.model_dump(), "deployments": len(doc.deployments),
            "deployment_sets": len(doc.deployment_sets)}

# Export time, app version and source host describe a snapshot, not its contents.
VOLATILE_META_KEYS = frozenset({"exported_at", "app_version", "source_host"})


def document_hash(doc: WorkspaceDocument | dict) -> str:
    """Content identity used for no-op detection and conflict checks."""
    value = doc.model_dump() if isinstance(doc, WorkspaceDocument) else json.loads(json.dumps(doc, default=str))
    meta = value.get("meta")
    if isinstance(meta, dict):
        meta = {k: v for k, v in meta.items() if k not in VOLATILE_META_KEYS}
        value["meta"] = meta
    return _canonical(value)

def public_state(state: AppState) -> dict[str, Any]:
    info = state.workbench.workspace_state()
    _, workspace_id = _ensure_identity(state)
    return {"workspace_id": workspace_id, "name": info.get("name") or "Default",
            "has_settings_file": state.workbench.settings_file.exists(),
            "sync": {k: v for k, v in info.items() if k != "schema_version"}}

