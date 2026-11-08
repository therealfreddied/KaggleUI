"""Workspace, CLI-management and shutdown routes.

Configuration routes use the admin session or a scoped management key. The shutdown route requires
the explicit ``shutdown`` scope because it ends this KaggleUI process; it never claims to stop the
Kaggle notebook itself.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from .. import __version__
from ..deploy.manager import DeployError
from ..gateway.common import state_of
from ..schemas import WorkspaceSettings
from ..storage import StorageError
from ..util import short_id
from ..workbench import backup as artifact_backup
from ..workbench import sync as workspace_sync
from ..workbench.runner import start_local
from ..workbench.workspace import (
    WorkspaceError,
    apply_document,
    export_document,
    public_state,
    validate_document,
)
from .management import SCOPES, require_scope

router = APIRouter(prefix="/_kui/api", tags=["workspace"])

def _st(request: Request):
    return state_of(request)

def _admin(request: Request):
    rec = require_scope(request, "status")
    if rec.get("label") != "admin":
        raise HTTPException(403, "Admin token required")
    return _st(request)

def _scope(request: Request, scope: str):
    require_scope(request, scope)
    return _st(request)

def _sync_error(exc: Exception) -> HTTPException:
    if isinstance(exc, workspace_sync.SyncConflict):
        return HTTPException(409, {"message": str(exc), "local_sha": exc.local_sha,
                                   "remote_sha": exc.remote_sha, "local_changed": exc.local_changed,
                                   "remote_changed": exc.remote_changed})
    return HTTPException(422, str(exc))

# -- workspace ---------------------------------------------------------------------------------
@router.get("/workspace")
def workspace(request: Request):
    st = _scope(request, "workspace")
    doc = export_document(st)
    return {**public_state(st), "settings": st.settings.model_dump(),
            "deployments": len(doc.deployments), "deployment_sets": len(doc.deployment_sets),
            "artifacts": len(doc.artifacts), "connections": len(doc.storage_connections)}

@router.put("/workspace/config")
async def configure_workspace(request: Request, patch: dict[str, Any]):
    st = _scope(request, "workspace")
    allowed = set(WorkspaceSettings.model_fields)
    unknown = set(patch) - allowed
    if unknown:
        raise HTTPException(422, f"unknown workspace fields: {', '.join(sorted(unknown))}")
    data = {**st.settings.workspace.model_dump(), **patch}
    try:
        st.settings.workspace = WorkspaceSettings.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    st.workbench.save_settings(st.settings.model_dump())
    st.bus.publish("settings", st.settings.model_dump())
    workspace_sync.schedule_auto_backup(st)
    return {"workspace": st.settings.workspace.model_dump()}

@router.get("/workspace/export")
def export_workspace(request: Request):
    st = _scope(request, "workspace")
    return export_document(st).model_dump()

@router.post("/workspace/import")
def import_workspace(request: Request, body: dict[str, Any]):
    st = _scope(request, "workspace")
    if not body.get("confirm"):
        raise HTTPException(422, "confirm=true is required to replace local workspace settings")
    raw = body.get("document")
    if not isinstance(raw, dict):
        raise HTTPException(422, "document must be a workspace document")
    try:
        doc = validate_document(raw)
        result = apply_document(st, doc, replace=bool(body.get("replace", True)),
                                replace_connections=bool(body.get("replace_connections", True)))
    except WorkspaceError as exc:
        raise HTTPException(422, str(exc)) from exc
    return result

@router.get("/workspace/preview")
async def preview_workspace(request: Request):
    st = _scope(request, "workspace")
    try:
        return await workspace_sync.preview(st)
    except (workspace_sync.SyncError, WorkspaceError) as exc:
        raise _sync_error(exc) from exc

@router.get("/workspace/status")
async def sync_status(request: Request):
    st = _scope(request, "workspace")
    return await workspace_sync.status(st)

@router.post("/workspace/backup")
async def backup_workspace(request: Request, body: dict[str, Any] | None = None):
    st = _scope(request, "workspace")
    body = body or {}
    try:
        return await workspace_sync.backup(st, force=bool(body.get("force")),
                                           message=body.get("message"), fork=bool(body.get("fork")),
                                           confirm=bool(body.get("confirm")))
    except (workspace_sync.SyncError, WorkspaceError) as exc:
        raise _sync_error(exc) from exc

@router.post("/workspace/restore")
async def restore_workspace(request: Request, body: dict[str, Any] | None = None):
    st = _scope(request, "workspace")
    body = body or {}
    try:
        return await workspace_sync.restore(st, confirm=bool(body.get("confirm")),
                                            resolution=str(body.get("resolution") or "replace"),
                                            sha=body.get("sha"))
    except (workspace_sync.SyncError, WorkspaceError) as exc:
        raise _sync_error(exc) from exc

@router.get("/workspace/history")
async def workspace_history(request: Request, limit: int = 20):
    st = _scope(request, "workspace")
    try:
        return await workspace_sync.history(st, limit)
    except workspace_sync.SyncError as exc:
        raise _sync_error(exc) from exc

@router.post("/workspace/rollback")
async def rollback_workspace(request: Request, body: dict[str, Any]):
    st = _scope(request, "workspace")
    if not body.get("sha"):
        raise HTTPException(422, "sha is required")
    if not body.get("confirm"):
        raise HTTPException(422, "confirm=true is required to roll back")
    try:
        return await workspace_sync.restore(st, confirm=True, resolution="replace", sha=body["sha"])
    except (workspace_sync.SyncError, WorkspaceError) as exc:
        raise _sync_error(exc) from exc

# -- management keys ---------------------------------------------------------------------------
@router.get("/management")
def list_management(request: Request):
    st = _admin(request)
    return {"tokens": st.management.list(), "scopes": list(SCOPES)}

@router.post("/management")
def create_management(request: Request, body: dict[str, Any]):
    st = _admin(request)
    try:
        token, record = st.management.create(str(body.get("label") or "cli"), body.get("scopes"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"token": token, "record": record, "notice": "This value is shown once; store it in kui, not Git"}

@router.delete("/management/{token_id}")
def revoke_management(request: Request, token_id: str):
    st = _admin(request)
    if not st.management.revoke(token_id):
        raise HTTPException(404, "no such active management key")
    return {"revoked": token_id}

# -- instance / CLI ----------------------------------------------------------------------------
def _ensure_instance_id(st) -> str:
    info = st.workbench.workspace_state()
    if not info.get("instance_id"):
        from ..util import short_id

        info["instance_id"] = short_id(10)
        st.workbench.save_workspace_state(info)
    return str(info["instance_id"])

@router.get("/instance")
def instance(request: Request):
    st = _scope(request, "status")
    doc = export_document(st)
    return {
        "instance_id": _ensure_instance_id(st), "workspace_id": doc.meta.workspace_id,
        "workspace_name": doc.meta.name, "version": __version__, "started_at": st.started_at,
        "url": st.main_url(), "host": st.host.name, "mock": st.mock, "fake_hw": st.fake_hw,
        "deployments": st.manager.list(), "jobs": st.jobs.list(limit=20),
        "uptime_s": max(0.0, time.time() - st.started_at),
    }

@router.get("/workbench/deployments")
def cli_deployments(request: Request):
    st = _scope(request, "deployments")
    return {"deployments": st.manager.list()}

@router.post("/workbench/deployments/{dep_id}/start")
async def cli_start_deployment(request: Request, dep_id: str):
    st = _scope(request, "deployments")
    dep = st.manager.get(dep_id)
    if dep is not None and dep.status in {"queued", "installing", "downloading", "loading", "ready"}:
        return dep.public()
    record = st.workbench.deployment_record(dep_id)
    if record is None:
        raise HTTPException(404, f"no deployment {dep_id}")
    if dep is not None:
        await st.manager.stop(dep_id, remove=False)
    try:
        return st.manager.create(_spec(record)).public()
    except DeployError as exc:
        raise HTTPException(422, str(exc)) from exc

def _spec(record):
    from ..schemas import DeploymentSpec

    return DeploymentSpec.model_validate(record.spec)

@router.post("/workbench/deployments/{dep_id}/stop")
async def cli_stop_deployment(request: Request, dep_id: str):
    st = _scope(request, "deployments")
    if st.manager.get(dep_id) is None and st.workbench.deployment_record(dep_id) is None:
        raise HTTPException(404, f"no deployment {dep_id}")
    await st.manager.stop(dep_id, remove=False)
    return {"stopped": dep_id, "kept_definition": True}

@router.get("/workbench/jobs")
def cli_jobs(request: Request, limit: int = 30):
    st = _scope(request, "jobs")
    return {"jobs": st.jobs.list(limit=limit)}

@router.get("/workbench/storage/cli")
def cli_storage(request: Request):
    st = _scope(request, "storage")
    return {"connections": st.storage.public(), "artifacts": st.artifacts.public()}

@router.post("/shutdown")
async def shutdown(request: Request, body: dict[str, Any] | None = None):
    st = _scope(request, "shutdown")
    body = body or {}
    if not body.get("confirm"):
        raise HTTPException(422, "confirm=true is required; this stops the KaggleUI server")
    stop_models = body.get("stop_models", True)
    if stop_models:
        await st.manager.shutdown()
    if st.auto_backup_task is not None:
        st.auto_backup_task.cancel()
    if st.settings.workspace.auto_backup and st.settings.workspace.enabled:
        try:
            await workspace_sync.backup(st, confirm=True)
        except Exception as exc:  # A backup failure must not trap the server.
            st.bus.publish("workspace", {"action": "backup_failed", "error": str(exc)})
    st.workbench.save_settings(st.settings.model_dump())
    asyncio.create_task(_terminate())
    return {"status": "stopping", "stop_models": stop_models,
            "note": "This ends the KaggleUI server, not the Kaggle notebook session."}

async def _terminate() -> None:
    await asyncio.sleep(0.35)
    os.kill(os.getpid(), signal.SIGTERM)



# -- artifact backup / restore ----------------------------------------------------------------
def _job_log(st, kind: str):
    return st.paths.logs / f"{kind}-{short_id()}.log"

@router.get("/workbench/model-backups")
def artifact_backup_status(request: Request, connection: str | None = None):
    st = _scope(request, "storage")
    return {"artifacts": artifact_backup.status(st, connection)}

@router.get("/workbench/model-backups/remote")
async def remote_artifact_backups(request: Request, connection: str, target_path: str = "kaggleui"):
    st = _scope(request, "storage")
    try:
        rows = await artifact_backup.remote_backups(st, connection, target_path)
    except (artifact_backup.BackupError, StorageError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"artifacts": rows}

@router.post("/workbench/model-backups")
async def backup_artifacts(request: Request, body: dict[str, Any]):
    st = _scope(request, "storage")
    if not body.get("confirm"):
        raise HTTPException(422, "confirm=true is required; this uploads model files to the destination")
    connection = str(body.get("connection") or "")
    if not connection:
        raise HTTPException(422, "connection is required")
    ids = [str(x) for x in body.get("artifact_ids") or []]
    deps = [str(x) for x in body.get("deployment_ids") or []]
    if not ids and not deps:
        raise HTTPException(422, "choose artifact_ids or deployment_ids")
    target = str(body.get("target_path") or "kaggleui")
    job = st.jobs.create("artifact.backup", {"connection": connection, "artifact_ids": ids,
                                             "deployment_ids": deps, "target_path": target},
                         logs_path=str(_job_log(st, "backup")))

    async def work(ctx):
        ctx.log(f"backing up {len(ids) + len(deps)} artifact reference(s) to {connection}")
        await artifact_backup.backup(st, ctx, connection_name=connection, artifact_ids=ids,
                                     deployment_ids=deps, target=target)

    await start_local(st.jobs, job, work)
    return st.jobs.get(job.id).public()

@router.post("/workbench/model-backups/restore")
async def restore_artifacts(request: Request, body: dict[str, Any]):
    st = _scope(request, "storage")
    if not body.get("confirm"):
        raise HTTPException(422, "confirm=true is required; this downloads model files")
    connection = str(body.get("connection") or "")
    artifact_id = str(body.get("artifact_id") or "")
    if not connection or not artifact_id:
        raise HTTPException(422, "connection and artifact_id are required")
    version = body.get("version")
    target = str(body.get("target_path") or "kaggleui")
    job = st.jobs.create("artifact.restore", {"connection": connection, "artifact_id": artifact_id,
                                              "version": version, "target_path": target},
                         logs_path=str(_job_log(st, "restore")))

    async def work(ctx):
        ctx.log(f"restoring {artifact_id} from {connection}:{target}")
        result = await artifact_backup.restore(st, ctx, connection_name=connection, artifact_id=artifact_id,
                                               version=version, target=target)
        ctx.artifact(result)

    await start_local(st.jobs, job, work)
    return st.jobs.get(job.id).public()

@router.get("/workbench/jobs/{job_id}")
def cli_job(request: Request, job_id: str):
    st = _scope(request, "jobs")
    job = st.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return job.public()

@router.get("/workbench/jobs/{job_id}/log")
def cli_job_log(request: Request, job_id: str, offset: int = 0, limit: int = 200):
    st = _scope(request, "jobs")
    job = st.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    from ..workbench.jobs import read_log

    return {"job": job.public(), "lines": read_log(job.logs_path, max(0, offset), max(1, min(limit, 2000)))}
