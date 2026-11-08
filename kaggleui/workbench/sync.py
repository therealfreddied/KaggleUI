"""GitHub workspace sync with explicit revisions, previews and conflicts.

The repository stores one JSON document per workspace. Every write sends the blob SHA it read, so a
concurrent change is rejected by GitHub instead of being silently overwritten. Local state records the
last remote SHA and the last exported hash; that lets the UI distinguish "nothing changed" from
"someone else changed the workspace" and from "this install has unsaved changes".
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import time
from typing import TYPE_CHECKING, Any

import httpx

from .. import __version__
from ..schemas import WorkspaceDocument
from .workspace import (
    WorkspaceError,
    apply_document,
    diff_documents,
    document_hash,
    export_document,
    validate_document,
)

if TYPE_CHECKING:
    from ..server.state import AppState

API = "https://api.github.com"
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MAX_DOC_BYTES = 950_000

class SyncError(Exception):
    """Remote unreachable, credential missing, or the change cannot be applied safely."""

class SyncConflict(SyncError):
    def __init__(self, message: str, *, local_sha: str | None = None, remote_sha: str | None = None,
                 local_changed: bool = False, remote_changed: bool = False):
        super().__init__(message)
        self.local_sha = local_sha
        self.remote_sha = remote_sha
        self.local_changed = local_changed
        self.remote_changed = remote_changed

def _safe_path(value: str) -> str:
    cleaned = value.strip().strip("/")
    parts = [p for p in cleaned.split("/") if p]
    if not parts or any(p in {".", ".."} for p in parts):
        raise SyncError("workspace path must be a repository-relative directory")
    if any(len(p) > 120 for p in parts):
        raise SyncError("workspace path segments must be at most 120 characters")
    return "/".join(parts)

def _document_path(path: str) -> str:
    return f"{_safe_path(path)}/workspace.json"

class GitHubWorkspace:
    """Minimal contents-API client; one workspace file, one discovery record per instance."""

    def __init__(self, repo: str, branch: str, path: str, token: str, *, timeout: float = 20.0):
        if not REPO_RE.match(repo.strip()):
            raise SyncError("workspace repo must look like owner/name")
        if not token:
            raise SyncError("GitHub token is not set; add GITHUB_TOKEN in the panel or environment")
        self.repo = repo.strip()
        self.branch = branch.strip() or "main"
        self.path = _document_path(path)
        self.token = token
        self.timeout = timeout

    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28", "User-Agent": f"KaggleUI/{__version__}"}

    def _url(self, path: str) -> str:
        return f"{API}/repos/{self.repo}/contents/{path}"

    async def get_file(self, path: str | None = None, ref: str | None = None) -> tuple[dict, str] | None:
        params = {"ref": ref} if ref else {"ref": self.branch}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self._url(path or self.path), headers=self._headers(), params=params)
        if response.status_code == 404:
            return None
        if response.status_code in (401, 403):
            raise SyncError("GitHub rejected the workspace token (check repository contents permission)")
        if response.status_code != 200:
            raise SyncError(f"GitHub returned {response.status_code} while reading the workspace")
        payload = response.json()
        if payload.get("type") != "file":
            raise SyncError("workspace path is not a file")
        encoded = payload.get("content") or ""
        if payload.get("encoding") != "base64" or not encoded:
            raise SyncError("GitHub did not return file content (the workspace may exceed 1 MB)")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise SyncError("GitHub returned invalid workspace content") from exc
        try:
            import json

            doc = json.loads(raw)
        except ValueError as exc:
            raise SyncError("Remote workspace is not valid JSON") from exc
        return (doc if isinstance(doc, dict) else {}), str(payload.get("sha") or "")

    async def put_file(self, data: dict, *, sha: str | None, message: str,
                       path: str | None = None) -> str:
        import json

        body = json.dumps(data, indent=1, sort_keys=True).encode()
        if len(body) > MAX_DOC_BYTES:
            raise SyncError(f"Workspace document is {len(body)} bytes; GitHub's contents API accepts "
                            f"at most {MAX_DOC_BYTES} here. Split model references into artifacts.")
        payload: dict[str, Any] = {"message": message[:200] or "Update KaggleUI workspace",
                                   "content": base64.b64encode(body).decode(), "branch": self.branch}
        if sha:
            payload["sha"] = sha
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.put(self._url(path or self.path), headers=self._headers(), json=payload)
        if response.status_code == 409:
            raise SyncConflict("GitHub rejected the write because the file changed; refresh and compare")
        if response.status_code in (401, 403):
            raise SyncError("GitHub rejected the workspace token (contents:write is required)")
        if response.status_code not in (200, 201):
            raise SyncError(f"GitHub returned {response.status_code} while saving the workspace")
        return str(response.json().get("content", {}).get("sha") or "")

    async def history(self, limit: int = 20) -> list[dict[str, Any]]:
        params = {"path": self.path, "sha": self.branch, "per_page": str(max(1, min(limit, 100)))}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{API}/repos/{self.repo}/commits", headers=self._headers(),
                                        params=params)
        if response.status_code == 404:
            return []
        if response.status_code != 200:
            raise SyncError(f"GitHub returned {response.status_code} while reading history")
        out = []
        for item in response.json():
            commit = item.get("commit", {})
            out.append({"sha": item.get("sha"), "message": commit.get("message", ""),
                        "date": commit.get("committer", {}).get("date"), "url": item.get("html_url")})
        return out

def _client(state: AppState, token: str | None = None) -> GitHubWorkspace:
    cfg = state.settings.workspace
    if not cfg.repo.strip():
        raise SyncError("Workspace sync is not configured; set a repository in Workspace settings")
    secret = token or os.environ.get(cfg.token_secret or "GITHUB_TOKEN")
    return GitHubWorkspace(cfg.repo, cfg.branch, cfg.path, secret or "")

def _state(state: AppState) -> dict[str, Any]:
    return state.workbench.workspace_state()

def _save(state: AppState, **changes: Any) -> dict[str, Any]:
    data = {**_state(state), **changes}
    state.workbench.save_workspace_state(data)
    return data

def _config_key(state: AppState) -> str:
    cfg = state.settings.workspace
    return f"{cfg.repo}|{cfg.branch}|{cfg.path}"

async def status(state: AppState) -> dict[str, Any]:
    cfg = state.settings.workspace
    local = export_document(state)
    local_hash = document_hash(local)
    info = _state(state)
    out: dict[str, Any] = {
        "configured": bool(cfg.enabled and cfg.repo.strip()),
        "repo": cfg.repo, "branch": cfg.branch, "path": cfg.path,
        "token_secret": cfg.token_secret, "token_set": bool(os.environ.get(cfg.token_secret or "GITHUB_TOKEN")),
        "workspace_id": local.meta.workspace_id, "name": local.meta.name,
        "local_hash": local_hash, "local_changed": bool(info.get("last_backup_hash")) and
        local_hash != info.get("last_backup_hash"),
        "last_backup_at": info.get("last_backup_at"), "last_restore_at": info.get("last_restore_at"),
        "remote_sha": info.get("remote_sha"), "remote_available": False, "remote_changed": False,
        "conflict": False, "error": None,
    }
    if not out["configured"] or not out["token_set"]:
        return out
    if info.get("config_key") and info.get("config_key") != _config_key(state):
        out["error"] = "Workspace repository settings changed; fetch a preview before overwriting templates"
        return out
    try:
        remote = await _client(state).get_file()
    except (SyncError, httpx.HTTPError) as exc:
        out["error"] = str(exc)
        return out
    out["remote_available"] = remote is not None
    if remote:
        doc, sha = remote
        out["remote_sha"] = sha
        out["remote_changed"] = bool(info.get("remote_sha")) and sha != info.get("remote_sha")
        out["conflict"] = bool(out["remote_changed"] and out["local_changed"])
        try:
            incoming = validate_document(doc)
            out["remote_workspace_id"] = incoming.meta.workspace_id
            out["remote_name"] = incoming.meta.name
        except WorkspaceError as exc:
            out["error"] = str(exc)
    return out

async def preview(state: AppState, doc: dict | WorkspaceDocument | None = None) -> dict[str, Any]:
    remote = doc
    sha: str | None = None
    if remote is None:
        got = await _client(state).get_file()
        if got is None:
            return {"exists": False, "diff": None}
        remote, sha = got
    incoming = validate_document(remote)
    current = export_document(state)
    return {"exists": True, "sha": sha, "incoming": incoming.model_dump(),
            "diff": diff_documents(current, incoming),
            "document_hash": document_hash(incoming)}

async def backup(state: AppState, *, force: bool = False, message: str | None = None,
                 fork: bool = False, confirm: bool = False) -> dict[str, Any]:
    cfg = state.settings.workspace
    if not cfg.enabled or not cfg.repo.strip():
        raise SyncError("Workspace sync is off; enable it in Workspace settings")
    client = _client(state)
    current = export_document(state)
    current_hash = document_hash(current)
    info = _state(state)
    remote = await client.get_file()
    remote_sha = remote[1] if remote else None
    remote_hash = document_hash(remote[0]) if remote else None
    local_changed = bool(info.get("last_backup_hash")) and current_hash != info.get("last_backup_hash")
    remote_changed = bool(info.get("remote_sha")) and remote_sha != info.get("remote_sha")

    if remote_changed and not force:
        raise SyncConflict("The remote workspace changed since this install last read it; preview and restore "
                           "or confirm an explicit overwrite", local_sha=info.get("remote_sha"),
                           remote_sha=remote_sha, local_changed=local_changed, remote_changed=True)
    if remote_hash == current_hash and not force:
        _save(state, remote_sha=remote_sha, last_backup_hash=current_hash, last_backup_at=time.time(),
              config_key=_config_key(state))
        return {"status": "unchanged", "sha": remote_sha, "document_hash": current_hash}
    if local_changed and not confirm and not force:
        raise SyncConflict("This install has unsaved workspace changes; confirm to publish them",
                           local_sha=info.get("remote_sha"), remote_sha=remote_sha,
                           local_changed=True, remote_changed=remote_changed)

    path = None
    if fork:
        fork_path = f"{_safe_path(cfg.path)}/forks/{current.meta.workspace_id}.json"
        path = fork_path
        fork_doc = current.model_copy(deep=True)
        fork_doc.meta.workspace_id = fork_doc.meta.workspace_id or current.meta.workspace_id
    else:
        fork_doc = current
    sha = await client.put_file(fork_doc.model_dump(), sha=remote_sha if not fork else None,
                                message=message or f"KaggleUI workspace backup ({current.meta.name})",
                                path=path)
    stamp = time.time()
    _save(state, remote_sha=sha, last_backup_hash=current_hash, last_backup_at=stamp,
          config_key=_config_key(state), last_action="backup",
          history=[*list(info.get("history") or [])[-19:], {"at": stamp, "action": "backup", "sha": sha}])
    state.bus.publish("workspace", {"action": "backup", "sha": sha, "document_hash": current_hash})
    return {"status": "backed_up", "sha": sha, "document_hash": current_hash, "path": path or client.path,
            "bytes": len(export_document(state).model_dump_json())}

async def restore(state: AppState, *, confirm: bool = False, resolution: str = "replace",
                  sha: str | None = None) -> dict[str, Any]:
    client = _client(state)
    got = await client.get_file(ref=sha) if sha else await client.get_file()
    if got is None:
        raise SyncError("No workspace snapshot exists at that location")
    remote_doc, remote_sha = got
    incoming = validate_document(remote_doc)
    current = export_document(state)
    diff = diff_documents(current, incoming)
    info = _state(state)
    local_hash = document_hash(current)
    local_changed = bool(info.get("last_backup_hash")) and local_hash != info.get("last_backup_hash")
    remote_changed = bool(info.get("remote_sha")) and remote_sha != info.get("remote_sha")
    if (local_changed or remote_changed) and not confirm:
        raise SyncConflict("Local and remote workspaces differ; preview first and pass confirm=true",
                           local_sha=info.get("remote_sha"), remote_sha=remote_sha,
                           local_changed=local_changed, remote_changed=remote_changed)
    if resolution == "keep_local":
        return {"status": "kept_local", "diff": diff, "remote_sha": remote_sha}
    if resolution not in {"replace", "fork"}:
        raise SyncError("resolution must be replace, keep_local or fork")
    if resolution == "fork":
        # Keep the current local document under a new local identity before applying the remote one.
        forked = current.model_copy(deep=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        from .workspace import local_backup

        local_backup(state, forked, reason=f"fork-{stamp}")
    applied = apply_document(state, incoming, replace=True, replace_connections=True)
    stamp = time.time()
    _save(state, remote_sha=remote_sha, last_backup_hash=document_hash(incoming),
          last_restore_at=stamp, config_key=_config_key(state), last_action="restore",
          history=[*list(info.get("history") or [])[-19:], {"at": stamp, "action": "restore", "sha": remote_sha}])
    state.bus.publish("workspace", {"action": "restore", "sha": remote_sha, "diff": diff})
    return {"status": "restored", "remote_sha": remote_sha, "diff": diff, **applied}

async def history(state: AppState, limit: int = 20) -> dict[str, Any]:
    return {"commits": await _client(state).history(limit)}

async def discovery_document(state: AppState) -> dict[str, Any]:
    url = state.main_url()
    if "?" in url:
        url = url.split("?", 1)[0]
    info = _state(state)
    return {
        "schema_version": 1, "kind": "kaggleui-instance", "instance_id": info.get("instance_id") or "",
        "workspace_id": info.get("workspace_id"), "name": info.get("name") or "Default",
        "version": __version__, "url": url, "started_at": state.started_at, "updated_at": time.time(),
        "metadata": {"mock": state.mock, "fake_hw": state.fake_hw, "host": state.host.name,
                     "deployments": len(state.manager.list())},
    }

async def publish_discovery(state: AppState) -> dict[str, Any] | None:
    cfg = state.settings.workspace
    if not cfg.publish_discovery or not cfg.enabled or not cfg.repo.strip():
        return None
    client = _client(state)
    info = _state(state)
    instance_id = info.get("instance_id") or state.paths.home.name or "local"
    doc = await discovery_document(state)
    doc["instance_id"] = instance_id
    path = f"{_safe_path(cfg.path)}/instances/{instance_id}.json"
    existing = await client.get_file(path=path)
    sha = await client.put_file(doc, sha=existing[1] if existing else None,
                                message="KaggleUI instance discovery", path=path)
    _save(state, instance_id=instance_id, discovery_sha=sha)
    return {"path": path, "sha": sha}


def schedule_auto_backup(state: AppState, *, delay: float = 1.5) -> None:
    """Coalesce workspace changes into one background commit.

    Deploying a set touches several definitions in a row; committing each one would spam the
    repository and race itself. A remote change is still a conflict, never an overwrite.
    """
    pending = getattr(state, "auto_backup_task", None)
    if pending is not None and not pending.done():
        pending.cancel()
    cfg = state.settings.workspace
    if not (cfg.enabled and cfg.auto_backup and cfg.repo.strip()):
        return

    async def run() -> None:
        await asyncio.sleep(delay)
        try:
            result = await backup(state, confirm=True, message="automatic workspace backup")
        except Exception as exc:  # Network and conflict failures are reported, never raised into a route.
            state.bus.publish("workspace", {"action": "auto_backup_failed", "error": str(exc)})
            return
        state.bus.publish("workspace", {"action": "auto_backup", "result": result})

    state.auto_backup_task = asyncio.create_task(run())

async def restore_on_launch(state: AppState) -> dict[str, Any] | None:
    """Fetch and apply the remote workspace before the link is published.

    A local document with changes since the last backup is never overwritten automatically; the
    operator gets a conflict event instead. Deployment definitions are applied, but nothing is
    started unless ``auto_start_set`` is enabled, and definition imports strip remote-code options.
    """
    cfg = state.settings.workspace
    if not (cfg.enabled and cfg.restore_on_launch and cfg.repo.strip()):
        return None
    try:
        remote = await _client(state).get_file()
        if remote is None:
            state.bus.publish("workspace", {"action": "launch_restore_skipped", "reason": "remote is empty"})
            return None
        incoming = validate_document(remote[0])
        current = export_document(state)
        current_hash = document_hash(current)
        info = _state(state)
        local_dirty = bool(info.get("last_backup_hash")) and current_hash != info.get("last_backup_hash")
        if local_dirty and document_hash(incoming) != current_hash:
            state.bus.publish("workspace", {
                "action": "launch_restore_conflict",
                "reason": "local workspace has changes since the last backup; open Workspace to compare"})
            print("[kaggleui] workspace restore skipped: local changes since the last backup", flush=True)
            return None
        result = apply_document(state, incoming, replace=True, replace_connections=True)
        stamp = time.time()
        _save(state, remote_sha=remote[1], last_backup_hash=document_hash(incoming),
              last_restore_at=stamp, config_key=_config_key(state), last_action="launch_restore")
        started = 0
        if state.settings.workspace.auto_start_set and state.workbench.active_set:
            active = state.workbench.deployment_set(state.workbench.active_set)
            if active:
                for spec in active.deployments:
                    existing = {d.served_name for d in state.manager.deployments.values()
                                if d.status in {"queued", "installing", "downloading", "loading", "ready"}}
                    from ..deploy.manager import default_name

                    if default_name(spec) in existing:
                        continue
                    try:
                        state.manager.create(spec)
                        started += 1
                    except Exception as exc:
                        result["warnings"].append(f"could not start {default_name(spec)}: {exc}")
        result.update({"status": "restored_on_launch", "started": started, "remote_sha": remote[1]})
        state.bus.publish("workspace", {"action": "launch_restore", "started": started,
                                        "workspace_id": incoming.meta.workspace_id})
        print(f"[kaggleui] workspace restored from GitHub ({incoming.meta.name}, "
              f"{len(incoming.deployments)} definitions, {started} started)", flush=True)
        return result
    except (SyncError, WorkspaceError, httpx.HTTPError) as exc:
        state.bus.publish("workspace", {"action": "launch_restore_failed", "error": str(exc)})
        print(f"[kaggleui] workspace restore failed: {exc}", flush=True)
        return None

