from __future__ import annotations

import asyncio
import base64
import json

import pytest

from kaggleui.schemas import Settings, WorkspaceMeta
from kaggleui.server.state import AppState
from kaggleui.workbench import sync, workspace


class FakeGitHub:
    def __init__(self, document, sha="sha1"):
        self.remote = (document, sha)
        self.path = "workspaces/default/workspace.json"
        self.puts = []

    async def get_file(self, path=None, ref=None):
        return self.remote

    async def put_file(self, data, *, sha, message, path=None):
        self.puts.append({"data": data, "sha": sha, "message": message, "path": path})
        self.remote = (data, "new-sha")
        return "new-sha"


@pytest.fixture
def state(paths):
    settings = Settings()
    settings.exposure.provider = "none"
    settings.workspace.enabled = True
    settings.workspace.repo = "owner/repo"
    settings.workspace.restore_on_launch = True
    return AppState(paths=paths, settings=settings, port=8999, mock=True, fake_hw="t4x2")


# monkeypatch is used through a fixture below; keep a factory that takes pytest's monkeypatch.
@pytest.fixture
def connect(monkeypatch):
    def make(state, fake):
        monkeypatch.setattr(sync, "_client", lambda _state, token=None: fake)
        return fake

    return make


def test_unchanged_backup_does_not_commit(state, connect):
    current = workspace.export_document(state)
    fake = connect(state, FakeGitHub(current.model_dump(), "abc"))
    info = state.workbench.workspace_state()
    state.workbench.save_workspace_state({**info, "remote_sha": "abc",
                                          "last_backup_hash": workspace.document_hash(current)})
    result = __import__("asyncio").run(sync.backup(state))
    assert result["status"] == "unchanged"
    assert fake.puts == []


def test_remote_change_is_a_conflict(state, connect):
    current = workspace.export_document(state)
    incoming = current.model_copy(update={"meta": WorkspaceMeta(workspace_id="other", name="Other")})
    fake = connect(state, FakeGitHub(incoming.model_dump(), "new"))
    info = state.workbench.workspace_state()
    state.workbench.save_workspace_state({**info, "remote_sha": "old",
                                          "last_backup_hash": workspace.document_hash(current)})
    with pytest.raises(sync.SyncConflict):
        __import__("asyncio").run(sync.backup(state))
    assert fake.puts == []


def test_launch_restore_skips_on_local_dirt(state, connect):
    current = workspace.export_document(state)
    incoming = current.model_copy(update={"meta": WorkspaceMeta(workspace_id="other", name="Other")})
    connect(state, FakeGitHub(incoming.model_dump(), "remote"))
    info = state.workbench.workspace_state()
    state.workbench.save_workspace_state({**info, "last_backup_hash": "different"})
    queue = state.bus.subscribe()
    result = __import__("asyncio").run(sync.restore_on_launch(state))
    assert result is None
    event = queue.get_nowait()["data"]
    assert event["action"] == "launch_restore_conflict"
    assert state.workbench.workspace_state()["workspace_id"] != "other"


def test_launch_restore_applies_and_records_sha(state, connect):
    incoming = workspace.export_document(state).model_copy(
        update={"meta": WorkspaceMeta(workspace_id="portable", name="Team")})
    connect(state, FakeGitHub(incoming.model_dump(), "remote"))
    queue = state.bus.subscribe()
    result = __import__("asyncio").run(sync.restore_on_launch(state))
    assert result["status"] == "restored_on_launch"
    assert state.workbench.workspace_state()["workspace_id"] == "portable"
    actions = []
    while not queue.empty():
        actions.append(queue.get_nowait()["data"]["action"])
    assert "launch_restore" in actions


async def test_auto_backup_coalesces_and_commits(state, connect):
    current = workspace.export_document(state)
    fake = connect(state, FakeGitHub(current.model_dump(), "abc"))
    info = state.workbench.workspace_state()
    state.workbench.save_workspace_state({**info, "remote_sha": "abc",
                                          "last_backup_hash": workspace.document_hash(current)})
    state.settings.workspace.auto_backup = True
    state.settings.access.password_enabled = True
    sync.schedule_auto_backup(state, delay=0)
    assert state.auto_backup_task is not None
    await state.auto_backup_task
    assert len(fake.puts) == 1
    assert fake.puts[0]["sha"] == "abc"

async def test_disabling_auto_backup_cancels_the_pending_commit(state, connect):
    current = workspace.export_document(state)
    fake = connect(state, FakeGitHub(current.model_dump(), "abc"))
    state.settings.workspace.auto_backup = True
    sync.schedule_auto_backup(state, delay=30)
    pending = state.auto_backup_task
    assert pending is not None
    state.settings.workspace.auto_backup = False
    sync.schedule_auto_backup(state, delay=30)
    await asyncio.sleep(0)
    assert pending.cancelled()
    assert fake.puts == []


class FakeAPI:
    """Stands in for GitHub's REST API; records every call so tests can assert the wiring."""

    def __init__(self, routes):
        self.routes = routes  # list of (method, path_prefix, status, payload)
        self.calls = []

    async def __call__(self, token, method, path, *, body=None, accept=None, timeout=None):
        self.calls.append({"method": method, "path": path, "body": body})
        for want_method, prefix, status, payload in self.routes:
            if method == want_method and path.startswith(prefix):
                return status, payload, "repo"
        raise AssertionError(f"unexpected GitHub call {method} {path}")


def test_wrapped_base64_file_content_decodes(monkeypatch):
    """GitHub wraps base64 at 60 columns; validate=True used to reject its own response."""
    doc = {"meta": {"workspace_id": "abc", "name": "Unit"}, "deployments": []}
    encoded = base64.b64encode(json.dumps(doc).encode()).decode()
    wrapped = "\n".join(encoded[i:i + 60] for i in range(0, len(encoded), 60)) + "\n"
    api = FakeAPI([("GET", "/repos/owner/repo/contents/", 200,
                    {"type": "file", "sha": "abc123", "encoding": "base64", "content": wrapped})])
    monkeypatch.setattr(sync, "_github_request", api)
    client = sync.GitHubWorkspace("owner/repo", "main", "workspaces/default", "t")
    payload, sha = asyncio.run(client.get_file())
    assert payload == doc
    assert sha == "abc123"


def test_save_token_verifies_and_stores(state, monkeypatch):
    stored = {}
    monkeypatch.setattr(sync.secretstore, "set_secret", lambda name, value: stored.update({name: value}))
    api = FakeAPI([("GET", "/user", 200, {"login": "octocat"})])
    monkeypatch.setattr(sync, "_github_request", api)
    result = asyncio.run(sync.save_token(state, "  ghp_example  "))
    assert result["login"] == "octocat"
    assert stored == {"GITHUB_TOKEN": "ghp_example"}
    assert result["warning"] is None


def test_save_token_rejects_bad_credentials(state, monkeypatch):
    monkeypatch.setattr(sync, "_github_request",
                        FakeAPI([("GET", "/user", 401, {"message": "Bad credentials"})]))
    with pytest.raises(sync.SyncError, match="rejected"):
        asyncio.run(sync.save_token(state, "nope"))


def test_link_repository_creates_private_repo_with_first_commit(state, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    api = FakeAPI([
        ("GET", "/user", 200, {"login": "octocat"}),
        ("POST", "/user/repos", 201, {"full_name": "octocat/kaggleui-workspace",
                                      "default_branch": "main"}),
    ])
    monkeypatch.setattr(sync, "_github_request", api)
    result = asyncio.run(sync.link_repository(state, mode="create", name="kaggleui-workspace"))
    assert result["created"] is True
    assert result["repo"] == "octocat/kaggleui-workspace"
    body = [c for c in api.calls if c["method"] == "POST"][0]["body"]
    assert body["private"] is True
    assert body["auto_init"] is True
    assert state.settings.workspace.repo == "octocat/kaggleui-workspace"
    assert state.settings.workspace.branch == "main"
    assert state.settings.workspace.enabled is True


def test_link_repository_reports_missing_existing_repo(state, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr(sync, "_github_request", FakeAPI([
        ("GET", "/user", 200, {"login": "octocat"}),
        ("GET", "/repos/octocat/nope", 404, {"message": "Not Found"}),
    ]))
    with pytest.raises(sync.SyncError, match="create a private repo"):
        asyncio.run(sync.link_repository(state, mode="existing", repo="octocat/nope"))


def test_put_file_falls_back_to_the_default_branch(monkeypatch):
    """A repo whose default branch is master must not fail because settings say main."""
    calls = []

    async def api(token, method, path, *, body=None, accept=None, timeout=None):
        calls.append((method, path, body))
        if method == "PUT" and (body or {}).get("branch") == "main":
            return 422, {"message": "Branch main not found"}, ""
        if method == "GET" and path == "/repos/owner/repo":
            return 200, {"default_branch": "master"}, ""
        if method == "PUT":
            return 201, {"content": {"sha": "new"}}, ""
        raise AssertionError(f"unexpected {method} {path}")

    monkeypatch.setattr(sync, "_github_request", api)
    client = sync.GitHubWorkspace("owner/repo", "main", "workspaces/default", "t")
    sha = asyncio.run(client.put_file({"meta": {}}, sha=None, message="x"))
    assert sha == "new"
    assert client.branch == "master"
    assert calls[-1][2]["branch"] == "master"


def test_put_file_bootstraps_an_empty_repository(monkeypatch):
    """A repo created without a first commit must still accept the first workspace document."""
    calls = []

    async def api(token, method, path, *, body=None, accept=None, timeout=None):
        calls.append((method, path, dict(body or {})))
        if method == "PUT" and "branch" in (body or {}):
            return 409, {"message": "Git Repository is empty."}, ""
        if method == "PUT":
            return 201, {"content": {"sha": "first"}}, ""
        raise AssertionError(f"unexpected {method} {path}")

    monkeypatch.setattr(sync, "_github_request", api)
    client = sync.GitHubWorkspace("owner/repo", "main", "workspaces/default", "t")
    sha = asyncio.run(client.put_file({"meta": {}}, sha="stale", message="init"))
    assert sha == "first"
    assert "branch" not in calls[-1][2]
    assert "sha" not in calls[-1][2]


def test_repointed_workspace_ignores_the_old_location_baseline(state, connect):
    """Changing repo/path must not read the previous location's sha as a remote edit."""
    current = workspace.export_document(state)
    other = current.model_dump()
    other["meta"]["name"] = "Elsewhere"
    fake = connect(state, FakeGitHub(other, "other-sha"))
    state.workbench.save_workspace_state({"remote_sha": "stale-sha", "last_backup_hash": "stale-hash",
                                          "config_key": "someone/else|main|workspaces/other"})
    state.settings.workspace.path = "workspaces/default"
    result = asyncio.run(sync.backup(state, confirm=True))
    assert result["status"] == "backed_up"
    assert fake.puts and fake.puts[0]["sha"] == "other-sha"


def test_repointed_workspace_still_confirms_before_overwriting(state, connect):
    current = workspace.export_document(state)
    connect(state, FakeGitHub(current.model_dump(), "other-sha"))
    state.workbench.save_workspace_state({"remote_sha": "stale-sha",
                                          "config_key": "someone/else|main|workspaces/other"})
    with pytest.raises(sync.SyncConflict):
        asyncio.run(sync.backup(state))


def test_status_after_a_settings_change_reports_the_new_location(state, connect, monkeypatch):
    current = workspace.export_document(state)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    connect(state, FakeGitHub(current.model_dump(), "sha-new"))
    state.workbench.save_workspace_state({"remote_sha": "stale-sha",
                                          "config_key": "someone/else|main|workspaces/other"})
    state.settings.workspace.path = "workspaces/default"
    out = asyncio.run(sync.status(state))
    assert out["error"] is None
    assert out["remote_available"] is True
    assert out["remote_sha"] == "sha-new"
    assert out["remote_changed"] is False
