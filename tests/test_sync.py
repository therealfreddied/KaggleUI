from __future__ import annotations

import asyncio

import pytest

from kaggleui.schemas import Settings, WorkspaceMeta
from kaggleui.server.state import AppState
from kaggleui.workbench import sync, workspace


class FakeGitHub:
    def __init__(self, document, sha="sha1"):
        self.remote = (document, sha)
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
