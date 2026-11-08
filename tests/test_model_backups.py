from __future__ import annotations

import pytest

from kaggleui.schemas import Settings, StorageConnection
from kaggleui.server.state import AppState
from kaggleui.storage.base import RemoteEntry, StorageError
from kaggleui.workbench import backup


@pytest.fixture
def state(paths):
    settings = Settings()
    settings.exposure.provider = "none"
    return AppState(paths=paths, settings=settings, port=8999, mock=True, fake_hw="t4x2")

async def test_remote_backups_reads_manifest_markers(state, monkeypatch):
    state.workbench.put_connection(StorageConnection(name="r2", kind="rclone", config={"remote": "r2:kui"}))
    seen = {}

    async def fake_list(name, prefix=""):
        seen.update(name=name, prefix=prefix)
        return [
            RemoteEntry(path="kaggleui/manifests/qwen@v1.json", size=10),
            RemoteEntry(path="kaggleui/manifests/notes.txt", size=1),
            RemoteEntry(path="kaggleui/manifests/broken.json", size=2),
        ]

    monkeypatch.setattr(state.storage, "list", fake_list)
    rows = await backup.remote_backups(state, "r2")
    assert seen == {"name": "r2", "prefix": "kaggleui/manifests"}
    assert rows == [{"artifact_id": "qwen", "version": "v1", "connection": "r2", "size": 10,
                     "path": "kaggleui/manifests/qwen@v1.json"}]

async def test_remote_backups_treats_missing_directory_as_empty(state, monkeypatch):
    state.workbench.put_connection(StorageConnection(name="r2", kind="rclone", config={"remote": "r2:kui"}))

    async def missing(name, prefix=""):
        raise StorageError("rclone could not list r2:kui/kaggleui/manifests: directory not found")

    monkeypatch.setattr(state.storage, "list", missing)
    assert await backup.remote_backups(state, "r2") == []

async def test_remote_backups_refuses_local_connections(state):
    state.workbench.put_connection(StorageConnection(name="disk", kind="local", config={"root": "/data"}))
    with pytest.raises(backup.BackupError, match="local connections"):
        await backup.remote_backups(state, "disk")

async def test_backup_and_restore_use_the_job_context(state, monkeypatch, tmp_path):
    from kaggleui.schemas import ArtifactManifest
    from kaggleui.workbench.runner import JobContext

    state.workbench.put_connection(StorageConnection(name="r2", kind="rclone", config={"remote": "r2:kui"}))
    sample = tmp_path / "weights.bin"
    sample.write_bytes(b"weights" * 100)
    state.artifacts.add(ArtifactManifest(artifact_id="demo", version="v1", kind="model"), [sample])

    uploads: dict[str, bytes] = {}

    async def fake_upload(name, local, remote, *, progress=None):
        uploads[remote] = local.read_bytes()

    async def fake_download(name, remote, local, *, progress=None):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(uploads[remote])

    monkeypatch.setattr(state.storage, "upload", fake_upload)
    monkeypatch.setattr(state.storage, "download", fake_download)
    job = state.jobs.create("artifact.backup", {}, logs_path=str(tmp_path / "job.log"))
    result = await backup.backup(state, JobContext(state.jobs, job), connection_name="r2", artifact_ids=["demo"])
    assert result["status"] == "backed_up"
    assert "kaggleui/manifests/demo@v1.json" in uploads
    assert all(not path.startswith("/") for path in uploads)
    assert state.jobs.get(job.id).progress == 1.0

    state.artifacts.remove("demo", "v1", confirm=True)
    restored = await backup.restore(state, JobContext(state.jobs, job), connection_name="r2",
                                    artifact_id="demo", version="v1")
    assert restored["status"] == "restored"
    assert (state.artifacts.dir_for("demo", "v1") / "weights.bin").read_bytes() == b"weights" * 100
