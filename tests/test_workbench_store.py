"""WorkbenchStore: atomic writes, torn-file tolerance and durable deployment records."""

import json

from kaggleui.schemas import DeploymentSet, DeploymentSpec
from kaggleui.workbench.store import WorkbenchStore


def _spec(**extra) -> DeploymentSpec:
    return DeploymentSpec.model_validate({"modality": "text", "model": {"source": "hf", "id": "org/model-GGUF",
                                                                     "engine": "llamacpp"}, **extra})


def test_files_are_private_and_survive_reload(paths, tmp_path):
    store = WorkbenchStore(tmp_path / "wb")
    store.load()
    saved = store.put_deployment_set(DeploymentSet(name="Night", deployments=[_spec()]))
    store.put_deployment("dep1", _spec(), "qwen", created_at=123.0, runtime={"argv": ["/bin/true"]})

    assert saved.id
    assert (tmp_path / "wb" / "sets.json").stat().st_mode & 0o077 == 0

    again = WorkbenchStore(tmp_path / "wb")
    again.load()
    assert [s.name for s in again.deployment_sets()] == ["Night"]
    record = again.deployment_record("dep1")
    assert record is not None and record.served_name == "qwen" and record.created_at == 123.0
    assert record.runtime["argv"] == ["/bin/true"]


def test_torn_file_does_not_break_boot(tmp_path):
    root = tmp_path / "wb"
    root.mkdir()
    (root / "sets.json").write_text('{"schema_version": 1, "sets": [')
    (root / "deployments.json").write_text(json.dumps({"schema_version": 1, "deployments": [
        {"id": "a", "spec": {}, "served_name": "a"}, {"id": "b", "spec": {}, "served_name": "b"}]}))

    store = WorkbenchStore(root)
    store.load()

    assert store.deployment_sets() == []
    assert {r.id for r in store.deployment_records()} == {"a", "b"}


def test_active_set_cleared_when_deleted(paths, tmp_path):
    store = WorkbenchStore(tmp_path / "wb")
    store.load()
    saved = store.put_deployment_set(DeploymentSet(name="Day", deployments=[]))
    store.mark_active(saved.id)
    store.delete_deployment_set(saved.id)
    assert store.active_set is None
def test_concurrent_job_updates_keep_each_others_fields(tmp_path):
    """Two fields written concurrently must both survive; update() is read-modify-write."""
    import threading
    import time

    from kaggleui.workbench.jobs import JobStore

    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.create("forge.train", {})
    store.update(job.id, state="running")          # request_cancel must not take its queued shortcut
    real_get = store.get

    def slow_get(job_id):
        record = real_get(job_id)
        time.sleep(0.1)  # widen the window an unlocked update() would leave open
        return record

    store.get = slow_get
    errors: list[BaseException] = []

    def progress():
        try:
            store.update(job.id, progress=0.25)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    def cancel():
        try:
            store.request_cancel(job.id)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=progress, name="progress"), threading.Thread(target=cancel, name="cancel")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.get = real_get
    assert not errors
    record = store.get(job.id)
    assert record.progress == 0.25
    assert record.cancel_requested is True
    store.close()
