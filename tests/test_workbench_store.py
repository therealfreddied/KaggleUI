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
