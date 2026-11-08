from __future__ import annotations

import pytest

from kaggleui.schemas import (
    DeploymentSet,
    DeploymentSpec,
    Settings,
    StorageConnection,
    WorkspaceDocument,
    WorkspaceMeta,
)
from kaggleui.server.state import AppState
from kaggleui.workbench import workspace


@pytest.fixture
def state(paths):
    settings = Settings()
    settings.exposure.provider = "none"
    settings.workspace.enabled = True
    settings.workspace.repo = "owner/repo"
    return AppState(paths=paths, settings=settings, port=8999, mock=True, fake_hw="t4x2")


def spec(**options) -> DeploymentSpec:
    return DeploymentSpec(modality="text", name="demo",
                          model={"source": "hf", "id": "org/model", "engine": "mock"},
                          options=options)


def document(state, **overrides) -> WorkspaceDocument:
    doc = workspace.export_document(state)
    data = doc.model_dump()
    data.update(overrides)
    return WorkspaceDocument.model_validate(data)


def test_export_strips_runtime_options(state):
    definition = spec(pid=123, port=9000, ctx=8192, trust_remote_code=True)
    state.workbench.put_deployment("d1", definition, "demo", created_at=0)
    doc = workspace.export_document(state)
    options = doc.deployments[0].options
    assert options == {"ctx": 8192, "trust_remote_code": True}
    assert {"pid", "port"}.isdisjoint(options)
    assert "trust_remote_code" in options  # reviewed on import, where it is removed with a warning


def test_validate_rejects_newer_schema_and_dangling_set(state):
    raw = workspace.export_document(state).model_dump()
    raw["meta"]["schema_version"] = 2
    with pytest.raises(workspace.WorkspaceError, match="newer"):
        workspace.validate_document(raw)

    raw = workspace.export_document(state).model_dump()
    raw["active_set"] = "missing"
    with pytest.raises(workspace.WorkspaceError, match="active_set"):
        workspace.validate_document(raw)


def test_validate_rejects_duplicate_connections(state):
    raw = workspace.export_document(state).model_dump()
    raw["storage_connections"] = [
        StorageConnection(name="same", kind="local", config={"root": "/a"}).model_dump(),
        StorageConnection(name="same", kind="local", config={"root": "/b"}).model_dump(),
    ]
    with pytest.raises(workspace.WorkspaceError, match="duplicate"):
        workspace.validate_document(raw)


def test_apply_backs_up_and_strips_remote_code(state):
    incoming = document(state, deployments=[spec(trust_remote_code=True).model_dump()],
                        deployment_sets=[DeploymentSet(id="set1", name="Night",
                                                       deployments=[spec()]).model_dump()],
                        active_set="set1")
    result = workspace.apply_document(state, incoming)
    assert any("trust_remote_code" in warning for warning in result["warnings"])
    records = state.workbench.deployment_records()
    assert len(records) == 1
    assert "trust_remote_code" not in DeploymentSpec.model_validate(records[0].spec).options
    assert state.workbench.active_set == "set1"
    backups = list((state.paths.home / "workbench" / "backups").glob("*.json"))
    assert len(backups) == 1


def test_positive_copy_keeps_identity_and_does_not_start(state):
    incoming = document(state, meta=WorkspaceMeta(workspace_id="portable", name="Team").model_dump())
    workspace.apply_document(state, incoming)
    info = state.workbench.workspace_state()
    assert info["workspace_id"] == "portable"
    assert info["name"] == "Team"
    assert state.manager.list() == []

