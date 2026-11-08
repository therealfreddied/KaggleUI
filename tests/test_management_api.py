from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from kaggleui.schemas import Settings, StorageConnection
from kaggleui.server.app import create_app
from kaggleui.server.state import AppState
from kaggleui.workbench import backup


@pytest.fixture
def state(paths):
    settings = Settings()
    settings.exposure.provider = "none"
    return AppState(paths=paths, settings=settings, port=8999, mock=True, fake_hw="t4x2")


@pytest.fixture
def client(state):
    with TestClient(create_app(state)) as c:
        yield c


def admin(state):
    return {"Authorization": f"Bearer {state.auth.admin_token}"}


def test_scoped_key_is_narrower_than_admin(client, state):
    created = client.post("/_kui/api/management", json={"label": "cli", "scopes": ["status", "jobs"]},
                          headers=admin(state))
    assert created.status_code == 200
    token = created.json()["token"]
    assert token.startswith("kui_mgmt_")
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/_kui/api/instance", headers=headers).status_code == 200
    assert client.get("/_kui/api/workbench/jobs", headers=headers).status_code == 200
    forbidden = client.get("/_kui/api/workbench/deployments", headers=headers)
    assert forbidden.status_code == 403
    assert client.get("/_kui/api/instance", headers={"Authorization": "Bearer nope"}).status_code == 401

    listed = client.get("/_kui/api/management", headers=admin(state)).json()
    assert listed["tokens"][0]["prefix"] == token[:16]
    assert "token" not in listed["tokens"][0]

    token_id = listed["tokens"][0]["id"]
    assert client.delete(f"/_kui/api/management/{token_id}", headers=admin(state)).status_code == 200
    assert client.get("/_kui/api/instance", headers=headers).status_code == 401


def test_shutdown_requires_scope_and_confirmation(client, state):
    token = client.post("/_kui/api/management", json={"label": "cli", "scopes": ["status"]},
                        headers=admin(state)).json()["token"]
    assert client.post("/_kui/api/shutdown", json={"confirm": True},
                       headers={"Authorization": f"Bearer {token}"}).status_code == 403
    assert client.post("/_kui/api/shutdown", json={"confirm": False},
                       headers=admin(state)).status_code == 422


def test_model_backup_rejects_local_connection_through_the_job(client, state):
    state.workbench.put_connection(StorageConnection(name="disk", kind="local",
                                                     config={"root": str(state.paths.home / "out")}))
    accepted = client.post("/_kui/api/workbench/model-backups",
                           json={"connection": "disk", "artifact_ids": ["missing"], "confirm": True},
                           headers=admin(state))
    assert accepted.status_code == 200, accepted.text
    job_id = accepted.json()["id"]
    for _ in range(100):
        job = client.get(f"/_kui/api/workbench/jobs/{job_id}", headers=admin(state)).json()
        if job["state"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert job["state"] == "failed"
    assert "local connections" in (job["error"] or "")
    log = client.get(f"/_kui/api/workbench/jobs/{job_id}/log", headers=admin(state))
    assert log.status_code == 200


def test_backup_guard_rejects_read_only(state):
    state.workbench.put_connection(StorageConnection(name="ro", kind="rclone", read_only=True,
                                                     config={"remote": "gdrive:kaggleui"}))
    with pytest.raises(backup.BackupError, match="read-only"):
        __import__("asyncio").run(backup.backup(state, None, connection_name="ro"))

