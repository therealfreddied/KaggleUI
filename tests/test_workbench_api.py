"""Workbench control plane: admin gate, durable jobs, deployment sets."""

import pytest
from fastapi.testclient import TestClient

from kaggleui.schemas import Settings
from kaggleui.server.app import create_app
from kaggleui.server.state import AppState

ADMIN = {"Authorization": "Bearer tok"}


@pytest.fixture
def state(paths):
    s = Settings()
    s.exposure.provider = "none"
    app_state = AppState(paths=paths, settings=s, port=8998, mock=True, fake_hw="t4x2")
    app_state.auth.admin_token = "tok"
    return app_state


@pytest.fixture
def client(state):
    with TestClient(create_app(state)) as c:
        yield c


def test_every_route_needs_the_admin_token(client):
    assert client.get("/_kui/api/jobs").status_code == 401
    assert client.get("/_kui/api/workbench").status_code == 401
    assert client.get("/_kui/api/workbench/deployment-sets").status_code == 401
    assert client.post("/_kui/api/jobs/nope/cancel").status_code == 401


def test_jobs_list_and_missing_job(client):
    assert client.get("/_kui/api/jobs", headers=ADMIN).json() == {"jobs": []}
    assert client.get("/_kui/api/jobs/missing", headers=ADMIN).status_code == 404
    assert client.get("/_kui/api/jobs/missing/logs", headers=ADMIN).status_code == 404


def test_deployment_set_round_trip(client):
    body = {"name": "Night shift", "deployments": [
        {"modality": "text", "model": {"source": "hf", "id": "org/model-GGUF", "engine": "llamacpp"},
         "residency": "on_demand", "idle_timeout_s": 600}]}
    saved = client.post("/_kui/api/workbench/deployment-sets", json=body, headers=ADMIN).json()
    assert saved["id"]
    assert client.get("/_kui/api/workbench/deployment-sets", headers=ADMIN).json()["sets"][0]["name"] == "Night shift"

    assert client.post(f"/_kui/api/workbench/deployment-sets/{saved['id']}/activate", headers=ADMIN).json() == {
        "active": saved["id"]}
    assert client.get("/_kui/api/workbench", headers=ADMIN).json()["active_set"] == saved["id"]
    profile = client.get("/_kui/api/workbench/optimize", headers=ADMIN).json()
    assert profile["optimize"]["enabled"] is False and profile["cache"]["semantic"] is False

    client.delete(f"/_kui/api/workbench/deployment-sets/{saved['id']}", headers=ADMIN)
    assert client.get("/_kui/api/workbench/deployment-sets", headers=ADMIN).json() == {"sets": [], "active": None}


def test_invalid_set_and_placement_are_422(client):
    bad_set = client.post("/_kui/api/workbench/deployment-sets", json={"deployments": []}, headers=ADMIN)
    assert bad_set.status_code == 422
    bad_spec = client.post("/_kui/api/workbench/placement/preview", json={"spec": {}}, headers=ADMIN)
    assert bad_spec.status_code == 422


def test_placement_preview_reports_errors_and_resolution(client):
    spec = {"modality": "text", "model": {"source": "hf", "id": "org/m-GGUF", "engine": "llamacpp"},
            "placement": {"devices": [0, 1], "tensor_split": [0.6, 0.4], "gpu_layers": "auto"}}
    body = client.post("/_kui/api/workbench/placement/preview", json={"spec": spec}, headers=ADMIN).json()
    assert body["ok"] is True
    assert body["resolved"]["gpus"] == [0, 1]
    assert body["resolved"]["gpu_layers"] == 999
    assert body["resolved"]["tensor_split"] == [0.6, 0.4]

    broken = dict(spec, placement={"devices": [0, 1], "tensor_split": [1.0]})
    body = client.post("/_kui/api/workbench/placement/preview", json={"spec": broken}, headers=ADMIN).json()
    assert body["ok"] is False and body["errors"]

def test_convert_and_edit_routes_are_wired(client):
    assert client.get("/_kui/api/workbench/ml/convert", headers=ADMIN).json() == {"jobs": [], "artifacts": []}
    assert client.get("/_kui/api/workbench/ml/edit", headers=ADMIN).json() == {"jobs": []}
    missing = client.post("/_kui/api/workbench/ml/convert", headers=ADMIN,
                          json={"artifact_id": "nope@m1", "quant": "Q4_K_M"})
    assert missing.status_code == 400 and "No artifact" in missing.json()["detail"]

