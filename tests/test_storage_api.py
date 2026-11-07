"""Storage routes: connection CRUD, artifact export job, restore, redaction of config values."""

import time
from pathlib import Path

from fastapi.testclient import TestClient

from kaggleui.schemas import ArtifactManifest, Settings
from kaggleui.server.app import create_app
from kaggleui.server.state import AppState

ADMIN = {"Authorization": "Bearer tok"}


def _client(paths, tmp_path) -> TestClient:
    settings = Settings()
    settings.exposure.provider = "none"
    state = AppState(paths=paths, settings=settings, port=8995, mock=True, fake_hw="t4x2")
    state.auth.admin_token = "tok"
    return TestClient(create_app(state)), state


def _seed_artifact(state, tmp_path: Path) -> ArtifactManifest:
    src = tmp_path / "seed"
    src.mkdir(exist_ok=True)
    model = src / "model-Q4_K_M.gguf"
    model.write_bytes(b"gguf" * 300)
    card = src / "README.md"
    card.write_text("# demo\n")
    return state.artifacts.add(ArtifactManifest(artifact_id="demo", version="v1", kind="model"),
                               [model, card])


def _wait(client, job_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/_kui/api/jobs/{job_id}", headers=ADMIN).json()
        if body["state"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_connection_crud_redacts_config_values(paths, tmp_path):
    client, state = _client(paths, tmp_path)
    with client:
        created = client.post("/_kui/api/workbench/storage", headers=ADMIN, json={
            "name": "drive", "kind": "rclone", "config": {"remote": "gdrive:kaggleui", "token": "abc"}})
        assert created.status_code == 200, created.text
        assert created.json()["config"]["token"] == "***"

        listed = client.get("/_kui/api/workbench/storage", headers=ADMIN).json()["connections"]
        assert listed == [created.json()]

        tested = client.post("/_kui/api/workbench/storage/drive/test", headers=ADMIN).json()
        assert tested["ok"] is False and "rclone" in tested["error"]

        assert client.delete("/_kui/api/workbench/storage/nope", headers=ADMIN).status_code == 409
        assert client.delete("/_kui/api/workbench/storage/drive", headers=ADMIN).json() == {"ok": True}
        assert client.get("/_kui/api/workbench/storage", headers=ADMIN).json()["connections"] == []


def test_export_then_restore_round_trip(paths, tmp_path):
    client, state = _client(paths, tmp_path)
    remote = tmp_path / "remote"
    remote.mkdir()
    _seed_artifact(state, tmp_path)
    with client:
        client.post("/_kui/api/workbench/storage", headers=ADMIN, json={
            "name": "disk", "kind": "local", "config": {"root": str(remote)}})

        listed = client.get("/_kui/api/workbench/artifacts", headers=ADMIN).json()["artifacts"]
        assert [a["artifact_id"] for a in listed] == ["demo"]
        assert client.get("/_kui/api/workbench/artifacts/demo/verify").status_code in (404, 405)
        assert client.post("/_kui/api/workbench/artifacts/demo/verify", headers=ADMIN).json()["ok"] is True

        job = client.post("/_kui/api/workbench/artifacts/demo/export", headers=ADMIN,
                          json={"connection": "disk", "target_path": "exports/demo"}).json()
        finished = _wait(client, job["id"])
        assert finished["state"] == "succeeded", finished
        assert (remote / "exports" / "demo" / "model-Q4_K_M.gguf").read_bytes() == \
               (state.artifacts.dir_for("demo", "v1") / "model-Q4_K_M.gguf").read_bytes()
        logs = client.get(f"/_kui/api/jobs/{job['id']}/logs", headers=ADMIN).json()
        assert "exporting demo v1" in logs["lines"][0]

        restored = client.post("/_kui/api/workbench/restore", headers=ADMIN, json={
            "connection": "disk", "remote": "exports/demo", "name": "copy", "kind": "model"}).json()
        done = _wait(client, restored["id"])
        assert done["state"] == "succeeded", done
        copy = state.artifacts.get("copy")
        assert copy is not None and sorted(f.path for f in copy.files) == ["README.md", "model-Q4_K_M.gguf"]
        assert copy.provenance == {"connection": "disk", "remote": "exports/demo"}

        assert client.delete("/_kui/api/workbench/storage/disk", headers=ADMIN).status_code == 409
        assert client.post("/_kui/api/workbench/artifacts/copy/delete", headers=ADMIN,
                           json={"version": copy.version}).status_code == 400


def test_artifact_delete_requires_confirm(paths, tmp_path):
    client, state = _client(paths, tmp_path)
    manifest = _seed_artifact(state, tmp_path)
    with client:
        refused = client.post("/_kui/api/workbench/artifacts/demo/delete", headers=ADMIN,
                              json={"version": "v1"})
        assert refused.status_code == 400 and "confirm" in refused.json()["detail"]
        assert client.post("/_kui/api/workbench/artifacts/demo/delete", headers=ADMIN,
                           json={"version": "v1", "confirm": True}).json() == {"ok": True}
        assert state.artifacts.get("demo") is None
        assert manifest.version == "v1"


def test_rclone_config_upload_is_private_and_bounded(paths, tmp_path):
    client, state = _client(paths, tmp_path)
    with client:
        client.post("/_kui/api/workbench/storage", headers=ADMIN, json={
            "name": "gdrive", "kind": "rclone", "config": {"remote": "gdrive:kaggleui"}})
        uploaded = client.post("/_kui/api/workbench/storage/gdrive/config", headers=ADMIN,
                               files={"file": ("rclone.conf", b"[gdrive]\ntype = drive\n", "text/plain")})
        assert uploaded.status_code == 200, uploaded.text
        conf = paths.home / "rclone.conf"
        assert conf.read_text().startswith("[gdrive]")
        assert conf.stat().st_mode & 0o777 == 0o600
        assert state.workbench.connection("gdrive").config["config_file"] == str(conf)

        too_big = client.post("/_kui/api/workbench/storage/gdrive/config", headers=ADMIN,
                              files={"file": ("rclone.conf", b"x" * (64 * 1024 + 1), "text/plain")})
        assert too_big.status_code == 413
