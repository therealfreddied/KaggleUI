import json
import os
import sys
import types

import pytest
from fastapi.testclient import TestClient

from kaggleui import secretstore
from kaggleui.schemas import Settings
from kaggleui.server.app import create_app
from kaggleui.server.state import AppState


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for n in (*secretstore.SECRET_NAMES, "HUGGING_FACE_HUB_TOKEN", "KAGGLEUI_ATTACHED_SECRETS"):
        monkeypatch.delenv(n, raising=False)
    secretstore._sources.clear()
    yield
    secretstore._sources.clear()


def fake_kaggle(monkeypatch, secrets: dict):
    mod = types.ModuleType("kaggle_secrets")

    class UserSecretsClient:
        def get_secret(self, label):
            if label not in secrets:
                raise KeyError(label)
            return secrets[label]

    mod.UserSecretsClient = UserSecretsClient
    monkeypatch.setitem(sys.modules, "kaggle_secrets", mod)


def test_alias_names_are_found(monkeypatch):
    fake_kaggle(monkeypatch, {"hf_token": "hf_abc", "NGROK_TOKEN": "ng_1"})
    found = secretstore.load_attached_secrets()
    assert found["HF_TOKEN"] and found["NGROK_AUTHTOKEN"]
    assert os.environ["HF_TOKEN"] == "hf_abc" and os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_abc"
    assert secretstore._sources["HF_TOKEN"] == "kaggle"


def test_reload_picks_up_secret_added_later(monkeypatch):
    store: dict = {}
    fake_kaggle(monkeypatch, store)
    assert not secretstore.load_attached_secrets()["HF_TOKEN"]
    store["HF_TOKEN"] = "hf_new"
    assert secretstore.load_attached_secrets(refresh=True)["HF_TOKEN"]


def test_local_store_persists_with_private_permissions(tmp_path):
    secretstore.init_local(tmp_path)
    secretstore.set_secret("NGROK_DOMAIN", "me.ngrok-free.app")
    path = tmp_path / "secrets.json"
    assert json.loads(path.read_text()) == {"NGROK_DOMAIN": "me.ngrok-free.app"}
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    os.environ.pop("NGROK_DOMAIN")
    secretstore._sources.clear()
    secretstore.init_local(tmp_path)  # e.g. after a server restart in the same session
    assert os.environ["NGROK_DOMAIN"] == "me.ngrok-free.app"
    secretstore.clear_secret("NGROK_DOMAIN")
    assert json.loads(path.read_text()) == {}


def test_panel_endpoints_never_echo_values(paths):
    s = Settings()
    s.exposure.provider = "none"
    secretstore.init_local(paths.home)
    state = AppState(paths=paths, settings=s, port=8998, mock=True, fake_hw="t4x2")
    admin = {"Authorization": f"Bearer {state.auth.admin_token}"}
    with TestClient(create_app(state)) as c:
        assert c.put("/_kui/api/secrets/NGROK_AUTHTOKEN", json={"value": "SUPERSECRET123"}).status_code == 401
        r = c.put("/_kui/api/secrets/NGROK_AUTHTOKEN", json={"value": "SUPERSECRET123"}, headers=admin)
        assert r.status_code == 200 and "SUPERSECRET123" not in r.text
        item = next(i for i in r.json()["items"] if i["name"] == "NGROK_AUTHTOKEN")
        assert item == {"name": "NGROK_AUTHTOKEN", "set": True, "source": "kaggleui", "editable": True}
        assert c.get("/_kui/api/overview", headers=admin).json()["secrets"]["NGROK_AUTHTOKEN"] is True
        assert "SUPERSECRET123" not in c.get("/_kui/api/overview", headers=admin).text
        # KaggleUI's own credentials aren't editable here
        assert c.put("/_kui/api/secrets/KAGGLEUI_ADMIN_TOKEN", json={"value": "x"}, headers=admin).status_code == 400
        # launch() hand-off
        r = c.post("/_kui/api/secrets/import", json={"values": {"HF_TOKEN": "hf_x", "KAGGLEUI_PASSWORD": "p"}},
                   headers=admin)
        assert r.json() == {"updated": ["HF_TOKEN"]} and secretstore._sources["HF_TOKEN"] == "kaggle"
        assert c.delete("/_kui/api/secrets/NGROK_AUTHTOKEN", headers=admin).status_code == 200
        assert "NGROK_AUTHTOKEN" not in os.environ


def test_reload_endpoint_and_clear_error_for_unknown_calls(paths, monkeypatch):
    store = {"hf_token": "hf_late"}
    fake_kaggle(monkeypatch, store)
    s = Settings()
    s.exposure.provider = "none"
    state = AppState(paths=paths, settings=s, port=8995, mock=True, fake_hw="t4x2")
    with TestClient(create_app(state)) as c:
        c.post("/_kui/api/login", json={"token": state.auth.admin_token})  # cookie auth, like the browser
        r = c.post("/_kui/api/secrets/reload")
        assert r.status_code == 200 and r.json()["added"] == ["HF_TOKEN"] and "hf_late" not in r.text
        r = c.post("/_kui/api/a-newer-endpoint")
        assert r.status_code == 404 and "older than this page" in r.json()["detail"]


def test_package_exports_launch_and_stop():
    import kaggleui

    assert callable(kaggleui.launch) and callable(kaggleui.stop)


def test_colab_userdata_is_used_when_kaggle_is_absent(monkeypatch):
    colab = types.ModuleType("google.colab")

    class Userdata:
        @staticmethod
        def get(label):
            if label == "HF_TOKEN":
                return "hf_colab"
            raise Exception("SecretNotFoundError")

    colab.userdata = Userdata
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    found = secretstore.load_attached_secrets()
    assert found["HF_TOKEN"]
    assert secretstore.secret_details()["kaggle"] == {"state": "available", "source": "colab", "error": None}
    assert secretstore.attached_sourced() == ["HF_TOKEN"]

