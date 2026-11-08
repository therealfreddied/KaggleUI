from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from kaggleui.kui import cli
from kaggleui.kui.client import KuiError, ManagementClient
from kaggleui.kui.config import ProfileError, ProfileStore
from kaggleui.kui.kaggle_adapter import KaggleNotebooks


def test_profile_store_round_trip(tmp_path):
    store = ProfileStore(tmp_path / "config.json")
    store.set_section("instances", "lab", {"url": "http://127.0.0.1:8890", "token": "secret"})
    store.set_section("workspaces", "team", {"repo": "owner/repo", "branch": "main"})
    store.set_section("credentials", "me", {"username": "alice", "key": "k"})
    store.add_profile("work", instance="lab", workspace="team", credential="me")
    store.set_active("work")
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600

    reloaded = ProfileStore(store.path)
    assert reloaded.active_name() == "work"
    resolved = reloaded.resolve()
    assert resolved["instance"]["url"] == "http://127.0.0.1:8890"
    assert resolved["workspace"]["repo"] == "owner/repo"
    assert resolved["credential"]["username"] == "alice"
    public = reloaded.public()
    assert "secret" not in json.dumps(public)
    assert "token" not in public["instances"]["lab"]
    assert public["credentials"]["me"]["key"] == "***"


def test_profile_unknown_reference_is_an_error(tmp_path):
    store = ProfileStore(tmp_path / "config.json")
    with pytest.raises(ProfileError, match="no instance"):
        store.add_profile("work", instance="missing")


def test_normalize_hoists_global_flags():
    argv = cli._normalize(["status", "--json", "--profile", "work", "--url", "http://x"])
    assert argv[-1] == "status"
    assert argv[argv.index("--profile") + 1] == "work"
    assert argv[argv.index("--url") + 1] == "http://x"
    assert "--json" in argv


def test_client_maps_status_codes():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bad"):
            return httpx.Response(401, json={"detail": "no"})
        if request.url.path.endswith("/forbidden"):
            return httpx.Response(403, json={"detail": "scope"})
        if request.url.path.endswith("/conflict"):
            return httpx.Response(409, json={"detail": "changed"})
        return httpx.Response(200, json={"ok": True})

    client = ManagementClient("http://127.0.0.1:1", "key", transport=httpx.MockTransport(handler))
    with pytest.raises(KuiError) as bad:
        client.get("/bad")
    assert bad.value.code == 5 and bad.value.status == 401
    with pytest.raises(KuiError) as forbidden:
        client.get("/forbidden")
    assert forbidden.value.code == 5
    with pytest.raises(KuiError) as conflict:
        client.get("/conflict")
    assert conflict.value.code == 7


def test_client_without_key_stays_local():
    with pytest.raises(KuiError) as exc:
        ManagementClient("http://127.0.0.1:8890", "")
    assert exc.value.code == 3


def fake_kaggle(tmp_path: Path, *, body: str = "") -> Path:
    binary = tmp_path / "kaggle"
    binary.write_text(f"""#!{sys.executable}
import json, pathlib, sys
args = sys.argv[1:]
pathlib.Path({str(tmp_path / 'args.json')!r}).write_text(json.dumps(args))
if args[:2] == ['kernels', 'push']:
    folder = pathlib.Path(args[args.index('-p') + 1])
    pathlib.Path({str(tmp_path / 'metadata.json')!r}).write_text((folder / 'kernel-metadata.json').read_text())
    print('pushed')
elif args[:2] == ['kernels', 'status']:
    print({body!r})
else:
    print('other')
""")
    binary.chmod(0o755)
    return binary


def test_kaggle_deploy_uses_verified_flags(tmp_path):
    source = tmp_path / "notebook.ipynb"
    source.write_text("{}")
    adapter = KaggleNotebooks(binary=str(fake_kaggle(tmp_path)), credential={"username": "alice", "key": "k"})
    result = adapter.deploy(slug="owner/kernel", source=str(source), accelerator="NvidiaTeslaT4",
                            timeout_s=3600, internet=True)
    args = json.loads((tmp_path / "args.json").read_text())
    assert result["url"] == "https://www.kaggle.com/code/owner/kernel"
    assert args[:2] == ["kernels", "push"]
    assert "--accelerator" in args and "NvidiaTeslaT4" in args
    assert "--timeout" in args and "3600" in args
    assert "--no-run" not in args
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["id"] == "owner/kernel"
    assert metadata["enable_gpu"] is True and metadata["enable_tpu"] is False


def test_kaggle_stop_reports_unavailable(tmp_path):
    adapter = KaggleNotebooks(binary=str(fake_kaggle(tmp_path, body="KernelWorkerStatus.RUNNING")))
    result = adapter.stop("owner/kernel")
    assert result["stopped"] is False
    assert result["capability"] == "unavailable"
    assert result["state"] == "running"



def test_explicit_url_and_token_need_no_profile(tmp_path):
    store = ProfileStore(tmp_path / "config.json")
    client = cli._client(SimpleNamespace(profile=None, url="http://127.0.0.1:8921", token="kui_mgmt_x"), store)
    assert client.base == "http://127.0.0.1:8921"
    assert client._token == "kui_mgmt_x"

    with pytest.raises(KuiError, match="no management key"):
        cli._client(SimpleNamespace(profile=None, url="http://127.0.0.1:1", token=None), store)

    with pytest.raises(ProfileError):
        cli._client(SimpleNamespace(profile=None, url=None, token=None), store)

