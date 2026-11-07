"""Remote providers: GitHub is text-only, the HF worker contract, and URL shapes."""

import base64
import json
import sys
import types

import pytest

from kaggleui.storage import remotes
from kaggleui.storage.base import StorageError
from kaggleui.storage.hf_worker import run as hf_run
from kaggleui.storage.remotes import GitHubProvider


def test_github_refuses_binary_and_large_files(tmp_path):
    provider = GitHubProvider("me/catalog", token="t")
    big = tmp_path / "weights.json"
    big.write_bytes(b"x" * (remotes.GITHUB_MAX_BYTES + 1))
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"x")
    with pytest.raises(StorageError, match="1 MiB"):
        provider._check_file(big)
    with pytest.raises(StorageError, match="text metadata"):
        provider._check_file(weights)
    ok = tmp_path / "entry.json"
    ok.write_text("{}")
    assert provider._check_file(ok) == ok


def test_github_upload_puts_base64_into_contents_api(tmp_path, monkeypatch):
    seen: list[tuple[str, str, bytes]] = []

    async def fake(url, *, method="GET", data=None, headers=None, token=None):
        seen.append((url, method, data or b""))
        if method == "GET":
            return 404, b""
        return 201, b"{}"

    monkeypatch.setattr(remotes, "_arequest", fake)
    provider = GitHubProvider("me/catalog", path="kaggleui", branch="dev", token="t")
    src = tmp_path / "x.json"
    src.write_text('{"a": 1}')
    progress: list[float] = []
    import asyncio
    asyncio.run(provider.upload(src, "catalog/x.json", progress=progress.append))

    url, method, body = seen[-1]
    assert method == "PUT"
    assert url.endswith("/repos/me/catalog/contents/kaggleui/catalog/x.json")
    payload = json.loads(body)
    assert base64.b64decode(payload["content"]) == b'{"a": 1}'
    assert payload["branch"] == "dev"
    assert progress == [1.0]


def test_github_requires_token_and_rejects_traversal():
    provider = GitHubProvider("me/catalog")
    import asyncio
    with pytest.raises(StorageError, match="GITHUB_TOKEN"):
        asyncio.run(provider.delete("x.json"))
    with pytest.raises(StorageError):
        provider._url("../escape.json")


def test_hf_worker_test_action_uses_hf_api(monkeypatch):
    calls: list[tuple] = []

    class FakeInfo:
        id = "me/repo"
        private = False

    class FakeApi:
        def __init__(self, token=None):
            calls.append(("token", token))

        def repo_info(self, repo, repo_type="model"):
            calls.append(("repo_info", repo, repo_type))
            return FakeInfo()

    module = types.SimpleNamespace(HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    out = hf_run({"action": "test", "repo": "me/repo", "repo_type": "model", "token": "hf_x"})
    assert out["ok"] and out["repo"] == "me/repo"
    assert ("repo_info", "me/repo", "model") in calls


def test_hf_worker_delete_needs_confirm(monkeypatch):
    class FakeApi:
        def __init__(self, token=None):
            pass

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    with pytest.raises(ValueError, match="confirm"):
        hf_run({"action": "delete", "repo": "me/repo", "remote": "x.gguf"})


def test_hf_provider_needs_the_hub_package(monkeypatch):
    monkeypatch.setattr(remotes.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(StorageError, match="huggingface_hub"):
        remotes.HFProvider("me/repo")

