"""Local storage provider: round trip, traversal rejection, overwrite safety, probe write."""

import asyncio

import pytest

from kaggleui.storage import StorageError
from kaggleui.storage.base import RemoteEntry, check_remote
from kaggleui.storage.local import LocalProvider


def _provider(tmp_path):
    root = tmp_path / "remote"
    root.mkdir()
    return LocalProvider(root)


def test_upload_download_round_trip(tmp_path):
    provider = _provider(tmp_path)
    src = tmp_path / "blob.gguf"
    src.write_bytes(b"x" * 5000)

    asyncio.run(provider.upload(src, "models/blob.gguf"))
    assert (tmp_path / "remote" / "models" / "blob.gguf").stat().st_size == 5000
    assert [e.path for e in asyncio.run(provider.list("models"))] == ["models/blob.gguf"]

    out = tmp_path / "back.gguf"
    asyncio.run(provider.download("models/blob.gguf", out))
    assert out.read_bytes() == src.read_bytes()
    assert (tmp_path / "remote" / "models" / "blob.gguf.part").exists() is False


def test_stat_and_delete(tmp_path):
    provider = _provider(tmp_path)
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")
    asyncio.run(provider.upload(src, "a.bin"))

    entry = asyncio.run(provider.stat("a.bin"))
    assert isinstance(entry, RemoteEntry) and entry.size == 3
    assert asyncio.run(provider.stat("missing.bin")) is None

    asyncio.run(provider.delete("a.bin"))
    assert asyncio.run(provider.stat("a.bin")) is None
    with pytest.raises(StorageError):
        asyncio.run(provider.delete("a.bin"))


@pytest.mark.parametrize("remote", ["../escape.bin", "/etc/passwd", "a/../../b", "", "~/.ssh/id_rsa"])
def test_traversal_and_absolute_paths_are_rejected(tmp_path, remote):
    provider = _provider(tmp_path)
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")
    with pytest.raises(StorageError):
        asyncio.run(provider.upload(src, remote))
    with pytest.raises(StorageError):
        check_remote(remote)


def test_missing_root_is_reported_not_created(tmp_path):
    provider = LocalProvider(tmp_path / "nope")
    with pytest.raises(StorageError, match="not a directory"):
        asyncio.run(provider.test())
    assert not (tmp_path / "nope").exists()


def test_probe_needs_a_writable_root(tmp_path):
    provider = _provider(tmp_path)
    report = asyncio.run(provider.test())
    assert report["ok"] is True and report["free_gb"] > 0
    assert not (tmp_path / "remote" / ".kaggleui-probe").exists()


def test_upload_creates_intermediate_directories(tmp_path):
    provider = _provider(tmp_path)
    src = tmp_path / "f.bin"
    src.write_bytes(b"1")
    asyncio.run(provider.upload(src, "deep/nested/f.bin"))
    assert (tmp_path / "remote" / "deep" / "nested" / "f.bin").read_bytes() == b"1"
