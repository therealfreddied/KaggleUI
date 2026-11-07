"""Pinned release assets: digests are full-length, selection matches the host, extraction flattens."""

import hashlib
import io
import tarfile

import pytest

from kaggleui.engines import llamacpp_release as release
from kaggleui.engines.llamacpp_release import (
    CUDART_ASSETS,
    LLAMA_SWAP_ASSETS,
    LLAMACPP_ASSETS,
    ReleaseError,
    asset_for,
    cudart_for,
    extract,
    install_llama,
    swap_asset,
)


def test_every_asset_carries_a_full_digest_and_a_matching_url():
    assets = [*LLAMACPP_ASSETS.values(), *CUDART_ASSETS.values(), *LLAMA_SWAP_ASSETS.values()]
    assert len(assets) >= 10
    for asset in assets:
        assert len(asset.sha256) == 64 and set(asset.sha256) <= set("0123456789abcdef"), asset.name
        assert asset.digest == f"sha256:{asset.sha256}"
        assert asset.url.endswith("/" + asset.name)
        assert asset.url.startswith("https://github.com/")
        assert asset.size > 10_000_000


def test_asset_selection_matches_the_platform():
    assert asset_for(cuda=True, system="Linux", machine="x86_64").name == \
        "llama-b12189-bin-ubuntu-cuda-12.8-x64.tar.gz"
    assert asset_for(cuda=False, system="Linux", machine="aarch64").name == \
        "llama-b12189-bin-ubuntu-arm64.tar.gz"
    # macOS ships no CUDA build, so a CUDA request falls back to the CPU build instead of failing.
    assert asset_for(cuda=True, system="Darwin", machine="arm64").name == \
        "llama-b12189-bin-macos-arm64.tar.gz"
    with pytest.raises(ReleaseError, match="No pinned llama.cpp build"):
        asset_for(cuda=False, system="Linux", machine="s390x")


def test_cudart_only_exists_for_linux():
    assert cudart_for(system="Linux", machine="x86_64").name == \
        "cudart-llama-b12189-bin-ubuntu-cuda-12.8-x64.tar.gz"
    assert cudart_for(system="Linux", machine="aarch64").name == \
        "cudart-llama-b12189-bin-ubuntu-cuda-13.5-arm64.tar.gz"
    assert cudart_for(system="Darwin", machine="arm64") is None


def test_swap_selection_is_linux_only():
    assert swap_asset(system="Linux", machine="x86_64").name == "llama-swap_270_linux_amd64.tar.gz"
    assert swap_asset(system="Linux", machine="aarch64").name == "llama-swap_270_linux_arm64.tar.gz"
    with pytest.raises(ReleaseError, match="Linux"):
        swap_asset(system="Darwin", machine="arm64")


def _tarball(path, files: dict[str, bytes], root: str = "llama-b12189") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    blob = buffer.getvalue()
    path.write_bytes(blob)
    return blob


def test_extract_flattens_the_release_directory(tmp_path):
    archive = tmp_path / "llama.tar.gz"
    _tarball(archive, {"bin/llama-server": b"#!/bin/sh\n", "README.md": b"hi"})
    out = extract(archive, tmp_path / "target")
    assert (out / "bin" / "llama-server").read_text() == "#!/bin/sh\n"
    assert (out / "README.md").is_file()


def test_install_llama_verifies_digest_and_marks_binaries_executable(tmp_path, monkeypatch):
    archive = tmp_path / "blob.tar.gz"
    blob = _tarball(archive, {"bin/llama-server": b"#!/bin/sh\n", "bin/libllama.so": b"\0"})
    asset = release.Asset(name="fake-b12189.tar.gz", sha256=hashlib.sha256(blob).hexdigest(),
                          size=len(blob), url="https://example.invalid/fake.tar.gz")
    monkeypatch.setattr(release, "asset_for", lambda **_kw: asset)
    monkeypatch.setattr(release.platform, "system", lambda: "Linux")

    def fake_download(chosen, dest, **_kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        return dest

    monkeypatch.setattr(release, "download", fake_download)
    bin_dir = install_llama(root=tmp_path / "root", cuda=False)
    assert bin_dir.name == "bin"
    assert bin_dir.joinpath("llama-server").stat().st_mode & 0o111
    assert not bin_dir.joinpath("libllama.so").stat().st_mode & 0o111  # shared objects stay non-executable
    again = install_llama(root=tmp_path / "root", cuda=False)
    assert again == bin_dir


def test_download_rejects_a_wrong_digest(tmp_path, monkeypatch):
    asset = release.Asset(name="x.tar.gz", sha256="0" * 64, size=4, url="https://example.invalid/x")

    class Response:
        def __init__(self):
            self.chunks = [b"data", b""]

        def read(self, _n=-1):
            return self.chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(release.urllib.request, "urlopen", lambda *_a, **_k: Response())
    with pytest.raises(ReleaseError, match="digest mismatch"):
        release.download(asset, tmp_path / "x.tar.gz")
    assert not (tmp_path / "x.tar.gz").exists()
    assert not (tmp_path / "x.tar.gz.part").exists()

