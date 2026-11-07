"""Pinned llama.cpp and llama-swap release assets.

Every download is verified with the digest published by the upstream release. Digests are full
``sha256:`` values copied from ``api.github.com`` — never abbreviated, never computed locally for a
different file. Bump :data:`LLAMACPP_RELEASE` and the digests together.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

LLAMACPP_RELEASE = "b12189"
LLAMA_SWAP_RELEASE = "270"
_LLAMA_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_RELEASE}"
_SWAP_BASE = f"https://github.com/mostlygeek/llama-swap/releases/download/v{LLAMA_SWAP_RELEASE}"


@dataclass(frozen=True)
class Asset:
    name: str
    sha256: str
    size: int
    url: str

    @property
    def digest(self) -> str:
        return f"sha256:{self.sha256}"


def _llama(name: str, sha256: str, size: int) -> Asset:
    return Asset(name=name, sha256=sha256, size=size, url=f"{_LLAMA_BASE}/{name}")


def _swap(name: str, sha256: str, size: int) -> Asset:
    return Asset(name=name, sha256=sha256, size=size, url=f"{_SWAP_BASE}/{name}")


# llama.cpp b12189 — names copied verbatim from the release's asset list.
LLAMACPP_ASSETS: dict[tuple[str, str, str], Asset] = {
    ("Linux", "x86_64", "cuda"): _llama("llama-b12189-bin-ubuntu-cuda-12.8-x64.tar.gz",
                                        "664f720005135f621960b8f1550979a1bdd73b8ba887e2a47a314b0154fcaa0a", 188173082),
    ("Linux", "x86_64", "cpu"): _llama("llama-b12189-bin-ubuntu-x64.tar.gz",
                                       "c446a7ba4df3c3b1cc880a734fbff586d6d34b9cebc09e151754813b94ded8c9", 23058620),
    ("Linux", "aarch64", "cuda"): _llama("llama-b12189-bin-ubuntu-cuda-13.5-arm64.tar.gz",
                                         "ebed66a17dce7aae73a1facfc2b2284d4370d6768365c6c36ec04e624e9fad85", 164893248),
    ("Linux", "aarch64", "cpu"): _llama("llama-b12189-bin-ubuntu-arm64.tar.gz",
                                        "30441cf9f5d9ce1b511c85dc2f6e494db6ba18c23da39013ab35f65b54f7267f", 18732824),
    ("Darwin", "arm64", "cpu"): _llama("llama-b12189-bin-macos-arm64.tar.gz",
                                       "f30e2f3032df30a0c37c9fbd43125253dcd4cb96d6a879aa81e464944bc20aee", 16700174),
    ("Darwin", "x86_64", "cpu"): _llama("llama-b12189-bin-macos-x64.tar.gz",
                                        "c690cb11be36a910df7e3d5baf22ed2fb53c02243c72cd33aae8f5fd6c6afe55", 12438130),
}

# The CUDA builds link against libcudart; the arm64 build has no 12.8 variant, so it pairs with 13.5.
CUDART_ASSETS: dict[tuple[str, str], Asset] = {
    ("Linux", "x86_64"): _llama("cudart-llama-b12189-bin-ubuntu-cuda-12.8-x64.tar.gz",
                                "81408740817d383bd18b9f2f8d4fda14bbd1cf9dac63331e7b56c6fb82380650", 594371476),
    ("Linux", "aarch64"): _llama("cudart-llama-b12189-bin-ubuntu-cuda-13.5-arm64.tar.gz",
                                 "4420a93a69bfcda8915fd87fbf5eb2163d5e646aad72fe5ab31e490d92dade89", 559874232),
}

LLAMA_SWAP_ASSETS: dict[str, Asset] = {
    "x86_64": _swap("llama-swap_270_linux_amd64.tar.gz",
                    "b7f89c944e83e30c438846f8c513b9e64fb7bdcec1e3f47ed8be31a94d688a09", 24225160),
    "aarch64": _swap("llama-swap_270_linux_arm64.tar.gz",
                     "38ab271e1c99b7ce3c3307ef15faaf59c4165f85b2514bbb394919111bbeb759", 22618472),
}

Log = Callable[[str], None]
Progress = Callable[[float, str], None]


class ReleaseError(Exception):
    """The release cannot be installed on this machine."""


def machine_kind() -> str:
    arch = platform.machine().lower()
    return {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(arch, arch)


def asset_for(*, cuda: bool, system: str | None = None, machine: str | None = None) -> Asset:
    """Pick the llama.cpp build for this machine, falling back to the CPU build."""
    system = system or platform.system()
    machine = machine or machine_kind()
    variant = "cuda" if cuda else "cpu"
    chosen = LLAMACPP_ASSETS.get((system, machine, variant))
    if chosen is None and variant == "cuda":
        chosen = LLAMACPP_ASSETS.get((system, machine, "cpu"))
    if chosen is None:
        raise ReleaseError(f"No pinned llama.cpp build for {system}/{machine}")
    return chosen


def cudart_for(*, system: str | None = None, machine: str | None = None) -> Asset | None:
    system = system or platform.system()
    machine = machine or machine_kind()
    return CUDART_ASSETS.get((system, machine))


def swap_asset(*, machine: str | None = None, system: str | None = None) -> Asset:
    system = system or platform.system()
    machine = machine or machine_kind()
    if system != "Linux":
        raise ReleaseError(f"llama-swap prebuilt binaries cover Linux only (got {system})")
    chosen = LLAMA_SWAP_ASSETS.get(machine)
    if chosen is None:
        raise ReleaseError(f"No pinned llama-swap build for {machine}")
    return chosen


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(asset: Asset, dest: Path, *, log: Log = lambda _m: None, progress: Progress | None = None,
             timeout: float = 120.0) -> Path:
    """Download ``asset`` to ``dest`` (``.part`` + rename) and verify its digest."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    log(f"Downloading {asset.name} ({asset.size / 1e6:.0f} MB)")
    got = 0
    with urllib.request.urlopen(asset.url, timeout=timeout) as resp, part.open("wb") as out:  # noqa: S310 - pinned https URL
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            got += len(block)
            if progress and asset.size:
                progress(min(got / asset.size, 1.0), asset.name)
    actual = sha256_of(part)
    if actual != asset.sha256:
        part.unlink(missing_ok=True)
        raise ReleaseError(f"{asset.name}: digest mismatch (expected {asset.digest}, got sha256:{actual})")
    part.replace(dest)
    log(f"Verified {asset.name} ({asset.digest})")
    return dest


def extract(tarball: Path, dest: Path, *, flatten: bool = True) -> Path:
    """Extract a ``.tar.gz``; with ``flatten`` the single top-level directory is stripped."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        members = tar.getmembers()
        roots = {m.name.split("/", 1)[0] for m in members if m.name and not m.name.startswith("./")}
        strip = flatten and len(roots) == 1
        root = next(iter(roots)) if strip else None
        for member in members:
            name = member.name
            if strip and name.split("/", 1)[0] == root:
                name = name[len(root) + 1:] if "/" in name else ""
            if not name:
                continue
            member.name = name
            target = dest / name
            if not str(target.resolve()).startswith(str(dest.resolve())):
                raise ReleaseError(f"refusing to extract outside {dest}: {name}")
            # Archives are digest-verified before extraction, so the pinned build is the trust boundary.
            tar.extract(member, dest)
    return dest


def find_file(root: Path, name: str) -> Path | None:
    """Find a binary by name anywhere under ``root`` (releases vary between flat and ``build/bin``)."""
    root = Path(root)
    direct = root / name
    if direct.exists():
        return direct
    for candidate in root.rglob(name):
        if candidate.is_file():
            return candidate
    return None


def install_llama(*, root: Path, cuda: bool, log: Log = lambda _m: None, progress: Progress | None = None) -> Path:
    """Install the pinned build into ``root`` and return the directory holding the binaries."""
    asset = asset_for(cuda=cuda)
    root = Path(root)
    archive = root / "downloads" / asset.name
    if not archive.exists() or sha256_of(archive) != asset.sha256:
        download(asset, archive, log=log, progress=progress)
    target = root / "llama" / LLAMACPP_RELEASE
    existing = find_file(target, "llama-server")
    if existing is not None:
        log(f"llama.cpp {LLAMACPP_RELEASE} already installed")
        return existing.parent
    if target.exists():
        shutil.rmtree(target)
    extract(archive, target)
    if cuda:
        cudart = cudart_for()
        if cudart is not None:
            try:
                extra = root / "downloads" / cudart.name
                if not extra.exists() or sha256_of(extra) != cudart.sha256:
                    download(cudart, extra, log=log, progress=progress)
                extract(extra, target)
            except (OSError, ReleaseError) as exc:
                log(f"WARNING: CUDA runtime bundle failed ({exc}); the build may need system libcudart")
    bin_dir = find_file(target, "llama-server").parent
    for binary in bin_dir.iterdir():
        if binary.is_file() and not binary.name.endswith((".so", ".0", ".h", ".txt")):
            binary.chmod(binary.stat().st_mode | 0o111)
    log(f"llama.cpp {LLAMACPP_RELEASE} installed in {bin_dir}")
    return bin_dir


def install_swap(*, root: Path, log: Log = lambda _m: None, progress: Progress | None = None) -> Path:
    """Install the pinned llama-swap binary into ``root/bin`` and return its path."""
    asset = swap_asset()
    root = Path(root)
    archive = root / "downloads" / asset.name
    if not archive.exists() or sha256_of(archive) != asset.sha256:
        download(asset, archive, log=log, progress=progress)
    bin_dir = root / "bin"
    existing = bin_dir / "llama-swap"
    if existing.exists():
        log(f"llama-swap v{LLAMA_SWAP_RELEASE} already installed")
        return existing
    extract(archive, bin_dir)
    found = find_file(bin_dir, "llama-swap")
    if found is None:
        raise ReleaseError(f"{asset.name} did not contain a llama-swap binary")
    if found != existing:
        shutil.copy2(found, existing)
    existing.chmod(existing.stat().st_mode | 0o111)
    log(f"llama-swap v{LLAMA_SWAP_RELEASE} installed in {existing}")
    return existing
