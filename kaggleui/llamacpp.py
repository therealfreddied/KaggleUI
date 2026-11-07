"""llama.cpp runtime: installed binaries, GGUF fetching, and the installer CLI.

The engine runs this module as a supervised child (``python -m kaggleui.llamacpp install``) so installs
and downloads are cancellable jobs with a real log file:

    KUI_EVENT {"progress": 0.42, "message": "Downloading …"}
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .config import Paths
from .engines import llamacpp_release as release
from .gguf import GGUFError, read_metadata
from .util import human_bytes

REQUIRED_BINS = ("llama-server", "llama-quantize", "llama-gguf-split")


def event(progress: float | None = None, message: str | None = None, **extra: Any) -> None:
    """Emit one machine-readable progress line; the runner folds it into the job record."""
    payload: dict[str, Any] = dict(extra)
    if progress is not None:
        payload["progress"] = round(max(0.0, min(1.0, progress)), 4)
    if message:
        payload["message"] = message
    import json

    print("KUI_EVENT " + json.dumps(payload), flush=True)


def runtime_root(paths: Paths) -> Path:
    return paths.home / "llamacpp"


def installed_dir(paths: Paths) -> Path:
    return runtime_root(paths) / "llama" / release.LLAMACPP_RELEASE


def find_bin(paths: Paths, name: str) -> Path | None:
    found = release.find_file(installed_dir(paths), name)
    if found is None:
        return None
    return found


def require_bin(paths: Paths, name: str) -> Path:
    found = find_bin(paths, name)
    if found is None:
        raise RuntimeError(f"llama.cpp is not installed ({name} missing); run the llama.cpp installer first")
    return found


def swap_bin(paths: Paths) -> Path | None:
    candidate = runtime_root(paths) / "bin" / "llama-swap"
    return candidate if candidate.exists() else None


def is_installed(paths: Paths) -> bool:
    return all(find_bin(paths, name) is not None for name in REQUIRED_BINS)


def needs_cuda(paths: Paths, hw: Any) -> bool:
    """CUDA binaries only when the machine really has an NVIDIA device."""
    return bool(getattr(hw, "gpus", None)) and getattr(hw, "accelerator", "") != "tpu"


def install(paths: Paths, *, cuda: bool, log=lambda _m: None) -> Path:
    def progress(frac: float, message: str) -> None:
        event(frac * 0.95, message)

    event(0.0, "Installing llama.cpp")
    bin_dir = release.install_llama(root=runtime_root(paths), cuda=cuda, log=log, progress=progress)
    missing = [name for name in REQUIRED_BINS if release.find_file(bin_dir, name) is None]
    if missing:
        raise release.ReleaseError(f"installed build is missing {', '.join(missing)}")
    event(1.0, f"llama.cpp {release.LLAMACPP_RELEASE} ready")
    return bin_dir


def install_swap(paths: Paths, *, log=lambda _m: None) -> Path:
    event(0.0, "Installing llama-swap")
    binary = release.install_swap(root=runtime_root(paths), log=log,
                                  progress=lambda frac, msg: event(frac * 0.95, msg))
    event(1.0, f"llama-swap v{release.LLAMA_SWAP_RELEASE} ready")
    return binary


# -- GGUF fetching ------------------------------------------------------------------------------
def _headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "--").replace(" ", "_")


def model_dir(paths: Paths, repo_id: str) -> Path:
    return paths.models / "llama" / repo_slug(repo_id)


def resolve_url(repo_id: str, filename: str, revision: str | None = None) -> str:
    return (f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/"
            f"{quote(revision or 'main', safe='')}/{quote(filename, safe='/')}")


def download_file(url: str, dest: Path, *, expected: int | None = None, offset_base: int = 0,
                  total_base: int = 0, on_progress=None, timeout: float = 60.0) -> Path:
    """Stream one file with resume; verify the byte count when the Hub told us the size."""
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if expected is not None and have == expected:
        part.replace(dest)
        return dest
    if expected is not None and have > expected:
        part.unlink()
        have = 0
    headers = dict(_headers())
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)  # noqa: S310 - https URL assembled from a repo id
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            if have and resp.status != 206:
                have = 0
            mode = "ab" if have else "wb"
            with part.open(mode) as out:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    have += len(block)
                    if on_progress:
                        on_progress(offset_base + have, total_base or expected)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and expected is not None and part.exists() and part.stat().st_size == expected:
            part.replace(dest)
            return dest
        raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
    if expected is not None and have != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: expected {expected} bytes, downloaded {have}")
    part.replace(dest)
    return dest


def fetch(paths: Paths, repo_id: str, files: list[dict[str, str]], *, revision: str | None = None) -> list[Path]:
    """Download ``files`` (each ``{"path", "size"}``) into the model directory, then parse the GGUF."""
    target = model_dir(paths, repo_id)
    total = sum(int(f.get("size") or 0) for f in files) or 0
    done = 0
    out: list[Path] = []
    for item in files:
        name = item["path"]
        size = int(item.get("size") or 0) or None
        dest = target / Path(name).name
        if dest.exists() and (size is None or dest.stat().st_size == size):
            event(message=f"{dest.name} already present")
            done += size or 0
            out.append(dest)
            continue
        event(done / total if total else None, f"Downloading {dest.name}"
              + (f" ({human_bytes(size)})" if size else ""))
        download_file(resolve_url(repo_id, name, revision), dest, expected=size, offset_base=done, total_base=total,
                      on_progress=lambda got, tot: event(got / tot if tot else None))
        done += size or 0
        out.append(dest)
    first = out[0]
    try:
        meta = read_metadata(first)
    except GGUFError as exc:
        raise RuntimeError(f"{first.name} is not a readable GGUF: {exc}") from exc
    event(1.0, f"Downloaded {first.name}", metadata={"architecture": meta.get("general.architecture")})
    return out


# -- CLI ----------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kaggleui.llamacpp")
    parser.add_argument("--home", help="override KAGGLEUI_HOME")
    sub = parser.add_subparsers(dest="command", required=True)
    install_p = sub.add_parser("install", help="install the pinned llama.cpp build")
    install_p.add_argument("--cuda", action="store_true")
    install_p.add_argument("--no-cuda", action="store_true")
    sub.add_parser("install-swap", help="install the pinned llama-swap binary")
    fetch_p = sub.add_parser("fetch", help="download GGUF files for a repo")
    fetch_p.add_argument("--repo", required=True)
    fetch_p.add_argument("--file", action="append", default=[], required=True)
    fetch_p.add_argument("--size", action="append", default=[])
    fetch_p.add_argument("--revision")
    sub.add_parser("status", help="print installed releases")
    args = parser.parse_args(argv)

    paths = Paths(Path(args.home) if args.home else None)
    paths.ensure()

    if args.command == "install":
        install(paths, cuda=args.cuda and not args.no_cuda, log=lambda line: print(line, flush=True))
        return 0
    if args.command == "install-swap":
        install_swap(paths, log=lambda line: print(line, flush=True))
        return 0
    if args.command == "fetch":
        sizes = [int(v) for v in args.size]
        files = [{"path": p, "size": str(sizes[i]) if i < len(sizes) else ""} for i, p in enumerate(args.file)]
        fetch(paths, args.repo, files, revision=args.revision)
        return 0
    if args.command == "status":
        import json

        print(json.dumps({"llama": str(installed_dir(paths)), "installed": is_installed(paths),
                          "swap": str(swap_bin(paths)) if swap_bin(paths) else None}))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())


def doctor(paths: Paths) -> dict[str, Any]:
    """Reachability facts for the panel without launching anything."""
    bins = {name: str(find_bin(paths, name)) if find_bin(paths, name) else None for name in REQUIRED_BINS}
    return {"release": release.LLAMACPP_RELEASE, "binaries": bins, "installed": is_installed(paths),
            "swap": str(swap_bin(paths)) if swap_bin(paths) else None,
            "disk_free": shutil.disk_usage(paths.home).free if paths.home.exists() else None}
