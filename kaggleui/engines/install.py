"""On-demand dependency installation.

Light engines install into the notebook's own environment (Kaggle already ships torch).
Heavy or conflicting ones (vLLM, Chatterbox, Open WebUI, some world models) get isolated uv venvs,
so they can't break the base environment's torch.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from ..config import Paths
from ..proc import run_cmd
from .context import DeployError, EngineContext

_locks: dict[str, asyncio.Lock] = {}
_done: set[str] = set()


def _lock(key: str) -> asyncio.Lock:
    return _locks.setdefault(key, asyncio.Lock())


def _version_ok(dist: str, minimum: str | None) -> bool:
    try:
        have = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return False
    if not minimum:
        return True
    try:
        from packaging.version import Version

        return Version(have) >= Version(minimum)
    except Exception:
        return True


def missing_packages(requirements: list[str]) -> list[str]:
    """Requirements like ``diffusers>=0.35`` or ``imageio[ffmpeg]`` that aren't satisfied yet."""
    out = []
    for req in requirements:
        name, minimum = req, None
        if ">=" in req:
            name, minimum = req.split(">=", 1)
        dist = name.split("[", 1)[0].strip()
        if not _version_ok(dist, minimum):
            out.append(req)
    return out


async def pip_install(ctx: EngineContext, requirements: list[str], *, python: str | None = None,
                      label: str = "dependencies") -> None:
    py = python or sys.executable
    todo = missing_packages(requirements) if py == sys.executable else requirements
    if not todo:
        return
    key = f"pip:{py}:{','.join(sorted(todo))}"
    async with _lock(key):
        if key in _done:
            return
        ctx.progress("installing", None, f"Installing {label}: {', '.join(todo)}")
        uv = shutil.which("uv")
        if uv:
            cmd = [uv, "pip", "install", "--python", py, *todo]
        else:
            cmd = [py, "-m", "pip", "install", "-q", "--disable-pip-version-check", *todo]
        rc = await run_cmd(cmd, on_line=ctx.log)
        if rc != 0:
            raise DeployError(f"Installing {label} failed (exit {rc}). See the logs for details.")
        importlib.invalidate_caches()
        _done.add(key)


async def pip_upgrade(ctx: EngineContext, requirements: list[str], *, python: str | None = None,
                      label: str = "dependencies") -> None:
    """Install the newest available version even when an older one already satisfies the engine."""
    py = python or sys.executable
    async with _lock(f"pip-upgrade:{py}"):
        ctx.progress("installing", None, f"Installing {label}")
        uv = shutil.which("uv")
        if uv:
            cmd = [uv, "pip", "install", "--upgrade", "--python", py, *requirements]
        else:
            cmd = [py, "-m", "pip", "install", "-q", "--upgrade", "--disable-pip-version-check", *requirements]
        rc = await run_cmd(cmd, on_line=ctx.log)
        if rc != 0:
            raise DeployError(f"Installing {label} failed (exit {rc}). See the logs for details.")
        importlib.invalidate_caches()


async def pip_uninstall(ctx: EngineContext, packages: list[str], *, python: str | None = None) -> None:
    py = python or sys.executable
    uv = shutil.which("uv")
    cmd = [uv, "pip", "uninstall", "--python", py, *packages] if uv else [py, "-m", "pip", "uninstall", "-y", *packages]
    ctx.progress("installing", None, f"Removing {', '.join(packages)}")
    if await run_cmd(cmd, on_line=ctx.log) != 0:
        raise DeployError(f"Removing {', '.join(packages)} failed. See the logs for details.")
    importlib.invalidate_caches()


async def ensure_uv(ctx: EngineContext) -> str:
    uv = shutil.which("uv") or (str(ctx.paths.bin / "uv") if (ctx.paths.bin / "uv").exists() else None)
    if uv:
        return uv
    async with _lock("uv"):
        ctx.progress("installing", None, "Installing uv (fast Python package manager)")
        rc = await run_cmd([sys.executable, "-m", "pip", "install", "-q", "uv"], on_line=ctx.log)
        if rc != 0:
            raise DeployError("Couldn't install uv")
        uv = shutil.which("uv")
        if not uv:
            cand = Path(sys.executable).parent / "uv"
            uv = str(cand) if cand.exists() else None
        if not uv:
            raise DeployError("uv installed but not found on PATH")
        return uv


async def ensure_venv_at(paths: Paths, name: str, requirements: list[str], *,
                         progress: Callable[[str, float | None, str], None],
                         log: Callable[[str], None], python: str | None = None,
                         system_site_packages: bool = True, label: str | None = None) -> str:
    """Create (once) an isolated venv with ``requirements``; returns its python executable.

    ``python`` defaults to the interpreter running this process, so a notebook's 3.11 or Colab's 3.12
    matches what ``--system-site-packages`` exposes (CUDA torch included).
    """
    python = python or f"{sys.version_info.major}.{sys.version_info.minor}"
    venv = paths.envs / name
    py = venv / "bin" / "python"
    marker = venv / ".kaggleui-ready"
    async with _lock(f"venv:{name}"):
        if marker.exists() and marker.read_text() == "\n".join(requirements):
            return str(py)
        uv = shutil.which("uv") or (str(paths.bin / "uv") if (paths.bin / "uv").exists() else None)
        if not uv:
            progress("installing", None, "Installing uv (fast Python package manager)")
            rc = await run_cmd([sys.executable, "-m", "pip", "install", "-q", "uv"], on_line=log)
            if rc != 0:
                raise DeployError("Couldn't install uv")
            uv = shutil.which("uv") or (str(Path(sys.executable).parent / "uv")
                                        if (Path(sys.executable).parent / "uv").exists() else None)
            if not uv:
                raise DeployError("uv installed but not found on PATH")
        progress("installing", None, f"Creating isolated environment for {label or name}")
        if not py.exists():
            cmd = [uv, "venv", str(venv), "--python", python, "--seed"]
            if system_site_packages:
                cmd.append("--system-site-packages")
            rc = await run_cmd(cmd, on_line=log)
            if rc != 0:
                raise DeployError(f"Couldn't create the {name} environment")
        progress("installing", None, f"Installing {label or name} (first time only, can take a few minutes)")
        rc = await run_cmd([uv, "pip", "install", "--python", str(py), *requirements], on_line=log)
        if rc != 0:
            raise DeployError(f"Installing {label or name} failed (exit {rc}). See the logs.")
        marker.write_text("\n".join(requirements))
        return str(py)

async def ensure_venv(ctx: EngineContext, name: str, requirements: list[str], *, python: str = "3.11",
                      system_site_packages: bool = False, label: str | None = None) -> str:
    """Engine-facing wrapper: same behaviour, context supplies paths and progress reporting."""
    return await ensure_venv_at(ctx.paths, name, requirements, progress=ctx.progress, log=ctx.log,
                                python=python, system_site_packages=system_site_packages, label=label)


async def apt_install(ctx: EngineContext, packages: list[str]) -> None:
    """Best-effort system packages (Kaggle runs as root). Never fatal."""
    if not shutil.which("apt-get") or (hasattr(os, "geteuid") and os.geteuid() != 0):
        return
    todo = [p for p in packages if not shutil.which(p)]
    if not todo:
        return
    key = f"apt:{','.join(todo)}"
    async with _lock(key):
        if key in _done:
            return
        ctx.progress("installing", None, f"Installing system packages: {', '.join(todo)}")
        await run_cmd(f"(apt-get install -y -qq {' '.join(todo)} || (apt-get update -qq && apt-get install -y -qq "
                      f"{' '.join(todo)})) >/dev/null 2>&1", on_line=ctx.log)
        _done.add(key)


def worker_pythonpath(ctx: EngineContext) -> str:
    """A PYTHONPATH exposing only the kaggleui package (for workers running in isolated venvs)."""
    link_dir = ctx.paths.home / "pylink"
    link_dir.mkdir(parents=True, exist_ok=True)
    target = Path(__file__).resolve().parents[1]
    link = link_dir / "kaggleui"
    if not link.exists():
        try:
            link.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            pass
    return str(link_dir)


def nvidia_lib_path() -> str:
    """LD_LIBRARY_PATH entries for pip-installed CUDA libs (cuDNN/cuBLAS), needed by CTranslate2."""
    paths = []
    for entry in sys.path:
        base = Path(entry) / "nvidia"
        if base.is_dir():
            for sub in ("cudnn", "cublas", "cuda_runtime"):
                lib = base / sub / "lib"
                if lib.is_dir():
                    paths.append(str(lib))
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        paths.append(existing)
    return ":".join(paths)
