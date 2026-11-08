"""Kaggle notebook adapter.

Verified against the installed CLI: ``kaggle kernels`` exposes push/pull/status/output/logs and
``kernels push`` accepts ``--timeout`` and ``--accelerator``. There is no stop or cancel subcommand,
so :meth:`KaggleNotebooks.stop` reports that limitation instead of inventing an API call.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class KaggleError(Exception):
    pass

def _run(argv: list[str], *, env: dict[str, str] | None = None, timeout: float = 120.0) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise KaggleError("the kaggle CLI is not installed; pip install kaggle") from exc
    except subprocess.TimeoutExpired as exc:
        raise KaggleError(f"kaggle command timed out after {int(timeout)}s") from exc

class KaggleNotebooks:
    def __init__(self, *, binary: str = "kaggle", credential: dict[str, Any] | None = None):
        self.binary = binary
        self.credential = credential or {}

    def env(self) -> dict[str, str]:
        env = {**os.environ}
        username = self.credential.get("username")
        key = self.credential.get("key")
        key_file = self.credential.get("key_file")
        if username:
            env["KAGGLE_USERNAME"] = str(username)
        if key:
            env["KAGGLE_KEY"] = str(key)
        elif key_file:
            env["KAGGLE_CONFIG_DIR"] = str(Path(str(key_file)).expanduser().parent)
        return env

    def _call(self, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
        return _run([self.binary, *args], env=self.env(), timeout=timeout)

    def deploy(self, *, slug: str, source: str, title: str | None = None,
               accelerator: str | None = None, timeout_s: int | None = None, run: bool = True,
               dataset_sources: list[str] | None = None, internet: bool = True,
               extra_metadata: dict | None = None) -> dict:
        if not slug or "/" not in slug:
            raise KaggleError("notebook slug must look like owner/kernel-name")
        src = Path(source).expanduser()
        if not src.exists():
            raise KaggleError(f"notebook source does not exist: {src}")
        with tempfile.TemporaryDirectory(prefix="kui-kernel-") as tmp:
            folder = Path(tmp)
            code_file = src.name
            shutil.copy2(src, folder / code_file)
            metadata = {
                "id": slug, "title": title or slug.split("/", 1)[1], "code_file": code_file,
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": bool(accelerator and accelerator.lower() != "none"),
                "enable_tpu": bool(accelerator and accelerator.lower().startswith("tpu")),
                "enable_internet": bool(internet),
                "dataset_sources": dataset_sources or [], "competition_sources": [], "kernel_sources": [],
            }
            # A pushed notebook keeps what the existing version had (BYOD image, machine shape);
            # only the identity and the code file are ours to set.
            for key, value in (extra_metadata or {}).items():
                if key not in {"id", "code_file"}:
                    metadata[key] = value
            (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1))
            args = ["kernels", "push", "-p", str(folder)]
            if accelerator and accelerator.lower() != "none":
                args += ["--accelerator", accelerator]
            if timeout_s:
                args += ["--timeout", str(int(timeout_s))]
            if not run:
                args.append("--no-run")
            result = self._call(*args, timeout=300)
        if result.returncode != 0:
            raise KaggleError((result.stderr or result.stdout or "kaggle push failed").strip())
        return {"slug": slug, "url": f"https://www.kaggle.com/code/{slug}",
                "output": (result.stdout or result.stderr).strip()}

    def status(self, slug: str) -> dict:
        result = self._call("kernels", "status", slug, timeout=60)
        if result.returncode != 0:
            raise KaggleError((result.stderr or result.stdout or "kaggle status failed").strip())
        text = (result.stdout or result.stderr).strip()
        state = "unknown"
        for candidate in ("RUNNING", "COMPLETE", "ERROR", "QUEUED", "CANCEL_ACKNOWLEDGED",
                          "CANCEL_REQUESTED", "UNKNOWN"):
            if candidate in text.upper():
                state = candidate.lower()
                break
        return {"slug": slug, "state": state, "raw": text,
                "url": f"https://www.kaggle.com/code/{slug}"}

    def open(self, slug: str) -> str:
        """Return the canonical URL; never launch a browser from a remote notebook cell."""
        return f"https://www.kaggle.com/code/{slug}"

    def stop(self, slug: str) -> dict:
        current = self.status(slug)
        return {**current, "stopped": False, "capability": "unavailable",
                "note": "The Kaggle CLI exposes push/status/logs/output but no stop or cancel command. "
                        "End the run from the Kaggle notebook UI (or stop the notebook cell); KaggleUI "
                        "cannot end a Kaggle session on your behalf."}

    def output(self, slug: str, destination: str) -> dict:
        target = Path(destination).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        result = self._call("kernels", "output", slug, "-p", str(target), timeout=600)
        if result.returncode != 0:
            raise KaggleError((result.stderr or result.stdout or "kaggle output failed").strip())
        return {"slug": slug, "path": str(target)}

