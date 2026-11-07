"""Child process entry for merge/GGUF jobs: build the training venv, then re-exec the real body."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ..config import Paths
from ..engines.install import ensure_venv_at
from .train_worker import REQUIREMENTS, emit, log, progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaggleui.ml.quantize_worker")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    paths = Paths(os.environ.get("KAGGLEUI_HOME")).ensure()
    Path(spec["out_dir"]).mkdir(parents=True, exist_ok=True)
    try:
        python = ensure_venv_at(paths, "train", REQUIREMENTS, progress=progress, log=log,
                                python=spec.get("python"), label="the training stack")
    except Exception as exc:  # surfaced in the job record
        emit({"error": f"training environment: {exc}"})
        return 2
    if spec.get("mode") in ("gguf", "convert") and spec.get("quant") != "F16" and not spec.get("quantize_bin"):
        from ..engines.llamacpp_release import find_file, install_llama

        try:
            bin_dir = install_llama(root=paths.home / "llama", cuda=False, log=log, progress=progress)
        except Exception as exc:
            emit({"error": f"llama.cpp release: {exc}"})
            return 2
        found = find_file(bin_dir, "llama-quantize")
        if found is None:
            emit({"error": "the pinned llama.cpp release ships no llama-quantize binary"})
            return 2
        spec["quantize_bin"] = str(found)
        Path(args.spec).write_text(json.dumps(spec, indent=2))
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parent.parent.parent)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run([python, "-m", "kaggleui.ml.conversion", "--spec", str(args.spec)], env=env)
    return proc.returncode

if __name__ == "__main__":
    sys.exit(main())

