"""Child process entry point for one LoRA/QLoRA run.

Builds (or reuses) an isolated training venv, then hands the real work to ``kaggleui.ml.training``
inside it. Progress crosses the process boundary as ``KUI_EVENT`` lines, same as storage jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ..config import Paths
from ..engines.install import ensure_venv_at
from ..workbench.runner import EVENT_TAG

REQUIREMENTS = ["trl>=0.12", "peft>=0.13", "accelerate>=0.34", "datasets>=2.20", "transformers>=4.45"]

def emit(payload: dict) -> None:
    print(EVENT_TAG + json.dumps(payload), flush=True)

def progress(stage: str, fraction: float | None, message: str) -> None:
    emit({"progress": fraction, "stage": stage, "message": message})

def log(line: str) -> None:
    print(line, flush=True)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaggleui.ml.train_worker")
    parser.add_argument("--spec", required=True, help="Path to the run spec JSON")
    parser.add_argument("--out", default=None, help="Override the output directory")
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    paths = Paths(os.environ.get("KAGGLEUI_HOME")).ensure()
    out_dir = Path(args.out or spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        reqs = list(REQUIREMENTS)
        if spec.get("method") == "qlora":
            reqs.append("bitsandbytes>=0.43")
        python = ensure_venv_at(paths, "train", reqs, progress=progress, log=log,
                                python=spec.get("python"), label="the training stack")
    except Exception as exc:  # surfaced in the job record
        emit({"error": f"training environment: {exc}"})
        return 2
    check = subprocess.run([python, "-c", "import torch; print(torch.__version__)"], capture_output=True,
                           text=True)
    log(f"torch: {(check.stdout or check.stderr).strip() or 'missing'}")
    if check.returncode != 0:
        log("torch is missing in the training venv; installing the CPU wheel as a last resort")
        install = subprocess.run([python, "-m", "pip", "install", "-q", "torch"], capture_output=True,
                                 text=True)
        if install.returncode != 0:
            emit({"error": "torch is unavailable in the training environment; pick a GPU runtime"})
            return 3
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parent.parent.parent)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [python, "-m", "kaggleui.ml.training", "--spec", str(args.spec), "--out", str(out_dir)]
    proc = subprocess.run(cmd, env=env)
    return proc.returncode

if __name__ == "__main__":
    sys.exit(main())
