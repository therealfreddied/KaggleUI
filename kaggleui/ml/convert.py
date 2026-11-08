"""Merge an adapter into its base and convert the result to GGUF, both as durable jobs."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..engines.context import DeployError
from ..schemas import ArtifactManifest, ModelRef
from ..util import short_id
from ..workbench.runner import start_in_background
from .hub import resolve_base
from .recipes import QUANTS

if TYPE_CHECKING:
    from ..server.state import AppState

def _artifact_dir(state: AppState, ref: str, *, kinds: tuple[str, ...]) -> tuple[ArtifactManifest, Path]:
    name, _, version = ref.partition("@")
    manifest = state.artifacts.get(name, version or None)
    if manifest is None:
        raise DeployError(f"No artifact {ref}")
    if manifest.kind not in kinds:
        raise DeployError(f"{ref} is a {manifest.kind} artifact; expected {'/'.join(kinds)}")
    return manifest, state.artifacts.dir_for(manifest.artifact_id, manifest.version)

def _gguf_files(directory: Path, files: list[str], quant: str) -> list[Path]:
    """GGUF outputs ordered so the requested quant is first; the f16 staging file must not win."""
    paths = [directory / f for f in files if f.endswith(".gguf")]
    paths = [path for path in paths if path.is_file()]
    if quant == "F16":
        return paths
    exact = [path for path in paths if quant in path.name]
    return exact + [path for path in paths if path not in exact]

async def merge(state: AppState, base: ModelRef, adapter: str, *, name: str = "") -> dict:
    manifest, adapter_dir = _artifact_dir(state, adapter, kinds=("adapter",))
    if not any((adapter_dir / f).exists() for f in ("adapter_model.safetensors", "adapter_model.bin")):
        raise DeployError(f"{adapter} has no adapter weights")
    run_id = short_id()
    base_path = await resolve_base(base, state.paths, log=lambda line: None, dep_id=f"merge-{run_id}")
    out = state.paths.cache / "merge" / run_id / "out"
    label = name.strip() or f"{base.id.split('/')[-1]}-merged"
    spec = {"mode": "merge", "base_path": str(base_path), "adapter_path": str(adapter_dir),
            "out_dir": str(out), "scratch": str(out.parent), "python": None}
    spec_path = out.parent / "spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2))
    argv = [sys.executable, "-m", "kaggleui.ml.quantize_worker", "--spec", str(spec_path)]
    log_path = state.paths.logs / f"merge-{run_id}.log"
    record = state.jobs.create("merge", {"cmd": argv, "spec_path": str(spec_path), "base": base.model_dump(mode="json"),
                                        "adapter": f"{manifest.artifact_id}@{manifest.version}", "name": label},
                               logs_path=str(log_path), worker="quantize_worker")

    def on_event(event: dict) -> None:
        payload = event.get("artifact")
        if not isinstance(payload, dict) or not payload.get("dir"):
            return
        directory = Path(payload["dir"])
        files = [directory / f for f in payload.get("files") or [] if (directory / f).is_file()]
        if not files:
            return
        saved = state.artifacts.add(ArtifactManifest(
            artifact_id=f"model-{label}", version=time.strftime("m%Y%m%d-%H%M%S"), kind="model",
            created_at=time.time(), provenance={"merged_from": [base.id, f"{manifest.artifact_id}@{manifest.version}"],
                                                "job": record.id},
            recipe={"merge": {"base": base.id, "adapter": f"{manifest.artifact_id}@{manifest.version}"}}),
            files, overwrite=True)
        state.jobs.update(record.id, artifacts=[f"{saved.artifact_id}@{saved.version}"])

    start_in_background(state.jobs, record, argv, env=_env(), log_path=log_path, name=f"merge-{run_id}",
                        on_event=on_event)
    return record.public()

def quantize(state: AppState, model: str, quant: str, *, name: str = "") -> dict:
    if quant not in QUANTS:
        raise DeployError(f"quant must be one of {', '.join(QUANTS)}")
    manifest, model_dir = _artifact_dir(state, model, kinds=("model", "adapter"))
    if not (model_dir / "config.json").exists():
        raise DeployError(f"{model} is not a transformers model directory")
    run_id = short_id()
    label = name.strip() or manifest.artifact_id.removeprefix("model-")
    out = state.paths.cache / "gguf" / run_id / "out"
    spec = {"mode": "gguf", "model_path": str(model_dir), "out_dir": str(out), "quant": quant,
            "scratch": str(out.parent), "python": None, "quantize_bin": None}
    spec_path = out.parent / "spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2))
    argv = [sys.executable, "-m", "kaggleui.ml.quantize_worker", "--spec", str(spec_path)]
    log_path = state.paths.logs / f"gguf-{run_id}.log"
    record = state.jobs.create("gguf", {"cmd": argv, "spec_path": str(spec_path), "model": model,
                                        "quant": quant, "name": label},
                               logs_path=str(log_path), worker="quantize_worker")

    def on_event(event: dict) -> None:
        payload = event.get("artifact")
        if not isinstance(payload, dict) or not payload.get("dir"):
            return
        directory = Path(payload["dir"])
        candidates = _gguf_files(directory, payload.get("files") or [], quant)
        if not candidates:
            return
        from ..gguf import read_metadata

        try:
            meta = read_metadata(candidates[0])
        except Exception as exc:  # a broken GGUF must not land in the registry
            state.jobs.update(record.id, error=f"converted GGUF is unreadable: {exc}")
            return
        saved = state.artifacts.add(ArtifactManifest(
            artifact_id=f"gguf-{label}", version=time.strftime("g%Y%m%d-%H%M%S"), kind="export",
            created_at=time.time(), backend={"format": "gguf", "quant": quant,
                                             "architecture": str(meta.get("general.architecture") or "")},
            provenance={"model": model, "job": record.id}), candidates, overwrite=True)
        state.jobs.update(record.id, artifacts=[f"{saved.artifact_id}@{saved.version}"])

    start_in_background(state.jobs, record, argv, env=_env(), log_path=log_path, name=f"gguf-{run_id}",
                        on_event=on_event)
    return record.public()

async def convert(state: AppState, artifact: str, quant: str, *, out_name: str = "",
                  calibration: Path | None = None) -> dict:
    """One-shot adapter/model → GGUF job behind ``POST /workbench/ml/convert``.

    Adapter artifacts carry their base in ``provenance["base"]``; the worker merges before converting, so
    a single persisted job produces the export.
    """
    manifest, source_dir = _artifact_dir(state, artifact, kinds=("adapter", "model"))
    if quant not in QUANTS:
        raise DeployError(f"unknown quant {quant}; choose one of {', '.join(QUANTS)}")
    if not (source_dir / "config.json").is_file():
        raise DeployError(f"{artifact} is not a transformers model directory (no config.json)")
    base_path = None
    merged_from: list[str] = []
    if manifest.kind == "adapter":
        base_id = str(manifest.provenance.get("base") or manifest.recipe.get("base") or "")
        if not base_id:
            raise DeployError(f"{artifact} has no base model in its provenance")
        base = ModelRef(source="hf", id=base_id, variant=None, engine="llamacpp")
        run_id = short_id()
        base_path = str(await resolve_base(base, state.paths, log=lambda line: None, dep_id=f"merge-{run_id}"))
        merged_from = [base_id, artifact]
    else:
        run_id = short_id()
    label = out_name.strip() or manifest.artifact_id
    out = state.paths.cache / "convert" / run_id / "out"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"mode": "convert" if base_path else "gguf", "quant": quant, "out_dir": str(out),
            "scratch": str(out.parent), "python": None, "quantize_bin": None,
            "model_path": str(source_dir)}
    if calibration is not None:
        # Quantization reads the importance matrix built from this text; llama-quantize uses it.
        spec["calibration"] = str(calibration)
    if base_path:
        spec["base_path"] = base_path
        spec["adapter_path"] = str(source_dir)
    spec_path = out.parent / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    argv = [sys.executable, "-m", "kaggleui.ml.quantize_worker", "--spec", str(spec_path)]
    log_path = state.paths.logs / f"convert-{run_id}.log"
    record = state.jobs.create("convert", {"cmd": argv, "spec_path": str(spec_path), "artifact": artifact,
                                           "quant": quant, "name": label},
                               logs_path=str(log_path), worker="quantize_worker")

    def on_event(event: dict) -> None:
        payload = event.get("artifact")
        if not isinstance(payload, dict) or not payload.get("dir"):
            return
        directory = Path(payload["dir"])
        candidates = _gguf_files(directory, payload.get("files") or [], quant)
        if not candidates:
            return
        from ..gguf import read_metadata

        try:
            meta = read_metadata(candidates[0])
        except Exception as exc:  # a broken GGUF must not land in the registry
            state.jobs.update(record.id, error=f"converted GGUF is unreadable: {exc}")
            return
        saved = state.artifacts.add(ArtifactManifest(
            artifact_id=f"gguf-{label}", version=time.strftime("g%Y%m%d-%H%M%S"), kind="export",
            created_at=time.time(), backend={"format": "gguf", "quant": quant,
                                             "architecture": str(meta.get("general.architecture") or "")},
            provenance={"model": artifact, "job": record.id, "merged_from": merged_from}), candidates,
            overwrite=True)
        state.jobs.update(record.id, artifacts=[f"{saved.artifact_id}@{saved.version}"])

    start_in_background(state.jobs, record, argv, env=_env(), log_path=log_path, name=f"convert-{run_id}",
                        on_event=on_event)
    return record.public()

def _env() -> dict[str, str]:
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parent.parent.parent)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


