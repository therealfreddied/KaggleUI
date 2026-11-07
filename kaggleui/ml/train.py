"""Queue a training run as a persisted job with logs, cancel and checkpoint resume."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..engines.context import DeployError
from ..schemas import ArtifactManifest
from ..workbench.runner import start_in_background
from .datasets import validate
from .hub import latest_checkpoint, resolve_base
from .recipes import TrainingRun, worker_argv, worker_env

if TYPE_CHECKING:
    from ..server.state import AppState

def dataset_dir(state: AppState, artifact_id: str) -> Path:
    name, _, version = artifact_id.partition("@")
    manifest = state.artifacts.get(name, version or None)
    if manifest is None:
        raise DeployError(f"No dataset artifact {artifact_id}")
    return state.artifacts.dir_for(manifest.artifact_id, manifest.version)

def _dataset_file(directory: Path) -> Path:
    for candidate in sorted(directory.iterdir()):
        if candidate.name == "manifest.json" or not candidate.is_file():
            continue
        if candidate.suffix.lower() in (".jsonl", ".json", ".csv"):
            return candidate
    raise DeployError("The dataset artifact contains no .jsonl/.json/.csv file")

def normalized_jsonl(state: AppState, dataset: Path, scratch: Path) -> Path:
    """Validate an artifact's dataset and write the normalized JSONL the worker reads."""
    report = validate(dataset)
    if not report.ok:
        raise DeployError("; ".join(report.errors[:3]) or "dataset failed validation")
    out = scratch / "train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in dataset.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        turns = raw.get("messages") if isinstance(raw, dict) else raw
        rows.append(json.dumps({"messages": turns}, ensure_ascii=False))
    out.write_text("\n".join(rows) + "\n")
    return out

def check_recipe(state: AppState, run: TrainingRun) -> tuple[Path, Path]:
    """Resolve the base model and normalized dataset without starting anything."""
    if not state.artifacts.get(*run.dataset_artifact.split("@", 1)):
        raise DeployError(f"No dataset artifact {run.dataset_artifact}")
    scratch = state.paths.cache / "train" / (run.id or f"pending-{int(time.time())}")
    dataset = _dataset_file(dataset_dir(state, run.dataset_artifact))
    data = normalized_jsonl(state, dataset, scratch)
    return data, scratch

async def start(state: AppState, run: TrainingRun) -> dict:
    spec_path, scratch = await _prepare(state, run)
    log_path = state.paths.logs / f"train-{run.id}.log"
    argv = worker_argv(spec_path)
    record = state.jobs.create("train", {"cmd": argv, "spec_path": str(spec_path),
                                         "run": run.model_dump(mode="json")},
                               resume_supported=True, logs_path=str(log_path), worker="train_worker")
    start_in_background(state.jobs, record, argv, env=worker_env(), log_path=log_path,
                        name=f"train-{run.id}", on_event=lambda event: _on_event(state, record.id, run, event))
    return record.public()

async def _prepare(state: AppState, run: TrainingRun) -> tuple[Path, Path]:
    if not run.id:
        from ..util import short_id

        run.id = short_id()
    scratch = state.paths.cache / "train" / run.id
    dataset = _dataset_file(dataset_dir(state, run.dataset_artifact))
    data = normalized_jsonl(state, dataset, scratch)
    base = await resolve_base(run.base, state.paths, log=lambda line: None, dep_id=f"train-{run.id}")
    spec = run.worker_spec(dataset_path=data, model_path=base, out_dir=scratch / "out", scratch=scratch)
    spec_path = scratch / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    return spec_path, scratch

def resume(state: AppState, job_id: str) -> dict:
    record = state.jobs.get(job_id)
    if record is None:
        raise DeployError("No such training job")
    spec_path = Path(record.spec.get("spec_path", ""))
    if not spec_path.is_file():
        raise DeployError("The training spec is gone; start a new run")
    spec = json.loads(spec_path.read_text())
    checkpoint = latest_checkpoint(Path(spec["scratch"]) / "checkpoints")
    if checkpoint is None:
        raise DeployError("No checkpoint to resume from yet")
    spec["resume_from"] = str(checkpoint)
    spec_path.write_text(json.dumps(spec, indent=2))
    fresh = state.jobs.create("train", {**record.spec, "cmd": worker_argv(spec_path), "spec_path": str(spec_path)},
                              resume_supported=True, logs_path=record.logs_path, worker="train_worker")
    run_data = record.spec.get("run")
    run = TrainingRun.model_validate(run_data) if run_data else None
    on_event = (lambda event: _on_event(state, fresh.id, run, event)) if run else None
    start_in_background(state.jobs, fresh, worker_argv(spec_path), env=worker_env(),
                        log_path=record.logs_path, name=f"train-{job_id}-resume", on_event=on_event)
    return fresh.public()


def _on_event(state: AppState, job_id: str, run: TrainingRun, event: dict) -> None:
    """Turn the worker's final artifact event into a registry entry plus a job reference."""
    payload = event.get("artifact")
    if not isinstance(payload, dict) or not payload.get("dir"):
        return
    directory = Path(payload["dir"])
    files = [directory / name for name in payload.get("files") or [] if (directory / name).is_file()]
    if not files:
        return
    manifest = ArtifactManifest(
        artifact_id=f"adapter-{run.output_name}", version=time.strftime("a%Y%m%d-%H%M%S"), kind="adapter",
        created_at=time.time(), files=[], recipe=run.model_dump(mode="json"),
        provenance={"base": run.base.id, "dataset": run.dataset_artifact, "job": job_id})
    saved = state.artifacts.add(manifest, files, overwrite=True)
    ref = f"{saved.artifact_id}@{saved.version}"
    record = state.jobs.get(job_id)
    refs = [r for r in (record.artifacts if record else []) if "@" in r]
    if ref not in refs:
        state.jobs.update(job_id, artifacts=[*refs, ref])
