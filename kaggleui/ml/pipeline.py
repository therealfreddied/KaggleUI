"""Chain the existing Model Studio jobs into one resumable pipeline.

Each stage is a normal job (train, abliteration import, GGUF export); the pipeline just starts the
next one with the artifact the previous stage produced and waits for its terminal state. Stages that
already have a matching artifact in the job record are skipped, so a failed plan can be restarted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..engines.context import DeployError
from ..schemas import DeploymentSpec, ModelRef
from . import convert, edit, evaluate, train
from .recipes import TrainingRun

if TYPE_CHECKING:
    from ..server.state import AppState
    from ..workbench.jobs import JobRecord

StageKind = Literal["train", "eval", "edit", "export", "deploy_test"]
STAGES: tuple[StageKind, ...] = ("train", "eval", "edit", "export", "deploy_test")
POLL_S = 2.0
FORGE_KIND = "forge"
MANIFEST_NAME = "run.json"
MANIFEST_KEYS = ("stage", "engine", "engine_version", "inputs_sha256", "outputs", "status")
RESUMABLE_STATES = frozenset({"failed", "cancelled", "interrupted"})
# Only this much dataset text is fed to llama-imatrix: enough calibration, bounded runtime.
CALIBRATION_CHARS = 2_000_000
_STAGE_ENGINE = {"train": "trl", "eval": "harness", "export": "llama.cpp", "deploy_test": "kaggleui"}

class ForgePlan(BaseModel):
    base: ModelRef
    output_name: str = ""
    dataset_artifact: str = ""
    stages: list[StageKind] = Field(default_factory=lambda: ["train", "export"])
    method: Literal["lora", "qlora"] = "qlora"
    eval_deployment: str = ""
    eval_profile: str = "smoke"
    eval_artifact: str = ""
    edit_method: Literal["quick", "heretic"] = "quick"
    edit_trials: int = 40
    allow_blind_edit: bool = False
    allow_quick_quant: bool = False
    quant: str = "q4_k_m"

_tasks: dict[str, asyncio.Task] = {}


def _order(stages: list[StageKind]) -> list[StageKind]:
    wanted = set(stages)
    ordered = [s for s in STAGES if s in wanted]
    if "train" in wanted and not wanted & {"export", "deploy_test"}:
        raise DeployError("a training pipeline must finish with export or deploy_test")
    if "deploy_test" in wanted and "export" not in wanted:
        ordered = [s for s in ordered if s != "deploy_test"] + (["export", "deploy_test"])
        ordered = [s for s in STAGES if s in set(ordered)]
    return ordered


def _check(plan: ForgePlan, stage: StageKind) -> None:
    """Guardrails live here, not in the UI, so an API caller cannot skip them."""
    if stage == "train" and not plan.dataset_artifact:
        raise DeployError("train needs a dataset artifact")
    if stage == "eval" and not plan.eval_deployment:
        raise DeployError("eval needs a ready deployment (eval_deployment)")
    if stage == "edit" and not (plan.eval_artifact or plan.eval_deployment) and not plan.allow_blind_edit:
        raise DeployError("edit needs an eval artifact; run eval first or set allow_blind_edit")
    if stage == "export" and plan.quant != "q8_0" and not plan.allow_quick_quant and not plan.dataset_artifact:
        raise DeployError("a quantized export builds its importance matrix from the training dataset; attach "
                          "a dataset artifact or set allow_quick_quant to accept quantization without one")
    if stage == "deploy_test" and plan.base.source == "ollama":
        raise DeployError("deploy_test supports Hugging Face and artifact models only")


async def start(state: AppState, plan: ForgePlan) -> dict:
    stages = _order(plan.stages)
    for stage in stages:
        _check(plan, stage)
    if not plan.output_name:
        plan.output_name = f"{plan.base.id.split('/')[-1]}-forge"
    record = state.jobs.create(FORGE_KIND, {"plan": plan.model_dump(mode="json"), "stages": stages,
                                            "results": {}, "manifests": {}, "stage": ""})
    state.jobs.update(record.id, state="running", started_at=time.time(), message="Starting",
                      resume_supported=1)
    _tasks[record.id] = asyncio.create_task(_run(state, record, plan, stages))
    return state.jobs.get(record.id).public()

async def resume(state: AppState, job_id: str) -> dict:
    """Re-enter a dead pipeline at its first stage without a verified manifest."""
    record = state.jobs.get(job_id)
    if record is None or record.kind != FORGE_KIND:
        raise DeployError("No such pipeline job")
    if job_id in _tasks:
        raise DeployError("pipeline is still running")
    if record.state not in RESUMABLE_STATES:
        raise DeployError(f"pipeline {job_id} is {record.state}; only failed, cancelled or interrupted "
                          "runs can resume")
    spec = record.spec or {}
    plan = ForgePlan.model_validate(spec.get("plan") or {})
    stages = [s for s in (spec.get("stages") or []) if s in STAGES] or _order(plan.stages)
    results = dict(spec.get("results") or {})
    manifests = dict(spec.get("manifests") or {})
    refs = {r for r in results.values() if isinstance(r, str)}
    refs |= {plan.dataset_artifact, plan.eval_artifact} - {""}
    verify_artifacts(state, sorted(refs))
    state.jobs.update(job_id, state="running", started_at=time.time(), ended_at=None, error=None,
                      cancel_requested=0, message="Resuming", resume_supported=1)
    _tasks[job_id] = asyncio.create_task(_run(state, state.jobs.get(job_id), plan, stages,
                                              results=results, manifests=manifests))
    return state.jobs.get(job_id).public()


async def cancel(state: AppState, job_id: str) -> dict:
    record = state.jobs.get(job_id)
    if record is None or record.kind != FORGE_KIND:
        raise DeployError("No such pipeline job")
    state.jobs.update(job_id, cancel_requested=1)
    task = _tasks.get(job_id)
    if task is not None:
        task.cancel()
    return state.jobs.get(job_id).public()


def _manifest_dir(paths, job_id: str, stage: str) -> Path:
    return paths.home / "forge" / job_id / stage


def _digest_pairs(pairs: list[tuple[str, str]]) -> str:
    """One digest over every input file's recorded sha256, order-independent."""
    lines = "\n".join(f"{path}:{sha}" for path, sha in sorted(pairs))
    return hashlib.sha256(lines.encode()).hexdigest()


def _artifact_pairs(state: AppState, ref: str) -> list[tuple[str, str]]:
    artifact_id, _, version = ref.partition("@")
    manifest = state.artifacts.get(artifact_id, version or None)
    if manifest is None:
        raise DeployError(f"artifact {ref} is missing from the registry")
    return [(f.path, f.sha256) for f in manifest.files]


def _write_manifest(state: AppState, job_id: str, stage: StageKind, plan: ForgePlan, *,
                    engine_version: str, inputs: list[str], outputs: list[str], status: str) -> Path:
    pairs: list[tuple[str, str]] = []
    for ref in inputs:
        pairs.extend(_artifact_pairs(state, ref))
    payload = {
        "stage": stage,
        "engine": _engine_name(plan, stage),
        "engine_version": engine_version,
        "inputs_sha256": _digest_pairs(pairs),
        "outputs": outputs,
        "status": status,
    }
    directory = _manifest_dir(state.paths, job_id, stage)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return path


def _engine_name(plan: ForgePlan, stage: StageKind) -> str:
    if stage == "edit":
        return "heretic" if plan.edit_method == "heretic" else "mergekit"
    if stage == "export":
        from ..engines.llamacpp_release import LLAMACPP_RELEASE

        return "llama.cpp"
    return _STAGE_ENGINE[stage]


def _engine_version(plan: ForgePlan, stage: StageKind, meta: dict[str, Any]) -> str:
    if stage == "export":
        from ..engines.llamacpp_release import LLAMACPP_RELEASE

        return LLAMACPP_RELEASE
    return str(meta.get("engine_version") or "")


def _manifest_ok(paths, raw: str | None) -> bool:
    if not raw:
        return False
    path = Path(raw)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return payload.get("status") == "succeeded" and tuple(payload) == MANIFEST_KEYS


def _persist(state: AppState, job_id: str, results: dict[str, Any], manifests: dict[str, str]) -> None:
    current = state.jobs.get(job_id)
    if current is None:
        return
    state.jobs.update(job_id, spec={**current.spec, "results": results, "manifests": manifests})


def _inputs_for(state: AppState, stage: StageKind, plan: ForgePlan, results: dict[str, Any]) -> list[str]:
    """The artifacts a stage consumes, used both for the manifest digest and for resume checks."""
    if stage == "train":
        return [plan.dataset_artifact] if plan.dataset_artifact else []
    if stage == "eval":
        return [r for r in (results.get("train"), results.get("edit")) if isinstance(r, str)]
    if stage == "edit":
        return [r for r in (results.get("train"), results.get("eval") or plan.eval_artifact)
                if isinstance(r, str)]
    if stage == "export":
        source = results.get("edit") or results.get("train")
        if isinstance(source, str):
            return [source]
        return [plan.base.id] if state.artifacts.get(plan.base.id) is not None else []
    if stage == "deploy_test":
        return [r for r in (results.get("export"),) if isinstance(r, str)]
    return []


def verify_artifacts(state: AppState, refs: list[str]) -> None:
    """Recompute every recorded digest: resume must never build on a corrupted artifact."""
    from ..engines.llamacpp_release import sha256_of

    for ref in refs:
        artifact_id, _, version = ref.partition("@")
        manifest = state.artifacts.get(artifact_id, version or None)
        if manifest is None:
            raise DeployError(f"artifact {ref} is missing; refusing to resume")
        directory = state.artifacts.dir_for(manifest.artifact_id, manifest.version)
        for entry in manifest.files:
            # Recorded paths are relative to the artifact's version directory.
            path = directory / entry.path
            if not path.is_file() or sha256_of(path) != entry.sha256:
                raise DeployError(f"artifact {ref} failed integrity check: {entry.path}; refusing to resume")


def _summary(results: dict[str, Any]) -> str:
    if "eval" in results and "eval_after" in results:
        before = results.get("eval_scores") or {}
        after = results.get("eval_after_scores") or {}
        return (f"Pipeline finished — eval before {before.get('score', 'n/a')} "
                f"({before.get('passed', '?')}/{before.get('total', '?')}), "
                f"after {after.get('score', 'n/a')} ({after.get('passed', '?')}/{after.get('total', '?')})")
    return "Pipeline finished"


async def _after_eval(state: AppState, record: JobRecord, plan: ForgePlan,
                      results: dict[str, Any], manifests: dict[str, str]) -> None:
    """Score the edited model as served, so an edit is never accepted on a before-eval alone."""
    if plan.allow_blind_edit:
        return
    edit_ref = results.get("edit")
    if not isinstance(edit_ref, str):
        raise DeployError("the edit stage produced no artifact to evaluate")
    _check(plan, "eval")
    artifact_id, _, version = edit_ref.partition("@")
    state.jobs.update(record.id, message="eval: scoring the edited model")
    dep = state.deploy.create(DeploymentSpec(
        model=ModelRef(source="artifact", id=artifact_id, variant=version or None,
                       engine=plan.base.engine or "transformers"),
        name=f"{plan.output_name}-eval-after", interface="studio"))
    try:
        await state.deploy.ensure_ready(dep, timeout=1800.0)
        report = await evaluate.run(state, dep, plan.eval_profile, job_id=f"{record.id}-after")
    finally:
        await state.deploy.stop(dep.id)
    results["eval_after"] = report["artifact"]
    results["eval_after_scores"] = {k: report.get(k) for k in ("score", "passed", "total")}
    _persist(state, record.id, results, manifests)
    state.jobs.update(record.id, message="edit: before/after eval recorded")


def _calibration_text(state: AppState, job_id: str, dataset_ref: str) -> Path:
    """Concatenate the dataset's text once; llama-imatrix reads it to weight tensor choices."""
    if not dataset_ref:
        raise DeployError("a quantized export builds its importance matrix from the training dataset; "
                          "attach a dataset artifact or set allow_quick_quant to skip it")
    artifact_id, _, version = dataset_ref.partition("@")
    manifest = state.artifacts.get(artifact_id, version or None)
    if manifest is None:
        raise DeployError(f"dataset artifact {dataset_ref} is missing")
    chunks: list[str] = []
    total = 0
    for entry in manifest.files:
        if total >= CALIBRATION_CHARS:
            break
        try:
            text = Path(entry.path).read_text(errors="replace")
        except OSError:
            continue
        chunks.append(text)
        total += len(text)
    if not chunks:
        raise DeployError(f"dataset artifact {dataset_ref} holds no text to calibrate on")
    directory = _manifest_dir(state.paths, job_id, "export")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "imatrix-calibration.txt"
    path.write_text("".join(chunks)[:CALIBRATION_CHARS])
    return path


def _imatrix_path(state: AppState, convert_job: str) -> Path | None:
    """The matrix the conversion job built, located through its own spec."""
    record = state.jobs.get(convert_job)
    spec_path = ((record.spec or {}).get("spec_path") if record is not None else None) or ""
    if not spec_path:
        return None
    try:
        out_dir = Path(json.loads(Path(spec_path).read_text())["out_dir"])
    except (OSError, ValueError, KeyError):
        return None
    candidate = out_dir / "imatrix.dat"
    return candidate if candidate.is_file() else None


async def _run(state: AppState, record: JobRecord, plan: ForgePlan, stages: list[StageKind],
               *, results: dict[str, Any] | None = None,
               manifests: dict[str, str] | None = None) -> None:
    results = dict(results or {})
    manifests = dict(manifests or {})
    try:
        for index, stage in enumerate(stages):
            if state.jobs.get(record.id).cancel_requested:
                state.jobs.update(record.id, state="cancelled", ended_at=time.time(), message="Cancelled")
                return
            if stage in results and _manifest_ok(state.paths, manifests.get(stage)):
                # Never resume by re-running: a stage with a verified manifest is done.
                state.jobs.update(record.id, progress=(index + 1) / len(stages),
                                  message=f"{stage}: already finished, skipped")
                continue
            _check(plan, stage)
            state.jobs.update(record.id, stage=stage, progress=index / max(len(stages), 1),
                              message=f"{stage}: starting")
            meta: dict[str, Any] = {}
            inputs = _inputs_for(state, stage, plan, results)
            try:
                output = await _stage(state, record, plan, stage, results, meta)
            except BaseException:
                # A stage that dies without a manifest is a bug: write one with status=failed.
                manifests[stage] = str(_write_manifest(state, record.id, stage, plan,
                                                       engine_version=_engine_version(plan, stage, meta),
                                                       inputs=inputs, outputs=[], status="failed"))
                _persist(state, record.id, results, manifests)
                raise
            results[stage] = output
            if meta.get("scores"):
                results[f"{stage}_scores"] = meta["scores"]
            outputs = [output] if isinstance(output, str) else []
            outputs.extend(meta.get("outputs_extra") or [])
            manifests[stage] = str(_write_manifest(state, record.id, stage, plan,
                                                   engine_version=_engine_version(plan, stage, meta),
                                                   inputs=inputs, outputs=outputs,
                                                   status="succeeded"))
            _persist(state, record.id, results, manifests)
            state.jobs.update(record.id, progress=(index + 1) / len(stages), message=f"{stage}: done")
        if "edit" in stages and "eval_after" not in results:
            if state.jobs.get(record.id).cancel_requested:
                state.jobs.update(record.id, state="cancelled", ended_at=time.time(), message="Cancelled")
                return
            await _after_eval(state, record, plan, results, manifests)
        message = _summary(results)
        state.jobs.update(record.id, state="succeeded", ended_at=time.time(), progress=1.0,
                          message=message)
    except asyncio.CancelledError:
        state.jobs.update(record.id, state="cancelled", ended_at=time.time(), message="Cancelled")
        raise
    except DeployError as exc:
        state.jobs.update(record.id, state="failed", ended_at=time.time(), error=str(exc), message=str(exc))
    except Exception as exc:  # noqa: BLE001 - a pipeline must not die silently
        state.jobs.update(record.id, state="failed", ended_at=time.time(), message=f"{type(exc).__name__}: {exc}")
    finally:
        _tasks.pop(record.id, None)


def _job_artifact(state: AppState, job_id: str, kind: str) -> str:
    job = state.jobs.get(job_id)
    if job is None:
        raise DeployError(f"{kind} job {job_id} disappeared")
    for ref in job.artifacts:
        artifact_id, _, version = ref.partition("@")
        manifest = state.artifacts.get(artifact_id, version or None)
        if manifest is not None and manifest.kind == kind:
            return ref
    raise DeployError(f"{kind} job {job_id} finished without a {kind} artifact")


async def _await_job(state: AppState, job_id: str, label: str) -> JobRecord:
    while True:
        record = state.jobs.get(job_id)
        if record is None:
            raise DeployError(f"{label} job {job_id} disappeared")
        if record.terminal:
            if record.state != "succeeded":
                raise DeployError(f"{label} failed: {record.error or record.state}")
            return record
        await asyncio.sleep(POLL_S)


async def _stage(state: AppState, record: JobRecord, plan: ForgePlan, stage: StageKind,
                 results: dict[str, Any], meta: dict[str, Any] | None = None) -> Any:
    meta = meta if meta is not None else {}
    if stage == "train":
        run = TrainingRun(base=plan.base, dataset_artifact=plan.dataset_artifact,
                          method=plan.method, output_name=f"{plan.output_name}-adapter")
        job = await train.start(state, run)
        trained = await _await_job(state, job["id"], "train")
        meta["engine_version"] = (trained.spec or {}).get("engine_version") or ""
        return _job_artifact(state, job["id"], "adapter")
    if stage == "eval":
        dep = state.deploy.get(plan.eval_deployment)
        if dep is None:
            raise DeployError(f"No deployment {plan.eval_deployment}")
        report = await evaluate.run(state, dep, plan.eval_profile, job_id=record.id)
        meta["scores"] = {k: report.get(k) for k in ("score", "passed", "total")}
        return report["artifact"]
    if stage == "edit":
        spec = edit.StudioEditSpec(name=f"{plan.output_name}-edit", source_repo=plan.base.id,
                                   method=plan.edit_method, trials=plan.edit_trials)
        job = await edit.start(state, spec)
        edited = await _await_job(state, job["id"], "edit")
        meta["engine_version"] = (edited.spec or {}).get("engine_version") or ""
        return _job_artifact(state, job["id"], "model")
    if stage == "export":
        source = results.get("edit") or results.get("train")
        if not source:
            base = state.artifacts.get(plan.base.id)
            if base is None:
                raise DeployError("export needs a trained or edited artifact")
            source = f"{base.artifact_id}@{base.version}"
        calibration = None
        if plan.quant != "q8_0" and not plan.allow_quick_quant:
            calibration = _calibration_text(state, record.id, plan.dataset_artifact)
        job = await convert.convert(state, source, plan.quant, out_name=plan.output_name,
                                    calibration=calibration)
        await _await_job(state, job["id"], "export")
        if calibration is not None:
            matrix = _imatrix_path(state, job["id"])
            if matrix is not None:
                meta["outputs_extra"] = [str(matrix)]
        return _job_artifact(state, job["id"], "export")
    if stage == "deploy_test":
        gguf = results.get("export")
        if not gguf:
            raise DeployError("deploy_test needs an export")
        artifact_id, _, version = gguf.partition("@")
        dep = state.deploy.create(DeploymentSpec(
            model=ModelRef(source="artifact", id=artifact_id, variant=version or None, engine="llamacpp"),
            name=f"{plan.output_name}-test", interface="studio"))
        await state.deploy.ensure_ready(dep, timeout=900.0)
        return dep.id
    raise DeployError(f"unknown stage {stage}")
