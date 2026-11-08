"""Chain the existing Model Studio jobs into one resumable pipeline.

Each stage is a normal job (train, abliteration import, GGUF export); the pipeline just starts the
next one with the artifact the previous stage produced and waits for its terminal state. Stages that
already have a matching artifact in the job record are skipped, so a failed plan can be restarted.
"""

from __future__ import annotations

import asyncio
import time
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
    allow_quick_quant: bool = True
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
    if stage == "export" and plan.quant != "q8_0" and not plan.allow_quick_quant:
        raise DeployError("quantized export needs an importance matrix, which this pipeline does not "
                          "build; set allow_quick_quant to accept the quick conversion")
    if stage == "deploy_test" and plan.base.source == "ollama":
        raise DeployError("deploy_test supports Hugging Face and artifact models only")


async def start(state: AppState, plan: ForgePlan) -> dict:
    stages = _order(plan.stages)
    for stage in stages:
        _check(plan, stage)
    if not plan.output_name:
        plan.output_name = f"{plan.base.id.split('/')[-1]}-forge"
    record = state.jobs.create(FORGE_KIND, {"plan": plan.model_dump(mode="json"), "stages": stages,
                                            "results": {}, "stage": ""})
    state.jobs.update(record.id, state="running", started_at=time.time(), message="Starting")
    _tasks[record.id] = asyncio.create_task(_run(state, record, plan, stages))
    return state.jobs.get(record.id).public()


async def cancel(state: AppState, job_id: str) -> dict:
    record = state.jobs.get(job_id)
    if record is None or record.kind != FORGE_KIND:
        raise DeployError("No such pipeline job")
    state.jobs.update(job_id, cancel_requested=1)
    task = _tasks.get(job_id)
    if task is not None:
        task.cancel()
    return state.jobs.get(job_id).public()


async def _run(state: AppState, record: JobRecord, plan: ForgePlan, stages: list[StageKind]) -> None:
    results: dict[str, Any] = {}
    try:
        for index, stage in enumerate(stages):
            if state.jobs.get(record.id).cancel_requested:
                state.jobs.update(record.id, state="cancelled", ended_at=time.time(), message="Cancelled")
                return
            _check(plan, stage)
            state.jobs.update(record.id, stage=stage, progress=index / max(len(stages), 1),
                              message=f"{stage}: starting")
            results[stage] = await _stage(state, record, plan, stage, results)
            state.jobs.update(record.id, progress=(index + 1) / len(stages),
                              message=f"{stage}: done", spec={**record.spec, "results": results})
        state.jobs.update(record.id, state="succeeded", ended_at=time.time(), progress=1.0,
                          message="Pipeline finished", spec={**record.spec, "results": results})
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
                 results: dict[str, Any]) -> Any:
    if stage == "train":
        run = TrainingRun(base=plan.base, dataset_artifact=plan.dataset_artifact,
                          method=plan.method, output_name=f"{plan.output_name}-adapter")
        job = await train.start(state, run)
        await _await_job(state, job["id"], "train")
        return _job_artifact(state, job["id"], "adapter")
    if stage == "eval":
        dep = state.deploy.get(plan.eval_deployment)
        if dep is None:
            raise DeployError(f"No deployment {plan.eval_deployment}")
        report = await evaluate.run(state, dep, plan.eval_profile, job_id=record.id)
        return report["artifact"]
    if stage == "edit":
        spec = edit.StudioEditSpec(name=f"{plan.output_name}-edit", source_repo=plan.base.id,
                                   method=plan.edit_method, trials=plan.edit_trials)
        job = await edit.start(state, spec)
        await _await_job(state, job["id"], "edit")
        return _job_artifact(state, job["id"], "model")
    if stage == "export":
        source = results.get("edit") or results.get("train")
        if not source:
            base = state.artifacts.get(plan.base.id)
            if base is None:
                raise DeployError("export needs a trained or edited artifact")
            source = f"{base.artifact_id}@{base.version}"
        job = await convert.convert(state, source, plan.quant, out_name=plan.output_name)
        await _await_job(state, job["id"], "export")
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
