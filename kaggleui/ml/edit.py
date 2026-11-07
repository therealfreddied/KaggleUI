"""Studio editing: a thin, typed wrapper over the existing abliteration import pipeline."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, field_validator

from ..engines.context import DeployError
from ..schemas import ArtifactManifest
from ..workbench.runner import start_local

if TYPE_CHECKING:
    from ..server.state import AppState

TERMINAL = {"complete", "error", "cancelled"}

class StudioEditSpec(BaseModel):
    name: str = ""
    source_repo: str
    method: Literal["quick", "heretic"] = "quick"
    trials: int = 40
    quantize: bool = False

    @field_validator("trials")
    @classmethod
    def _trials(cls, value: int) -> int:
        if not 10 <= value <= 300:
            raise ValueError("trials must be between 10 and 300")
        return value

async def start(state: AppState, spec: StudioEditSpec) -> dict:
    try:
        imported = state.imports.abliterate(spec.source_repo, method=spec.method, trials=spec.trials,
                                            quantize=spec.quantize, name=spec.name)
    except ValueError as exc:
        raise DeployError(str(exc)) from exc
    job = state.jobs.create("edit", {"edit": spec.model_dump(mode="json"), "import_job": imported["id"]},
                            logs_path=str(state.paths.logs / f"edit-{imported['id']}.log"))

    async def work(ctx) -> None:
        ctx.log(f"abliterating {spec.source_repo} ({spec.method}, {spec.trials} trials)")
        while True:
            record = state.imports.jobs.get(imported["id"], {})
            status = record.get("status")
            if status in TERMINAL:
                break
            await asyncio.sleep(2)
            ctx.progress(record.get("progress") or 0.0, record.get("message") or status or "working")
        if status != "complete":
            raise DeployError(record.get("message") or f"import {imported['id']} ended as {status}")
        models = record.get("models") or []
        if not models:
            raise DeployError("the edit finished without a model directory")
        directory = Path(models[0]["path"])
        files = [p for p in directory.rglob("*") if p.is_file()]
        if not files:
            raise DeployError(f"no files under {directory}")
        saved = state.artifacts.add(ArtifactManifest(
            artifact_id=f"model-{record.get('name') or directory.name}", version=time.strftime("e%Y%m%d-%H%M%S"),
            kind="model", created_at=time.time(),
            recipe={"edit": spec.model_dump(mode="json")}, provenance={"import": imported["id"]}), files,
            overwrite=True)
        ctx.artifact(saved.model_dump())

    await start_local(state.jobs, job, work)
    return state.jobs.get(job.id).public()

