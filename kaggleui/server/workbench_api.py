"""Workbench control plane: durable jobs, deployment sets and placement previews.

Every route is admin-only: these endpoints start installs, write config and touch model storage, which
is exactly what ``require_admin`` protects elsewhere in the panel.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import ValidationError

from ..gateway.common import state_of
from ..schemas import ArtifactManifest, DeploymentSet, DeploymentSpec, ModelRef, StorageConnection
from ..storage import StorageError, redact
from ..util import short_id
from ..workbench import placement as place
from ..workbench.jobs import read_log
from ..workbench.registry import RegistryError
from ..workbench.runner import start_in_background, start_local

router = APIRouter(prefix="/_kui/api", tags=["workbench"])


def _admin(request: Request):
    st = state_of(request)
    if not st.auth.is_admin(request):
        raise HTTPException(401, "Admin token required")
    return st


# -- jobs -------------------------------------------------------------------------------------
@router.get("/jobs")
def list_jobs(request: Request, kind: str | None = None, state: str | None = None, limit: int = 50):
    st = _admin(request)
    records = st.jobs.list(limit=limit, kind=kind)
    if state:
        records = [r for r in records if r.state == state]
    return {"jobs": [r.public() for r in records]}


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str):
    st = _admin(request)
    record = st.jobs.get(job_id)
    if record is None:
        raise HTTPException(404, "No such job")
    return record.public()


@router.get("/jobs/{job_id}/logs")
def job_logs(request: Request, job_id: str, offset: int = 0, limit: int = 400):
    st = _admin(request)
    record = st.jobs.get(job_id)
    if record is None:
        raise HTTPException(404, "No such job")
    return {"job": record.public(),
            "lines": read_log(record.logs_path, max(0, offset), max(1, min(limit, 2000)))}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str):
    st = _admin(request)
    record = st.jobs.get(job_id)
    if record is None:
        raise HTTPException(404, "No such job")
    if record.terminal:
        return record.public()
    return (await st.jobs.cancel(job_id)).public()


@router.post("/jobs/{job_id}/resume")
def resume_job(request: Request, job_id: str):
    st = _admin(request)
    record = st.jobs.get(job_id)
    if record is None:
        raise HTTPException(404, "No such job")
    if not record.resume_supported:
        raise HTTPException(409, "This job cannot be resumed; start it again from the panel")
    cmd = record.spec.get("cmd")
    if not isinstance(cmd, list) or not cmd:
        raise HTTPException(409, "This job has no recorded command")
    fresh = st.jobs.create(record.kind, dict(record.spec), resume_supported=True, logs_path=record.logs_path)
    start_in_background(st.jobs, fresh, [str(part) for part in cmd], log_path=record.logs_path,
                        name=f"{record.kind}-resume")
    return fresh.public()


# -- deployment sets --------------------------------------------------------------------------
@router.get("/workbench/deployment-sets")
def list_sets(request: Request):
    st = _admin(request)
    return {"sets": [s.model_dump() for s in st.workbench.deployment_sets()], "active": st.workbench.active_set}


@router.post("/workbench/deployment-sets")
def put_set(request: Request, body: dict[str, Any]):
    st = _admin(request)
    try:
        deployment_set = DeploymentSet.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors()[0]["msg"]) from exc
    saved = st.workbench.put_deployment_set(deployment_set)
    return saved.model_dump()


@router.delete("/workbench/deployment-sets/{set_id}")
def delete_set(request: Request, set_id: str):
    st = _admin(request)
    st.workbench.delete_deployment_set(set_id)
    return {"ok": True}


@router.post("/workbench/deployment-sets/{set_id}/activate")
def activate_set(request: Request, set_id: str):
    st = _admin(request)
    if st.workbench.deployment_set(set_id) is None:
        raise HTTPException(404, "No such deployment set")
    st.workbench.mark_active(set_id)
    return {"active": set_id}


# -- optimization settings --------------------------------------------------------------------
@router.get("/workbench/optimize")
def get_optimize(request: Request):
    st = _admin(request)
    return {"optimize": st.settings.optimize.model_dump(), "cache": st.settings.cache.model_dump()}


@router.put("/workbench/optimize")
def put_optimize(request: Request, body: dict[str, Any]):
    """Merge request-shaping (`optimize`) and cache settings; both are opt-in and off by default."""
    st = _admin(request)
    optimize_patch = body.pop("optimize", None)
    cache_patch = body.pop("cache", body)  # flat bodies are cache-shaped for older callers
    try:
        optimize = st.settings.optimize.model_validate(
            {**st.settings.optimize.model_dump(), **(optimize_patch or {})})
        cache = st.settings.cache.model_validate({**st.settings.cache.model_dump(), **cache_patch})
    except ValidationError as exc:
        raise HTTPException(422, exc.errors()[0]["msg"]) from exc
    if cache.semantic:
        ready = _embedding_deployment(st)
        if ready is None:
            raise HTTPException(400, "Semantic caching needs a ready text deployment with embeddings enabled "
                                     "(options: {\"embeddings\": true}); deploy one first or use exact caching")
        if not cache.embedding_model:
            cache = cache.model_copy(update={"embedding_model": ready.name})
    st.settings.optimize = optimize
    st.settings.cache = cache
    st.bus.publish("settings", st.settings.model_dump())
    return {"optimize": optimize.model_dump(), "cache": cache.model_dump()}


def _embedding_deployment(st):
    for dep in st.manager.deployments.values():
        if dep.status == "ready" and dep.spec.modality == "text" and dep.spec.options.get("embeddings"):
            return dep
    return None


# -- placement preview ------------------------------------------------------------------------
@router.post("/workbench/placement/preview")
def preview_placement(request: Request, body: dict[str, Any]):
    """Explain what a spec's placement will do without downloading or launching anything."""
    st = _admin(request)
    try:
        spec = DeploymentSpec.model_validate(body.get("spec") or body)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors()[0]["msg"]) from exc
    arch = None
    raw_path = body.get("gguf")
    if raw_path:
        from ..gguf import GGUFError, read_metadata

        try:
            arch = read_metadata(raw_path)
        except GGUFError as exc:
            raise HTTPException(422, f"Not a readable GGUF: {exc}") from exc
    hw = st.probe.snapshot()
    errors = place.validate(spec, hw, arch=arch)
    resolved = place.resolve(spec, hw, arch=arch, assigned=_assigned(spec, hw))
    return {"ok": not errors, "errors": errors, "resolved": resolved.public()}


def _assigned(spec: DeploymentSpec, hw) -> list[int]:
    if spec.placement is not None and spec.placement.devices:
        return list(spec.placement.devices)
    return [g.index for g in hw.gpus]


# -- workbench snapshot -----------------------------------------------------------------------
@router.get("/workbench")
def snapshot(request: Request):
    st = _admin(request)
    return {
        "active_set": st.workbench.active_set,
        "sets": [s.model_dump() for s in st.workbench.deployment_sets()],
        "connections": [c.model_dump() for c in st.workbench.connections()],
        "deployments": [r.model_dump() for r in st.workbench.deployment_records()],
        "storage": st.settings.storage.model_dump(),
        "optimize": st.settings.optimize.model_dump(),
        "cache": st.settings.cache.model_dump(),
        "cache_stats": st.optimize.stats()["cache"],
        "jobs": [r.public() for r in st.jobs.list(limit=20)],
    }


# -- storage connections ----------------------------------------------------------------------
@router.get("/workbench/storage")
def list_storage(request: Request):
    st = _admin(request)
    return {"connections": st.storage.public()}


@router.post("/workbench/storage")
def save_storage(request: Request, body: dict[str, Any]):
    st = _admin(request)
    try:
        connection = StorageConnection.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors()[0]["msg"]) from exc
    if connection.kind == "local" and not connection.config.get("root"):
        raise HTTPException(422, "local connections need config.root")
    return redact(st.storage.save(connection))


@router.delete("/workbench/storage/{name}")
def delete_storage(request: Request, name: str):
    st = _admin(request)
    try:
        st.storage.delete(name, st.artifacts.list())
    except StorageError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@router.post("/workbench/storage/{name}/test")
async def test_storage(request: Request, name: str):
    st = _admin(request)
    try:
        return await st.storage.test(name)
    except StorageError as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/workbench/storage/{name}/config")
async def upload_rclone_config(request: Request, name: str, file: UploadFile = File(...)):
    """Store an ``rclone.conf`` for this connection (mode 600, never returned to the panel)."""
    st = _admin(request)
    connection = st.storage.workbench.connection(name)
    if connection is None or connection.kind != "rclone":
        raise HTTPException(404, "No rclone connection by that name")
    payload = await file.read()
    if len(payload) > 64 << 10:
        raise HTTPException(413, "rclone.conf is larger than 64 KiB")
    target = st.paths.home / "rclone.conf"
    target.write_bytes(payload)
    os.chmod(target, 0o600)
    connection = connection.model_copy(update={"config": {**connection.config, "config_file": str(target)}})
    st.storage.save(connection)
    return {"ok": True, "config_file": str(target)}


# -- artifacts --------------------------------------------------------------------------------
@router.get("/workbench/artifacts")
def list_artifacts(request: Request):
    st = _admin(request)
    return {"artifacts": st.artifacts.public()}


@router.get("/workbench/artifacts/{artifact_id}")
def get_artifact(request: Request, artifact_id: str, version: str | None = None):
    st = _admin(request)
    manifest = st.artifacts.get(artifact_id, version)
    if manifest is None:
        raise HTTPException(404, f"no artifact {artifact_id}")
    return manifest.model_dump()


@router.post("/workbench/artifacts/{artifact_id}/verify")
def verify_artifact(request: Request, artifact_id: str, body: dict[str, Any] | None = None):
    st = _admin(request)
    return st.artifacts.verify(artifact_id, (body or {}).get("version"))


@router.post("/workbench/artifacts/{artifact_id}/export")
async def export_artifact(request: Request, artifact_id: str, body: dict[str, Any]):
    st = _admin(request)
    manifest = st.artifacts.get(artifact_id, body.get("version"))
    if manifest is None:
        raise HTTPException(404, f"no artifact {artifact_id}")
    connection = st.storage.workbench.connection(body.get("connection", ""))
    if connection is None:
        raise HTTPException(404, "No such storage connection")
    if connection.kind == "hf" and not manifest.license and not body.get("confirm"):
        raise HTTPException(400, "this artifact has no license; pass confirm=true to publish it anyway")
    if connection.read_only:
        raise HTTPException(400, f"{connection.name} is marked read-only")
    target = body.get("target_path") or f"{manifest.artifact_id}/{manifest.version}"
    job = st.jobs.create("artifact.export", {"artifact_id": artifact_id, "connection": connection.name,
                                             "target_path": target}, logs_path=str(_job_log(st, "export")))
    base = st.artifacts.dir_for(manifest.artifact_id, manifest.version)
    files = [base / entry.path for entry in manifest.files] + [base / "manifest.json"]

    async def work(ctx):
        ctx.log(f"exporting {artifact_id} {manifest.version} to {connection.name}:{target}")
        for index, path in enumerate(files):
            remote = f"{target.rstrip('/')}/{path.relative_to(base)}"
            await st.storage.upload(connection.name, path, remote,
                                    progress=_step(ctx, index, len(files), remote))
        ctx.artifact({"artifact_id": manifest.artifact_id, "version": manifest.version,
                      "connection": connection.name, "path": target})

    await start_local(st.jobs, job, work)
    return st.jobs.get(job.id).public()


@router.post("/workbench/artifacts/{artifact_id}/delete")
def delete_artifact(request: Request, artifact_id: str, body: dict[str, Any]):
    st = _admin(request)
    version = body.get("version")
    if not version:
        raise HTTPException(422, "version is required")
    try:
        st.artifacts.remove(artifact_id, version, confirm=bool(body.get("confirm")))
    except RegistryError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


@router.post("/workbench/restore")
async def restore_artifact(request: Request, body: dict[str, Any]):
    st = _admin(request)
    connection = st.storage.workbench.connection(body.get("connection", ""))
    if connection is None:
        raise HTTPException(404, "No such storage connection")
    remote = str(body.get("remote") or "").strip()
    name = str(body.get("name") or "").strip()
    kind = str(body.get("kind") or "model")
    if not remote or not name:
        raise HTTPException(422, "remote and name are required")
    job = st.jobs.create("artifact.restore", {"connection": connection.name, "remote": remote, "name": name,
                                              "kind": kind}, logs_path=str(_job_log(st, "restore")))

    async def work(ctx):
        entries = await st.storage.list(connection.name, remote)
        # manifest.json is provenance the registry regenerates, never artifact payload.
        skip = f"{remote.rstrip('/')}/manifest.json"
        files = [e for e in entries if not e.is_dir and e.path != skip and e.path != "manifest.json"]
        if not files:
            raise StorageError(f"nothing to download under {remote}")
        scratch = st.paths.cache / "restore" / job.id
        scratch.mkdir(parents=True, exist_ok=True)
        ctx.log(f"downloading {len(files)} file(s) from {connection.name}:{remote}")
        local_files = []
        for index, entry in enumerate(files):
            dest = scratch / Path(entry.path).relative_to(remote)
            await st.storage.download(connection.name, entry.path, dest,
                                      progress=_step(ctx, index, len(files) + 1, entry.path))
            local_files.append(dest)
        ctx.progress(len(files) / (len(files) + 1), "registering artifact")
        manifest = ArtifactManifest(artifact_id=name, version=time.strftime("%Y%m%d-%H%M%S"), kind=kind,
                                    provenance={"connection": connection.name, "remote": remote})
        saved = st.artifacts.add(manifest, local_files)
        ctx.artifact(saved.model_dump())

    await start_local(st.jobs, job, work)
    return st.jobs.get(job.id).public()


def _step(ctx, index: int, total: int, label: str):
    """Progress callback for one file in a multi-file transfer."""
    return lambda frac, line: ctx.progress((index + frac) / total, line or label)


def _job_log(st, prefix: str) -> Path:
    return st.paths.logs / f"{prefix}-{short_id()}.log"


# -- Model Studio: datasets --------------------------------------------------------------------
def _safe_dataset_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "-" for c in name).strip(".-")
    return cleaned or "dataset"


async def _read_upload(upload: UploadFile, *, limit: int = 64 << 20) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, "dataset is larger than 64 MiB; split it or host it remotely")
    return data


@router.get("/workbench/ml/datasets")
def list_datasets(request: Request):
    st = _admin(request)
    return {"datasets": [m.model_dump() for m in st.artifacts.list() if m.kind == "dataset"]}


@router.post("/workbench/ml/datasets")
async def upload_dataset(request: Request, file: UploadFile = File(...), name: str = "", format: str = ""):
    from ..ml.datasets import validate

    st = _admin(request)
    data = await _read_upload(file)
    label = _safe_dataset_name(name or Path(file.filename or "dataset").stem)
    scratch = st.paths.cache / "datasets" / short_id()
    scratch.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "dataset.jsonl").suffix or ".jsonl"
    path = scratch / f"{label}{suffix}"
    path.write_bytes(data)
    report = validate(path, format or None)
    if not report.ok:
        raise HTTPException(422, {"detail": "dataset failed validation", "report": report.model_dump()})
    manifest = ArtifactManifest(artifact_id=f"dataset-{label}", version=time.strftime("d%Y%m%d-%H%M%S"),
                                kind="dataset", created_at=time.time(),
                                dataset={"format": report.detected_format, "samples": report.samples,
                                         "token_estimate": report.token_estimate,
                                         "warnings": report.warnings, "source_name": file.filename})
    saved = st.artifacts.add(manifest, [path], overwrite=True)
    return {"artifact": f"{saved.artifact_id}@{saved.version}", "report": report.model_dump(),
            "manifest": saved.model_dump()}


@router.post("/workbench/ml/datasets/validate")
async def validate_dataset(request: Request, file: UploadFile = File(...), format: str = ""):
    from ..ml.datasets import validate

    st = _admin(request)
    scratch = st.paths.cache / "datasets" / f"validate-{short_id()}"
    suffix = Path(file.filename or "dataset.jsonl").suffix or ".jsonl"
    path = scratch / f"upload{suffix}"
    scratch.mkdir(parents=True, exist_ok=True)
    path.write_bytes(await _read_upload(file))
    try:
        return validate(path, format or None, limit=500).model_dump()
    finally:
        path.unlink(missing_ok=True)


# -- Model Studio: training --------------------------------------------------------------------
def _text_deployment(st, name: str):
    dep = st.manager.resolve(name, "text")
    if dep is None:
        raise HTTPException(404, f"No text deployment answers to {name!r}")
    return dep


@router.post("/workbench/ml/train")
async def start_training(request: Request):
    from ..ml.recipes import TrainingRun
    from ..ml.train import start

    st = _admin(request)
    body = await request.json()
    try:
        run = TrainingRun.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        return await start(st, run)
    except Exception as exc:  # DeployError and download failures are user-facing
        raise HTTPException(400, str(exc)) from exc


@router.get("/workbench/ml/train")
def list_training(request: Request):
    st = _admin(request)
    return {"jobs": [r.public() for r in st.jobs.list(limit=50, kind="train")]}


@router.post("/workbench/ml/train/{job_id}/cancel")
async def cancel_training(request: Request, job_id: str):
    st = _admin(request)
    record = st.jobs.get(job_id)
    if record is None or record.kind != "train":
        raise HTTPException(404, "No such training job")
    return (await st.jobs.cancel(job_id)).public()


@router.post("/workbench/ml/train/{job_id}/resume")
def resume_training(request: Request, job_id: str):
    from ..engines.context import DeployError
    from ..ml.train import resume

    st = _admin(request)
    try:
        return resume(st, job_id)
    except DeployError as exc:
        raise HTTPException(409, str(exc)) from exc


# -- Model Studio: harness ---------------------------------------------------------------------
@router.get("/workbench/ml/harness")
def list_harness(request: Request):
    from ..ml.harness import profiles

    _admin(request)
    return {"profiles": [{"name": p.name, "version": p.version, "description": p.description,
                          "cases": len(p.cases),
                          "buildable": any(c.gold for c in p.cases)} for p in profiles()]}


@router.post("/workbench/ml/harness")
async def run_harness(request: Request):
    from ..ml.evaluate import run

    st = _admin(request)
    body = await request.json()
    dep = _text_deployment(st, str(body.get("deployment") or ""))
    report = await run(st, dep, str(body.get("profile") or "chat_basic"))
    return report


@router.get("/workbench/ml/harness/{artifact_id}")
def get_eval(request: Request, artifact_id: str):
    st = _admin(request)
    name, _, version = artifact_id.partition("@")
    manifest = st.artifacts.get(name, version or None)
    if manifest is None or manifest.kind != "export" or not manifest.eval_summary:
        raise HTTPException(404, "No such evaluation")
    return manifest.model_dump()


# -- Model Studio: teacher ---------------------------------------------------------------------
@router.post("/workbench/ml/teacher")
async def start_teacher(request: Request):
    from ..ml.teacher import check_collision, collect, validate_samples

    st = _admin(request)
    body = await request.json()
    prompts = body.get("prompts") or []
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise HTTPException(422, "prompts must be a non-empty list of strings")
    dep = _text_deployment(st, str(body.get("deployment") or ""))
    student = str(body.get("student_name") or "")
    if student:
        try:
            check_collision(dep.served_name, student)
        except Exception as exc:
            raise HTTPException(409, str(exc)) from exc
    name = _safe_dataset_name(str(body.get("name") or f"teacher-{dep.served_name}"))
    log_path = _job_log(st, "teacher")
    job = st.jobs.create("teacher", {"deployment": dep.id, "prompts": len(prompts), "name": name},
                         logs_path=str(log_path))

    async def work(ctx):
        ctx.progress(0.0, f"asking {dep.served_name} {len(prompts)} prompts")
        result = await collect(dep, prompts, max_new_tokens=int(body.get("max_new_tokens") or 512),
                               temperature=float(body.get("temperature") or 0.7),
                               budget_tokens=int(body.get("budget_tokens") or st.settings.ml.max_teacher_tokens),
                               progress=lambda frac, line: ctx.progress(0.05 + frac * 0.8, line))
        problems = validate_samples(result["samples"])
        if problems:
            raise RuntimeError("; ".join(problems[:3]))
        scratch = st.paths.cache / "datasets" / f"teacher-{job.id}"
        scratch.mkdir(parents=True, exist_ok=True)
        path = scratch / f"{name}.jsonl"
        import json as _json

        path.write_text("\n".join(_json.dumps(row, ensure_ascii=False) for row in result["samples"]) + "\n")
        ctx.progress(0.95, "registering dataset")
        manifest = ArtifactManifest(artifact_id=f"dataset-{name}", version=time.strftime("d%Y%m%d-%H%M%S"),
                                    kind="dataset", created_at=time.time(),
                                    dataset={"samples": len(result["samples"]), "tokens": result["tokens"],
                                             "teacher": result["teacher"], "source_name": "teacher"},
                                    provenance={"teacher": dep.id, "job": job.id})
        saved = st.artifacts.add(manifest, [path], overwrite=True)
        ctx.artifact(saved.model_dump())

    await start_local(st.jobs, job, work)
    return st.jobs.get(job.id).public()


@router.get("/workbench/ml/teacher")
def list_teacher(request: Request):
    st = _admin(request)
    return {"jobs": [r.public() for r in st.jobs.list(limit=50, kind="teacher")],
            "datasets": [m.model_dump() for m in st.artifacts.list()
                         if m.kind == "dataset" and m.dataset.get("teacher")]}


@router.post("/workbench/ml/teacher/{job_id}/cancel")
async def cancel_teacher(request: Request, job_id: str):
    st = _admin(request)
    record = st.jobs.get(job_id)
    if record is None or record.kind != "teacher":
        raise HTTPException(404, "No such teacher job")
    return (await st.jobs.cancel(job_id)).public()



# -- Model Studio: merge, quantize, edit --------------------------------------------------------
@router.post("/workbench/ml/merge")
async def merge_adapter(request: Request):
    from ..ml.convert import merge

    st = _admin(request)
    body = await request.json()
    try:
        base = ModelRef.model_validate(body.get("base") or {})
        return await merge(st, base, str(body.get("adapter") or ""), name=str(body.get("name") or ""))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/workbench/ml/quantize")
def quantize_model(request: Request, body: dict[str, Any]):
    from ..ml.convert import quantize

    st = _admin(request)
    try:
        return quantize(st, str(body.get("model") or ""), str(body.get("quant") or ""),
                        name=str(body.get("name") or ""))
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/workbench/ml/convert")
def list_convert(request: Request):
    st = _admin(request)
    return {"jobs": [r.public() for r in st.jobs.list(limit=50, kind="convert")],
            "artifacts": [m.model_dump() for m in st.artifacts.list() if m.kind in ("adapter", "model")]}

@router.post("/workbench/ml/convert")
async def start_convert(request: Request):
    from ..ml.convert import convert

    st = _admin(request)
    body = await request.json()
    try:
        return await convert(st, str(body.get("artifact_id") or ""), str(body.get("quant") or ""),
                             out_name=str(body.get("out_name") or ""))
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

@router.get("/workbench/ml/edit")
def list_edit(request: Request):
    st = _admin(request)
    return {"jobs": [r.public() for r in st.jobs.list(limit=50, kind="edit")]}

@router.post("/workbench/ml/edit")
async def studio_edit(request: Request):
    from ..ml.edit import StudioEditSpec, start

    st = _admin(request)
    try:
        spec = StudioEditSpec.model_validate(await request.json())
        return await start(st, spec)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


