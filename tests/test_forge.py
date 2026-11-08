"""Model Forge guardrails: run manifests, resume, integrity checks, imatrix and licensing."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from kaggleui.engines.context import DeployError
from kaggleui.ml import pipeline
from kaggleui.ml.pipeline import MANIFEST_KEYS, ForgePlan
from kaggleui.schemas import ArtifactManifest, Settings
from kaggleui.server.state import AppState

REPO = Path(__file__).resolve().parent.parent


def _state(paths) -> AppState:
    return AppState(paths=paths, settings=Settings(), port=8932, mock=True, fake_hw="t4x2")


def _artifact(state, paths, name: str, kind: str, files: dict[str, str]) -> str:
    root = paths.cache / f"src-{name}"
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, text in files.items():
        path = root / filename
        path.write_text(text)
        written.append(path)
    saved = state.artifacts.add(ArtifactManifest(artifact_id=name, version="v1", kind=kind), written,
                                overwrite=True)
    return f"{saved.artifact_id}@{saved.version}"


def _plan(dataset: str, *, stages: list[str], **kwargs) -> ForgePlan:
    return ForgePlan(base={"source": "hf", "id": "acme/tiny", "variant": None, "engine": "transformers"},
                     dataset_artifact=dataset, stages=stages, output_name="tiny-forge", **kwargs)


def _fake_stages(monkeypatch, calls: list[str], outputs: dict[str, str]):
    async def fake(state, record, plan, stage, results, meta=None):
        calls.append(stage)
        if meta is not None and stage == "train":
            meta["engine_version"] = "0.9.1"
        return outputs[stage]

    monkeypatch.setattr(pipeline, "_stage", fake)


async def _run(state, plan, stages):
    record = state.jobs.create(pipeline.FORGE_KIND, {"plan": plan.model_dump(mode="json"),
                                                     "stages": stages, "results": {}, "manifests": {}})
    state.jobs.update(record.id, state="running")
    await pipeline._run(state, record, plan, stages)
    return state.jobs.get(record.id)


async def test_each_stage_writes_a_manifest_with_the_agreed_keys(paths, monkeypatch):
    state = _state(paths)
    dataset = _artifact(state, paths, "data", "dataset", {"train.jsonl": '{"text": "hello"}\n'})
    adapter = _artifact(state, paths, "tiny-adapter", "adapter", {"adapter_model.safetensors": "{}"})
    export = _artifact(state, paths, "tiny-export", "export", {"model-q4_k_m.gguf": "GGUF"})
    calls: list[str] = []
    _fake_stages(monkeypatch, calls, {"train": adapter, "export": export})
    plan = _plan(dataset, stages=["train", "export"])

    record = await _run(state, plan, ["train", "export"])

    assert record.state == "succeeded", record.error
    digests = {}
    for stage in ("train", "export"):
        payload = json.loads((paths.home / "forge" / record.id / stage / "run.json").read_text())
        assert tuple(payload) == MANIFEST_KEYS
        assert payload["stage"] == stage and payload["status"] == "succeeded"
        digests[stage] = payload["inputs_sha256"]
    assert digests["train"] != digests["export"]
    manifest = json.loads((paths.home / "forge" / record.id / "train" / "run.json").read_text())
    assert manifest["engine"] == "trl" and manifest["engine_version"] == "0.9.1"
    assert manifest["outputs"] == [adapter]
    assert record.spec["manifests"]["train"].endswith("run.json")


async def test_resume_skips_finished_stages_and_runs_the_rest(paths, monkeypatch):
    state = _state(paths)
    dataset = _artifact(state, paths, "data", "dataset", {"train.jsonl": '{"text": "hello"}\n'})
    adapter = _artifact(state, paths, "tiny-adapter", "adapter", {"adapter_model.safetensors": "{}"})
    export = _artifact(state, paths, "tiny-export", "export", {"model-q4_k_m.gguf": "GGUF"})
    calls: list[str] = []
    _fake_stages(monkeypatch, calls, {"train": adapter, "export": export})
    plan = _plan(dataset, stages=["train", "export"])
    stages = ["train", "export"]

    failed = state.jobs.create(pipeline.FORGE_KIND, {"plan": plan.model_dump(mode="json"),
                                                     "stages": stages, "results": {"train": adapter},
                                                     "manifests": {}}, resume_supported=True)
    manifest_path = pipeline._write_manifest(state, failed.id, "train", plan, engine_version="0.9.1",
                                             inputs=[dataset], outputs=[adapter], status="succeeded")
    state.jobs.update(failed.id, spec={**failed.spec, "manifests": {"train": str(manifest_path)}},
                      state="failed", error="kernel died")

    resumed = await pipeline.resume(state, failed.id)
    await pipeline._tasks[failed.id]

    assert calls == ["export"]
    final = state.jobs.get(resumed["id"])
    assert final.state == "succeeded", final.error
    assert set(final.spec["results"]) == {"train", "export"}


async def test_resume_refuses_a_corrupted_artifact(paths, monkeypatch):
    state = _state(paths)
    dataset = _artifact(state, paths, "data", "dataset", {"train.jsonl": '{"text": "hello"}\n'})
    adapter = _artifact(state, paths, "tiny-adapter", "adapter", {"adapter_model.safetensors": "{}"})
    calls: list[str] = []
    _fake_stages(monkeypatch, calls, {"export": "whatever@v1"})
    plan = _plan(dataset, stages=["train", "export"])
    record = state.jobs.create(pipeline.FORGE_KIND, {"plan": plan.model_dump(mode="json"),
                                                     "stages": ["train", "export"],
                                                     "results": {"train": adapter}, "manifests": {}},
                               resume_supported=True)
    state.jobs.update(record.id, state="failed")
    stored = state.artifacts.get("tiny-adapter", "v1")
    (state.artifacts.dir_for("tiny-adapter", "v1") / stored.files[0].path).write_text("tampered")

    with pytest.raises(DeployError) as err:
        await pipeline.resume(state, record.id)
    assert "tiny-adapter" in str(err.value) and "integrity" in str(err.value)
    assert calls == []


async def test_export_without_calibration_text_is_refused_at_the_stage(paths, monkeypatch):
    state = _state(paths)
    dataset = _artifact(state, paths, "data", "dataset", {"train.jsonl": '{"text": "hello"}\n'})
    adapter = _artifact(state, paths, "tiny-adapter", "adapter", {"adapter_model.safetensors": "{}"})
    stored = state.artifacts.get("data", "v1")
    (state.artifacts.dir_for("data", "v1") / stored.files[0].path).unlink()
    plan = _plan(dataset, stages=["train", "export"], quant="q4_k_m", allow_quick_quant=False)
    plan.base.id = "acme/tiny"

    with pytest.raises(DeployError) as err:
        await pipeline._stage(state, state.jobs.create(pipeline.FORGE_KIND, {}), plan, "export",
                              {"train": adapter})
    assert "calibrate" in str(err.value)


def test_edit_without_eval_is_refused():
    plan = ForgePlan(base={"source": "hf", "id": "acme/tiny", "variant": None, "engine": "transformers"},
                     stages=["edit", "export"])
    with pytest.raises(DeployError) as err:
        pipeline._check(plan, "edit")
    assert "allow_blind_edit" in str(err.value)


async def test_teacher_names_the_dead_endpoint(paths, monkeypatch):
    import httpx

    from kaggleui.ml import teacher

    class FakeResponse:
        status_code = 503
        text = "model is overloaded"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            assert url.endswith("/dead/teacher")
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    state = _state(paths)
    state.settings.ml.teacher_remote = True

    with pytest.raises(DeployError) as err:
        await teacher.remote_collect(state, "dead/teacher", ["hello"])
    assert "dead/teacher" in str(err.value) and "503" in str(err.value)


def test_heretic_and_mergekit_are_never_imported():
    pattern = re.compile(r"^\s*(?:import|from)\s+(heretic|mergekit)\b", re.MULTILINE)
    offenders = []
    for path in (REPO / "kaggleui").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if pattern.search(text):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == []
    notices = (REPO / "THIRD-PARTY-NOTICES").read_text()
    assert "AGPL-3.0" in notices and "Heretic" in notices
    assert "LGPL-3.0" in notices and "mergekit" in notices
