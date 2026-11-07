"""Model Studio: recipes, checkpoints, harness grading, and the HTTP surface."""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from kaggleui.ml.harness import EvalCase, evaluate, find_profile, grade
from kaggleui.ml.hub import latest_checkpoint
from kaggleui.ml.recipes import TrainingRun, estimate_vram, worker_argv
from kaggleui.schemas import DeploymentSpec, Settings
from kaggleui.server.app import create_app
from kaggleui.server.state import AppState


def _run(**extra):
    data = {"base": {"source": "hf", "id": "Qwen/Qwen2.5-0.5B-Instruct", "engine": "transformers"},
            "dataset_artifact": "dataset-tiny@d1", **extra}
    return TrainingRun.model_validate(data)


def test_recipe_defaults_and_validation():
    run = _run()
    assert (run.r, run.alpha, run.epochs, run.seq_length, run.lr) == (16, 32, 1.0, 2048, 2e-4)
    assert run.output_name == "Qwen2.5-0.5B-Instruct-dataset-tiny-lora"
    from pathlib import Path

    spec = run.worker_spec(dataset_path=Path("d.jsonl"), model_path=Path("m"), out_dir=Path("o"),
                           scratch=Path("."))
    assert spec["lora"]["target_modules"][0] == "q_proj"
    assert spec["sft"]["per_device_train_batch_size"] == 1
    assert spec["sft"]["gradient_accumulation_steps"] == 16
    assert worker_argv("s.json")[-2:] == ["--spec", "s.json"]
    for bad in ({"r": 3}, {"r": 260}, {"epochs": 0}, {"epochs": 11}, {"seq_length": 64},
                {"seq_length": 9000}, {"lr": 0}, {"lr": -1}):
        with pytest.raises(ValidationError):
            _run(**bad)


def test_gguf_base_is_refused_with_the_export_hint():
    with pytest.raises(ValidationError) as info:
        TrainingRun.model_validate({"base": {"source": "hf", "id": "Qwen/Qwen2.5-0.5B", "engine": "llamacpp",
                                             "variant": "Q4_K_M"}, "dataset_artifact": "dataset-a@d1"})
    assert "Use the GGUF export, not training" in str(info.value)


def test_qlora_estimates_less_vram_than_lora():
    lora = estimate_vram(_run(), params_b=0.5)
    qlora = estimate_vram(_run(method="qlora"), params_b=0.5)
    assert 0 < qlora < lora


def test_latest_checkpoint_requires_an_adapter(tmp_path):
    for step, keep in ((100, False), (300, True), (200, True)):
        directory = tmp_path / f"checkpoint-{step}"
        directory.mkdir()
        if keep:
            (directory / "adapter_model.safetensors").write_text("x")
    assert latest_checkpoint(tmp_path).name == "checkpoint-300"
    (tmp_path / "checkpoint-300" / "adapter_model.safetensors").unlink()
    assert latest_checkpoint(tmp_path).name == "checkpoint-200"
    assert latest_checkpoint(tmp_path / "missing") is None


@pytest.mark.parametrize(
    ("case", "answer", "calls", "passed"),
    [
        (EvalCase(id="e", messages=[], grader="exact", expect=["4"]), "4", [], True),
        (EvalCase(id="e", messages=[], grader="exact", expect=["4"]), " 4 ", [], True),
        (EvalCase(id="e", messages=[], grader="contains_any", expect=["paris"]), "Paris.", [], True),
        (EvalCase(id="e", messages=[], grader="contains_any", expect=["paris"]), "Lyon", [], False),
        (EvalCase(id="e", messages=[], grader="regex", expect=[r"\b\d{3}\b"]), "code 404", [], True),
        (EvalCase(id="e", messages=[], grader="json_valid", required_keys=["a"]), '{"a": 1}', [], True),
        (EvalCase(id="e", messages=[], grader="json_valid", required_keys=["a"]), '{"b": 1}', [], False),
        (EvalCase(id="e", messages=[], grader="json_valid", required_keys=[]), "[]", [], False),
        (EvalCase(id="e", messages=[], grader="tool_call_name", expect=["get_weather"]), "", [
            {"function": {"name": "get_weather"}}], True),
        (EvalCase(id="e", messages=[], grader="tool_call_name", expect=["get_weather"]), "", [], False),
    ],
)
def test_graders(case, answer, calls, passed):
    assert grade(case, answer, calls)[0] is passed


def test_builtin_profiles_have_the_planned_sizes():
    sizes = {p.name: len(p.cases) for p in (find_profile(n) for n in ("chat_basic", "json_output", "tool_choice"))}
    assert sizes == {"chat_basic": 20, "json_output": 10, "tool_choice": 8}


async def test_evaluate_scores_a_mock_deployment(paths):
    state = AppState(paths=paths, settings=Settings(), port=8920, mock=True, fake_hw="t4x2")
    dep = state.manager.create(_spec())
    deadline = asyncio.get_running_loop().time() + 20
    while dep.status not in ("ready", "error") and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    assert dep.status == "ready", dep.message
    report = await evaluate(dep, find_profile("json_output"))
    assert report.total == 10 and 0.0 <= report.score <= 1.0


def _spec() -> DeploymentSpec:
    return DeploymentSpec.model_validate({
        "modality": "text", "model": {"source": "hf", "id": "mock/text", "engine": "mock"},
        "options": {"vram_estimate": 1_000_000_000}})


@pytest.fixture()
def client(paths):
    state = AppState(paths=paths, settings=Settings(), port=8921, mock=True, fake_hw="t4x2")
    app = create_app(state)
    with TestClient(app) as test:
        state.settings.api.require_key = False
        test.headers = {"Authorization": f"Bearer {state.auth.admin_token}"}
        yield test, state


def _dataset_bytes():
    rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i}"}]} for i in range(3)]
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def test_upload_validate_and_training_spec_routes(client):
    test, state = client
    bad = b'{"messages": [{"role": "assistant", "content": "x"}]}\n'
    response = test.post("/_kui/api/workbench/ml/datasets/validate",
                         files={"file": ("bad.jsonl", io.BytesIO(bad), "application/jsonl")})
    assert response.status_code == 200 and response.json()["ok"] is False

    response = test.post("/_kui/api/workbench/ml/datasets?name=tiny",
                         files={"file": ("tiny.jsonl", io.BytesIO(_dataset_bytes()), "application/jsonl")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["report"]["ok"] and body["report"]["samples"] == 3
    assert body["artifact"].startswith("dataset-tiny@")
    listed = test.get("/_kui/api/workbench/ml/datasets").json()["datasets"]
    assert [m["artifact_id"] for m in listed] == ["dataset-tiny"]

    response = test.post("/_kui/api/workbench/ml/train", json={"base": {}, "dataset_artifact": "x"})
    assert response.status_code == 422
    response = test.post("/_kui/api/workbench/ml/train", json={
        "base": {"source": "ollama", "id": "qwen3", "engine": "ollama"}, "dataset_artifact": body["artifact"]})
    assert response.status_code == 400 and "HF base model" in response.text


def test_harness_routes_run_against_the_mock_engine(client):
    test, state = client

    async def scenario():
        saved = state.manager.create(_spec())
        deadline = asyncio.get_running_loop().time() + 20
        while saved.status not in ("ready", "error") and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
        assert saved.status == "ready", saved.message
        return saved

    dep = asyncio.run(scenario())
    profiles = test.get("/_kui/api/workbench/ml/harness").json()["profiles"]
    assert {p["name"] for p in profiles} == {"chat_basic", "json_output", "tool_choice"}
    report = test.post("/_kui/api/workbench/ml/harness",
                       json={"deployment": dep.served_name, "profile": "json_output"}).json()
    assert report["total"] == 10 and report["profile"] == "json_output"
    stored = state.artifacts.get(report["artifact"].split("@")[0])
    assert stored is not None and stored.eval_summary["total"] == 10

