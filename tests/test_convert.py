"""Merge/quantize/edit guards and the adapter artifact handoff."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaggleui.engines.context import DeployError
from kaggleui.engines.llamacpp_launch import quantize_argv
from kaggleui.ml.convert import quantize
from kaggleui.ml.edit import StudioEditSpec
from kaggleui.schemas import ArtifactManifest, ModelRef, Settings
from kaggleui.server.state import AppState


def _state(paths) -> AppState:
    state = AppState(paths=paths, settings=Settings(), port=8941, mock=True, fake_hw="t4x2")
    state.settings.api.require_key = False
    return state

def test_quantize_refuses_unknown_quant_and_missing_model(paths):
    state = _state(paths)
    with pytest.raises(DeployError, match="quant must be"):
        quantize(state, "model-x@m1", "Q2_K")
    with pytest.raises(DeployError, match="No artifact"):
        quantize(state, "model-x@m1", "Q4_K_M")

def test_quantize_refuses_a_directory_without_config(paths):
    state = _state(paths)
    directory = paths.cache / "src"
    directory.mkdir(parents=True)
    (directory / "weights.bin").write_text("x")
    saved = state.artifacts.add(ArtifactManifest(artifact_id="model-bare", version="m1", kind="model"),
                                [directory / "weights.bin"])
    with pytest.raises(DeployError, match="transformers model directory"):
        quantize(state, f"{saved.artifact_id}@{saved.version}", "Q4_K_M")

def test_merge_refuses_a_non_adapter_artifact(paths):
    from kaggleui.ml.convert import merge

    state = _state(paths)
    directory = paths.cache / "src2"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    saved = state.artifacts.add(ArtifactManifest(artifact_id="model-x", version="m1", kind="model"),
                                [directory / "config.json"])

    async def go():
        await merge(state, ModelRef(source="hf", id="Qwen/Qwen2.5-0.5B", engine="transformers"),
                    f"{saved.artifact_id}@{saved.version}")

    import asyncio

    with pytest.raises(DeployError, match="expected adapter"):
        asyncio.run(go())

def test_edit_spec_maps_to_the_abliterate_contract():
    spec = StudioEditSpec(source_repo="Qwen/Qwen2.5-0.5B", method="heretic", trials=40, quantize=True)
    assert spec.model_dump()["method"] == "heretic"
    with pytest.raises(ValidationError):
        StudioEditSpec(source_repo="x", trials=3)

def test_quantize_argv_is_positional():
    assert quantize_argv(bin_path="q", source="in.gguf", dest="out.gguf", quant="q4_k_m") == [
        "q", "in.gguf", "out.gguf", "Q4_K_M"]

def test_training_completion_registers_an_adapter_artifact(paths):
    from kaggleui.ml.recipes import TrainingRun
    from kaggleui.ml.train import _on_event

    state = _state(paths)
    run = TrainingRun.model_validate({"base": {"source": "hf", "id": "Qwen/Qwen2.5-0.5B", "engine": "transformers"},
                                      "dataset_artifact": "dataset-a@d1"})
    job = state.jobs.create("train", {"run": run.model_dump(mode="json")})
    out = paths.cache / "train" / "x"
    out.mkdir(parents=True)
    (out / "adapter_model.safetensors").write_text("{}")
    (out / "adapter_config.json").write_text("{}")
    _on_event(state, job.id, run, {"artifact": {"dir": str(out),
                                                "files": ["adapter_model.safetensors", "adapter_config.json"]}})
    manifest = state.artifacts.get(f"adapter-{run.output_name}")
    assert manifest is not None and manifest.kind == "adapter"
    assert manifest.recipe["dataset_artifact"] == "dataset-a@d1"
    assert state.jobs.get(job.id).artifacts == [f"{manifest.artifact_id}@{manifest.version}"]


def test_convert_refuses_an_adapter_without_a_base(paths):
    import asyncio

    from kaggleui.ml.convert import convert

    state = _state(paths)
    directory = paths.cache / "adapter"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "adapter_model.safetensors").write_text("{}")
    saved = state.artifacts.add(ArtifactManifest(artifact_id="adapter-a", version="a1", kind="adapter"),
                                [directory / "config.json", directory / "adapter_model.safetensors"])
    with pytest.raises(DeployError, match="no base model"):
        asyncio.run(convert(state, f"{saved.artifact_id}@{saved.version}", "Q4_K_M"))

def test_convert_refuses_an_unknown_quant(paths):
    import asyncio

    from kaggleui.ml.convert import convert

    state = _state(paths)
    directory = paths.cache / "model"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    saved = state.artifacts.add(ArtifactManifest(artifact_id="model-q", version="m1", kind="model"),
                                [directory / "config.json"])
    with pytest.raises(DeployError, match="unknown quant"):
        asyncio.run(convert(state, f"{saved.artifact_id}@{saved.version}", "Q2_K"))


def test_gguf_files_pick_the_requested_quant(paths):
    from kaggleui.ml.convert import _gguf_files

    directory = paths.cache / "ggufs"
    directory.mkdir(parents=True)
    for name in ("model-f16.gguf", "model-Q4_K_M.gguf", "mmproj-f16.gguf"):
        (directory / name).write_bytes(b"GGUF")
    files = ["model-f16.gguf", "model-Q4_K_M.gguf", "mmproj-f16.gguf"]
    assert _gguf_files(directory, files, "Q4_K_M")[0].name == "model-Q4_K_M.gguf"
    assert _gguf_files(directory, files, "F16")[0].name == "model-f16.gguf"

