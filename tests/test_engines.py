"""Engine dispatch: the registry routes each spec to its runner, and GPU/TPU refusals fire before any install."""

from __future__ import annotations

import asyncio
import json

import pytest

from kaggleui.engines import registry
from kaggleui.engines.context import DeployError, EngineContext
from kaggleui.schemas import DeploymentSpec


def _ctx(paths, hw) -> EngineContext:
    return EngineContext(paths=paths, dep_id="t", served_name="m", gpus=[0], hw=hw,
                         log=lambda _line: None, progress=lambda *_args: None, internal_key="k", main_port=1)


def _spec(engine: str) -> DeploymentSpec:
    return DeploymentSpec(modality="text", model={"source": "hf", "id": "Qwen/Qwen3-0.6B", "engine": engine})


def test_sglang_refuses_tpu_before_installing(paths, hw):
    with pytest.raises(DeployError, match="TPU"):
        asyncio.run(registry.deploy(_spec("sglang"), _ctx(paths, hw("tpu")), mock=False))


def test_sglang_refuses_pre_t4_gpus(paths, hw):
    with pytest.raises(DeployError, match="T4"):
        asyncio.run(registry.deploy(_spec("sglang"), _ctx(paths, hw("p100")), mock=False))


def test_vllm_refuses_pre_t4_gpus(paths, hw):
    with pytest.raises(DeployError, match="T4"):
        asyncio.run(registry.deploy(_spec("vllm"), _ctx(paths, hw("p100")), mock=False))


# --- llama.cpp serving Ollama refs: one pull feeds both engines --------------------------------

def _ollama_store(paths, ref: str, payload: bytes = b"GGUF\x03fake"):
    name, _, tag = ref.rpartition(":")
    if "/" not in name:
        name = f"library/{name}"
    blob = paths.ollama_models / "blobs" / "sha256-deadbeef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    manifest = paths.ollama_models / "manifests" / "registry.ollama.ai" / name / tag
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:aa"},
                                               {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:deadbeef"}]}))
    return blob


def test_ollama_gguf_blob_resolves_library_ref(paths):
    from kaggleui.engines import ollama
    blob = _ollama_store(paths, "qwen3:8b")
    assert ollama.gguf_blob(paths, "qwen3:8b") == blob


def test_ollama_gguf_blob_resolves_hf_refs_and_default_tag(paths, hw):
    from kaggleui.engines import ollama
    blob = _ollama_store(paths, "hf.co/owner/repo-GGUF:Q4_K_M")
    assert ollama.gguf_blob(paths, "hf.co/owner/repo-GGUF:Q4_K_M") == blob
    assert ollama.gguf_blob(paths, "missing:latest") is None


def test_ollama_gguf_blob_rejects_a_non_gguf_model_layer(paths):
    from kaggleui.engines import ollama
    _ollama_store(paths, "qwen3:8b", payload=b"not-a-gguf")
    assert ollama.gguf_blob(paths, "qwen3:8b") is None


def test_llamacpp_pulls_ollama_refs_through_the_shared_server(paths, hw, monkeypatch):
    from kaggleui.engines import llamacpp, ollama
    blob = _ollama_store(paths, "qwen3:8b")
    pulled = []

    async def fake_ensure(_self, _ctx):
        return "http://127.0.0.1:11434"

    async def fake_pull(_base, ref, _ctx):
        pulled.append(ref)

    monkeypatch.setattr(ollama.OllamaServer, "ensure", fake_ensure)
    monkeypatch.setattr(ollama, "_pull", fake_pull)

    spec = DeploymentSpec(modality="text", model={"source": "ollama", "id": "qwen3", "variant": "8b", "engine": "llamacpp"})
    ctx = _ctx(paths, hw("t4x2"))
    assert asyncio.run(llamacpp.model_path(ctx, spec)) == blob
    assert pulled == ["qwen3:8b"]
