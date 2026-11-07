"""Artifact-backed llama.cpp deployments resolve the GGUF through the registry."""

from __future__ import annotations

import pytest

from kaggleui.engines.context import DeployError
from kaggleui.engines.llamacpp import local_gguf
from kaggleui.schemas import ArtifactManifest, DeploymentSpec, ModelRef
from kaggleui.workbench.registry import ArtifactRegistry


def _seed(paths, files: list[str]) -> str:
    directory = paths.cache / "seed"
    directory.mkdir(parents=True)
    written = []
    for name in files:
        path = directory / name
        path.write_bytes(b"GGUF" + b"\x00" * 64)
        written.append(path)
    registry = ArtifactRegistry(paths.models / "artifacts")
    registry.add(ArtifactManifest(artifact_id="demo", version="d1", kind="export"), written)
    return "demo"

def test_artifact_gguf_picks_the_first_shard(paths):
    _seed(paths, ["model-Q4_K_M-00001-of-00002.gguf", "model-Q4_K_M-00002-of-00002.gguf",
                  "mmproj-f16.gguf"])
    resolved = local_gguf(DeploymentSpec(modality="text", model=ModelRef(source="artifact", id="demo", engine="llamacpp")), paths)
    assert resolved.name == "model-Q4_K_M-00001-of-00002.gguf"

def test_artifact_gguf_refuses_an_artifact_without_gguf(paths):
    _seed(paths, ["adapter_model.safetensors"])
    with pytest.raises(DeployError, match="no GGUF file"):
        local_gguf(DeploymentSpec(modality="text", model=ModelRef(source="artifact", id="demo", engine="llamacpp")), paths)

def test_artifact_gguf_refuses_an_unknown_artifact(paths):
    with pytest.raises(DeployError, match="is registered"):
        local_gguf(DeploymentSpec(modality="text", model=ModelRef(source="artifact", id="nope", engine="llamacpp")), paths)

