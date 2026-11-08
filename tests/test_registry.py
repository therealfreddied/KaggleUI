"""Artifact registry: staging, hashing, overwrite protection, tamper detection, confirmed delete."""

import json

import pytest

from kaggleui.schemas import ArtifactManifest
from kaggleui.workbench.registry import ArtifactRegistry, RegistryError


def _files(tmp_path):
    one = tmp_path / "model-Q4_K_M.gguf"
    one.write_bytes(b"gguf" * 100)
    two = tmp_path / "tokenizer.json"
    two.write_bytes(b'{"vocab": 1}')
    return one, two


def _manifest(**extra):
    body = {"artifact_id": "demo", "version": "v1", "kind": "model", **extra}
    return ArtifactManifest(**body)


def test_add_hashes_files_and_writes_a_manifest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    one, two = _files(src)
    reg = ArtifactRegistry(tmp_path / "artifacts")

    saved = reg.add(_manifest(), [one, two])
    assert [f.path for f in saved.files] == ["model-Q4_K_M.gguf", "tokenizer.json"]
    assert saved.files[0].size == 400 and len(saved.files[0].sha256) == 64

    version_dir = reg.dir_for("demo", "v1")
    assert (version_dir / "manifest.json").is_file()
    assert json.loads(reg.index.read_text())["artifacts"][0]["version"] == "v1"
    assert reg.get("demo") == saved


def test_overwrite_needs_explicit_flag(tmp_path):
    one, _ = _files(tmp_path)
    reg = ArtifactRegistry(tmp_path / "artifacts")
    reg.add(_manifest(), [one])
    with pytest.raises(RegistryError, match="already exists"):
        reg.add(_manifest(), [one])
    assert reg.add(_manifest(), [one], overwrite=True).version == "v1"


def test_verify_detects_a_tampered_file(tmp_path):
    one, two = _files(tmp_path)
    reg = ArtifactRegistry(tmp_path / "artifacts")
    reg.add(_manifest(), [one, two])
    assert reg.verify("demo")["ok"]

    target = reg.dir_for("demo", "v1") / "tokenizer.json"
    target.write_bytes(b'{"vocab": 2}')
    report = reg.verify("demo")
    assert not report["ok"] and "hash mismatch: tokenizer.json" in report["problems"]

    target.unlink()
    assert "missing: tokenizer.json" in reg.verify("demo")["problems"]
    assert reg.verify("nope")["ok"] is False


def test_delete_requires_confirm_and_updates_the_index(tmp_path):
    one, _ = _files(tmp_path)
    reg = ArtifactRegistry(tmp_path / "artifacts")
    reg.add(_manifest(), [one])
    with pytest.raises(RegistryError, match="confirm"):
        reg.remove("demo", "v1", confirm=False)
    reg.remove("demo", "v1", confirm=True)
    assert reg.get("demo") is None
    assert json.loads(reg.index.read_text())["artifacts"] == []


def test_nested_layouts_keep_relative_paths(tmp_path):
    root = tmp_path / "proj"
    (root / "adapter").mkdir(parents=True)
    (root / "README.md").write_text("hi")
    (root / "adapter" / "adapter_model.safetensors").write_bytes(b"w" * 10)
    reg = ArtifactRegistry(tmp_path / "artifacts")
    saved = reg.add(_manifest(version="v2", kind="adapter"),
                    [root / "README.md", root / "adapter" / "adapter_model.safetensors"])
    assert sorted(f.path for f in saved.files) == ["README.md", "adapter/adapter_model.safetensors"]
    assert (reg.dir_for("demo", "v2") / "adapter" / "adapter_model.safetensors").is_file()


def test_ids_cannot_escape_the_root(tmp_path):
    one, _ = _files(tmp_path)
    reg = ArtifactRegistry(tmp_path / "artifacts")
    saved = reg.add(_manifest(artifact_id="../../etc"), [one])
    assert saved.artifact_id == "etc"
    assert reg.dir_for("a/b", "v1").parent == reg.root / "a-b"


def test_corrupt_index_is_treated_as_empty(tmp_path):
    reg = ArtifactRegistry(tmp_path / "artifacts")
    reg.root.mkdir(parents=True)
    reg.index.write_text("{not json")
    assert reg.list() == [] and reg.get("demo") is None

def test_concurrent_adds_keep_every_index_entry(tmp_path):
    """Several jobs saving artifacts at the same time must all stay listable."""
    import threading
    import time

    src = tmp_path / "src"
    src.mkdir()
    one, _ = _files(src)
    reg = ArtifactRegistry(tmp_path / "artifacts")
    real_entries = reg._entries

    def slow_entries():
        entries = real_entries()
        time.sleep(0.1)                      # widen the window an unlocked index cycle would leave open
        return entries

    reg._entries = slow_entries

    errors: list[BaseException] = []

    def add(artifact_id):
        try:
            reg.add(_manifest(artifact_id=artifact_id), [one])
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(f"art-{i}",), name=f"add-{i}") for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    reg._entries = real_entries
    assert not errors
    assert {m.artifact_id for m in reg.list()} == {f"art-{i}" for i in range(4)}
