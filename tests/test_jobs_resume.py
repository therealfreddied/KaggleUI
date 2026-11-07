"""Resume picks the newest adapter checkpoint and rewrites the worker spec."""

from __future__ import annotations

import json

from kaggleui.ml.hub import latest_checkpoint
from kaggleui.schemas import Settings
from kaggleui.server.state import AppState


def _state(paths) -> AppState:
    return AppState(paths=paths, settings=Settings(), port=8931, mock=True, fake_hw="t4x2")


def _checkpoint(root, step, *, adapter=True):
    directory = root / f"checkpoint-{step}"
    directory.mkdir(parents=True, exist_ok=True)
    if adapter:
        (directory / "adapter_model.safetensors").write_text("{}")
    return directory


def test_resume_rewrites_the_spec_with_the_newest_checkpoint(paths, monkeypatch):
    from kaggleui.ml import train as train_module

    calls = []
    monkeypatch.setattr(train_module, "start_in_background",
                        lambda store, job, cmd, **kwargs: calls.append((job.id, cmd)))
    scratch = paths.cache / "train" / "run1"
    _checkpoint(scratch / "checkpoints", 100)
    _checkpoint(scratch / "checkpoints", 200)
    _checkpoint(scratch / "checkpoints", 300, adapter=False)  # no adapter file: not resumable
    spec_path = scratch / "spec.json"
    spec_path.write_text(json.dumps({"scratch": str(scratch), "dataset_path": "d.jsonl"}))
    state = _state(paths)
    parent = state.jobs.create("train", {"cmd": ["python", "-m", "x"], "spec_path": str(spec_path)},
                               resume_supported=True, logs_path=str(paths.logs / "t.log"))

    fresh = train_module.resume(state, parent.id)

    assert calls and calls[0][0] == fresh["id"]
    spec = json.loads(spec_path.read_text())
    assert spec["resume_from"].endswith("checkpoint-200")
    assert latest_checkpoint(scratch / "checkpoints").name == "checkpoint-200"


def test_resume_refuses_without_a_checkpoint(paths):
    from kaggleui.engines.context import DeployError
    from kaggleui.ml.train import resume

    scratch = paths.cache / "train" / "run2"
    scratch.mkdir(parents=True)
    spec_path = scratch / "spec.json"
    spec_path.write_text(json.dumps({"scratch": str(scratch)}))
    state = _state(paths)
    job = state.jobs.create("train", {"spec_path": str(spec_path)}, resume_supported=True)
    try:
        resume(state, job.id)
    except DeployError as exc:
        assert "No checkpoint" in str(exc)
    else:
        raise AssertionError("resume should refuse a run without checkpoints")
