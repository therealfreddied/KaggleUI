from __future__ import annotations

import pytest

from kaggleui.engines import llamaswap
from kaggleui.engines.context import DeployError, EngineContext
from kaggleui.system.probe import Probe


@pytest.fixture
def ctx(paths):
    return EngineContext(paths=paths, dep_id="dep1", served_name="gguf-model", gpus=[], hw=Probe(paths, fake="t4x2").snapshot(),
                         log=lambda _l: None, progress=lambda *a: None, internal_key="k", main_port=8890)


@pytest.fixture
def inst(paths):
    return llamaswap.instance_for(paths)


def _stub_process(monkeypatch):
    async def start(_self):
        return None

    monkeypatch.setattr(llamaswap.ManagedProcess, "start", start)


def _stub_binary(monkeypatch, ctx, tmp_path):
    binary = tmp_path / "llama-swap"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(llamaswap.runtime, "swap_bin", lambda _paths: binary)


async def test_ensure_running_returns_once_models_answer(monkeypatch, tmp_path, paths, ctx, inst):
    """A proxy that answers /v1/models on the third poll must not burn the whole health timeout."""
    _stub_binary(monkeypatch, ctx, tmp_path)
    _stub_process(monkeypatch)
    calls = {"n": 0}

    async def online(_port):
        calls["n"] += 1
        return None if calls["n"] < 3 else ["gguf-model"]

    monkeypatch.setattr(llamaswap, "_models_online", online)
    monkeypatch.setattr(llamaswap, "HEALTH_TIMEOUT", 30.0)

    await llamaswap._ensure_running(inst, ctx)
    assert calls["n"] == 3
    assert inst.models == {}


async def test_ensure_running_raises_when_proxy_never_answers(monkeypatch, tmp_path, paths, ctx, inst):
    _stub_binary(monkeypatch, ctx, tmp_path)
    _stub_process(monkeypatch)
    calls = {"n": 0}

    async def online(_port):
        calls["n"] += 1
        return None

    monkeypatch.setattr(llamaswap, "_models_online", online)
    monkeypatch.setattr(llamaswap, "HEALTH_TIMEOUT", 0.25)

    with pytest.raises(DeployError) as err:
        await llamaswap._ensure_running(inst, ctx)
    assert "did not answer /v1/models" in str(err.value)
    assert calls["n"] >= 1


def test_write_config_is_atomic(paths, inst):
    inst.models = {"a": {"cmd": "x"}}
    llamaswap._write_config(inst)
    assert inst.config_path.exists()
    assert not inst.config_path.with_suffix(".tmp").exists()
    assert "a:" in inst.config_path.read_text() or "a" in inst.config_path.read_text()


def teardown_function():
    llamaswap._INSTANCES.clear()
