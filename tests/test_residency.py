"""Residency: cold deployments, lazy start, idle unload and restore after a restart."""

import asyncio

from fastapi.testclient import TestClient

from kaggleui.schemas import DeploymentSpec, Settings
from kaggleui.server.app import create_app
from kaggleui.server.state import AppState


def _app(paths) -> AppState:
    settings = Settings()
    settings.exposure.provider = "none"
    return AppState(paths=paths, settings=settings, port=8997, mock=True, fake_hw="t4x2")


def _spec(**extra) -> DeploymentSpec:
    body = {"modality": "text", "model": {"source": "hf", "id": "mock/text", "engine": "mock"},
            "options": {"vram_estimate": 1_000_000_000}, **extra}
    return DeploymentSpec.model_validate(body)


async def _wait_ready(dep, timeout=15.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if dep.status in ("ready", "error"):
            return dep
        await asyncio.sleep(0.05)
    raise AssertionError(f"{dep.served_name} stayed {dep.status}")


async def test_on_demand_starts_lazily_and_records_itself(paths):
    state = _app(paths)
    dep = state.manager.create(_spec(residency="on_demand", idle_timeout_s=30))
    assert dep.status == "cold" and dep.backend is None
    assert state.workbench.deployment_record(dep.id) is not None

    await asyncio.wait_for(state.manager.ensure_ready(dep), timeout=20)
    assert dep.status == "ready" and dep.backend is not None
    await state.manager.shutdown()


async def test_idle_sweep_returns_a_model_to_cold(paths):
    state = _app(paths)
    dep = state.manager.create(_spec(residency="on_demand", idle_timeout_s=30))
    await asyncio.wait_for(state.manager.ensure_ready(dep), timeout=20)

    dep.last_used = dep.ready_at = 0.0  # idle for longer than the timeout
    assert await state.residency.sweep() == 1
    assert dep.status == "cold" and dep.backend is None
    assert state.workbench.deployment_record(dep.id) is not None
    await state.manager.shutdown()


async def test_pinned_models_are_never_swept(paths):
    state = _app(paths)
    dep = state.manager.create(_spec(residency="pinned"))
    await _wait_ready(dep)
    dep.last_used = dep.ready_at = 0.0
    assert await state.residency.sweep() == 0
    assert dep.status == "ready"
    await state.manager.shutdown()


async def test_cold_deployment_is_rebuilt_after_restart(paths):
    first = _app(paths)
    dep = first.manager.create(_spec(residency="on_demand"))
    await first.manager.shutdown()

    second = _app(paths)
    restored = second.manager.deployments
    assert len(restored) == 1
    only = next(iter(restored.values()))
    assert only.status == "cold" and only.spec.residency == "on_demand"
    assert only.served_name == dep.served_name
    assert second.manager.resolve(only.served_name) is only
    await second.manager.shutdown()


def test_gateway_request_wakes_a_cold_model(paths):
    state = _app(paths)
    state.settings.api.require_key = False
    dep = state.manager.create(_spec(residency="on_demand"))
    assert dep.status == "cold"
    with TestClient(create_app(state)) as client:
        reply = client.post("/v1/chat/completions", json={
            "model": dep.served_name, "messages": [{"role": "user", "content": "hi"}]})
        assert reply.status_code == 200, reply.text
        assert reply.json()["choices"][0]["message"]["content"]
        assert dep.status == "ready"
