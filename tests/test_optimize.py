"""Optimize pipeline: context trimming, headroom caps, terse prompts, and the response cache."""

import time

import pytest
from fastapi.testclient import TestClient

from kaggleui.optimize.cache import ResponseCache, cache_key, cacheable, cosine, semantic_eligible
from kaggleui.optimize.context import SYSTEM_WARNING, apply_context, estimate_tokens
from kaggleui.optimize.headroom import apply_headroom
from kaggleui.optimize.terse import TERSE_INSTRUCTION, apply_terse
from kaggleui.schemas import Settings
from kaggleui.server.app import create_app
from kaggleui.server.state import AppState


def _msgs(*pairs):
    return [{"role": role, "content": text} for role, text in pairs]


def test_estimate_tokens_counts_tools_too():
    body = {"messages": _msgs(("user", "abcd"))}
    plain = estimate_tokens(body)
    body["tools"] = [{"type": "function", "function": {"name": "f", "description": "x" * 400}}]
    assert estimate_tokens(body) > plain + 50


def test_context_keeps_system_and_final_turn():
    messages = _msgs(("system", "be terse"), ("user", "old " * 400), ("assistant", "ok"),
                     ("user", "new question"))
    out, note = apply_context({"messages": messages}, max_prompt_tokens=120, strategy="balanced")
    roles = [m["role"] for m in out["messages"]]
    assert roles[0] == "system"
    assert out["messages"][-1]["content"] == "new question"
    assert len(out["messages"]) < len(messages)
    assert note and "dropped" in note


def test_context_leaves_small_prompts_untouched():
    body = {"messages": _msgs(("user", "hi"))}
    out, note = apply_context(body, max_prompt_tokens=1000)
    assert out is body and note is None


def test_context_never_truncates_a_system_prompt_that_overflows():
    body = {"messages": _msgs(("system", "rules " * 500), ("user", "hi"))}
    out, note = apply_context(body, max_prompt_tokens=64)
    assert out is body and note == SYSTEM_WARNING


def test_head_strategy_keeps_leading_turns():
    messages = _msgs(("system", "s"), ("user", "first " * 100), ("assistant", "a"), ("user", "last"))
    out, _ = apply_context({"messages": messages}, max_prompt_tokens=80, strategy="head")
    assert out["messages"][1]["content"].startswith("first")
    assert out["messages"][-1]["content"] == "last"


def test_headroom_caps_the_requested_completion():
    body = {"messages": _msgs(("user", "x" * 4000)), "max_tokens": 4000}
    out, note = apply_headroom(body, ctx=2048, margin=256)
    assert out["max_tokens"] <= 2048 - estimate_tokens(body) - 256
    assert "capped" in note or "->" in note


def test_headroom_is_a_noop_without_a_ctx():
    body = {"max_tokens": 10}
    assert apply_headroom(body, ctx=None) == (body, None)


def test_terse_appends_to_the_existing_system_prompt():
    body = {"messages": _msgs(("system", "Use JSON."), ("user", "hi"))}
    out, note = apply_terse(body)
    assert out["messages"][0]["content"].startswith("Use JSON.")
    assert TERSE_INSTRUCTION in out["messages"][0]["content"]
    assert "added" in note
    again, note2 = apply_terse(out)
    assert again is out and note2 is None


class _Dep:
    served_name = "qwen3:0.6b"

    class spec:
        class model:
            id = "qwen3"


def test_cache_key_ignores_stream_but_not_model_or_prompt():
    dep = _Dep()
    base = {"messages": _msgs(("user", "hi")), "temperature": 0}
    assert cache_key(dep, base) == cache_key(dep, {**base, "stream": True, "user": "u1"})
    assert cache_key(dep, base) != cache_key(dep, {**base, "messages": _msgs(("user", "bye"))})
    other = type("Other", (), {"served_name": "other", "spec": dep.spec})
    assert cache_key(dep, base) != cache_key(other, base)


def test_cache_ineligibility_rules():
    dep = _Dep()
    assert cacheable({"temperature": 0})
    assert not cacheable({"temperature": 0.7})
    assert not cacheable({"temperature": 0, "seed": 1})
    assert not cacheable({"temperature": 0, "tools": []})
    assert not cacheable({})
    assert cache_key(dep, {"temperature": 0.7}) is None
    assert semantic_eligible({"messages": _msgs(("user", "hi"))})
    assert not semantic_eligible({"messages": _msgs(("user", "hi")), "stream": True})
    assert not semantic_eligible({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})


def test_response_cache_serves_exact_and_semantic_hits(tmp_path, monkeypatch):
    dep = _Dep()
    cache = ResponseCache(tmp_path)
    body = {"messages": _msgs(("user", "hi")), "temperature": 0}
    assert cache.lookup(dep, body, ttl_s=60) is None
    cache.store(dep, body, {"id": "a"}, vector=[1.0, 0.0])
    assert cache.lookup(dep, body, ttl_s=60)["id"] == "a"
    assert cache.lookup_semantic(dep, [0.99, 0.01], threshold=0.9, ttl_s=60)["id"] == "a"
    assert cache.lookup_semantic(dep, [0.0, 1.0], threshold=0.9, ttl_s=60) is None
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cache.stats.exact_hits == 1 and cache.stats.semantic_hits == 1

    # Expired entries are dropped, and a rebuilt cache still finds the persisted file.
    later = time.time() + 120
    monkeypatch.setattr("kaggleui.optimize.cache.time.time", lambda: later)
    assert cache.lookup(dep, body, ttl_s=60) is None
    cache.store(dep, body, {"id": "b"})
    rebuilt = ResponseCache(tmp_path)
    assert rebuilt.lookup(dep, body, ttl_s=60)["id"] == "b"

    # The file name is the key, so a different deployment never shares an entry.
    other = type("Other", (), {"served_name": "other", "spec": dep.spec})
    assert rebuilt.lookup(other, body, ttl_s=60) is None


def test_response_cache_prunes_to_its_budget(tmp_path):
    dep = _Dep()
    cache = ResponseCache(tmp_path, max_entries=2)
    for i in range(4):
        cache.store(dep, {"messages": _msgs(("user", f"q{i}")), "temperature": 0}, {"id": str(i)})
    assert len(list((tmp_path / "exact").glob("*.json"))) == 2
    assert cache.stats.evictions == 2


@pytest.fixture
def state(paths):
    settings = Settings()
    settings.exposure.provider = "none"
    return AppState(paths=paths, settings=settings, port=8998, mock=True, fake_hw="t4x2")


@pytest.fixture
def client(state):
    with TestClient(create_app(state)) as c:
        yield c


def _deploy(client, state):
    admin = {"Authorization": f"Bearer {state.auth.admin_token}"}
    spec = {"modality": "text", "model": {"source": "ollama", "id": "qwen3", "variant": "0.6b",
                                          "engine": "ollama"}}
    dep = client.post("/_kui/api/deployments", json=spec, headers=admin).json()
    for _ in range(100):
        rows = client.get("/_kui/api/deployments", headers=admin).json()
        current = next(d for d in rows if d["id"] == dep["id"])
        if current["status"] == "ready":
            return current["name"]
        import time
        time.sleep(0.05)
    raise AssertionError(current)


def test_gateway_serves_identical_answers_from_the_exact_cache(client, state):
    served = _deploy(client, state)
    state.settings.cache.exact = True
    admin = {"Authorization": f"Bearer {state.auth.admin_token}"}
    state.settings.api.require_key = False
    body = {"model": served, "messages": [{"role": "user", "content": "hello"}], "temperature": 0,
            "max_tokens": 16}
    first = client.post("/v1/chat/completions", json=body)
    second = client.post("/v1/chat/completions", json=body)
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    # A warm request must reach the engine again, so the cache key has to include temperature.
    warm = client.post("/v1/chat/completions", json={**body, "temperature": 0.7})
    assert warm.status_code == 200 and warm.json()["id"] != first.json()["id"]
    stats = client.get("/_kui/api/workbench", headers=admin).json()
    assert isinstance(stats, dict)


def test_anthropic_surface_uses_the_same_pipeline(state):
    app = create_app(state)
    with TestClient(app) as c:
        served = _deploy(c, state)
        state.settings.api.enabled = True
        state.settings.api.require_key = False
        state.settings.api.anthropic = True
        message = {"model": served, "max_tokens": 16,
                   "messages": [{"role": "user", "content": "hello"}]}
        r = c.post("/v1/messages", json=message)
        assert r.status_code == 200, r.text
        assert r.json()["type"] == "message"



def test_bench_measures_the_live_backend_and_saves_an_artifact(client, state, tmp_path):
    import asyncio

    from kaggleui.workbench.bench import bench, format_table, prompt_for, save

    served = _deploy(client, state)
    assert len(prompt_for(64)) == 64 * 4
    report = asyncio.run(bench(state, served, ctxs=(512,), prompt_tokens=(32,), runs=1))
    row = report["results"][0]
    assert row["ctx"] == 512 and row["prompt_tokens"] == 32
    assert row["ttft_s"] <= row["total_s"] and row["output_tokens"] > 0
    table = format_table(report)
    assert served in report["model"] and str(row["ctx"]) in table

    save(state, report, out=tmp_path / "bench.json")
    assert (tmp_path / "bench.json").read_text().endswith("\n")
    stored = [m for m in state.artifacts.list() if m.artifact_id.startswith("bench-")]
    assert stored and stored[0].kind == "export"
    assert stored[0].recipe["bench"]["model"] == served
