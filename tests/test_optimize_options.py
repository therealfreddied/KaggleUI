from __future__ import annotations

from types import SimpleNamespace

from kaggleui.optimize.pipeline import Optimizer
from kaggleui.schemas import CacheProfile, DeploymentSpec, ModelRef, OptimizationProfile


def _state(tmp_path):
    return SimpleNamespace(paths=SimpleNamespace(cache=tmp_path / "cache"),
                           settings=SimpleNamespace(optimize=OptimizationProfile(enabled=False),
                                                    cache=CacheProfile(exact=True, semantic=False)),
                           bus=SimpleNamespace(publish=lambda *a, **k: None),
                           manager=SimpleNamespace(deployments={}))

def _dep(options: dict):
    spec = DeploymentSpec(modality="text", model=ModelRef(id="org/m", engine="mock"), options=options)
    return SimpleNamespace(spec=spec, info={}, id="dep1", served_name="m", logs=[],
                           backend=None, status="ready")

def test_prompt_prepends_to_existing_system(tmp_path):
    opt = Optimizer(_state(tmp_path))
    body = {"messages": [{"role": "system", "content": "app"}, {"role": "user", "content": "hi"}]}
    out = opt.prepare(_dep({"system_prompt": "model rule"}), body)
    assert out["messages"][0]["content"] == "model rule\n\napp"
    assert body["messages"][0]["content"] == "app"

def test_prompt_replace_mode(tmp_path):
    opt = Optimizer(_state(tmp_path))
    body = {"messages": [{"role": "system", "content": "app"}, {"role": "user", "content": "hi"}]}
    out = opt.prepare(_dep({"system_prompt": "only me", "system_prompt_mode": "replace"}), body)
    assert out["messages"][0]["content"] == "only me"

def test_global_prompt_used_as_default(tmp_path):
    state = _state(tmp_path)
    state.settings.optimize = OptimizationProfile(enabled=False, system_prompt="global rule")
    out = Optimizer(state).prepare(_dep({}), {"messages": [{"role": "user", "content": "hi"}]})
    assert out["messages"][0] == {"role": "system", "content": "global rule"}

def test_slim_harness_strips_claude_code_sections(tmp_path):
    system = ("You are Claude Code, Anthropic's official CLI for Claude.\n"
              "# Tone and style\nbe terse\n# Doing tasks\nuse tools\n# Code References\nfile:line\n")
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": "hi"}]}
    opt = Optimizer(_state(tmp_path))
    out = opt.prepare(_dep({"slim_harness": True}), body)
    code = "You are Claude Code, Anthropic's official CLI for Claude."
    assert out["messages"][0]["content"].strip() == code
    assert any(note.startswith("harness:") for note in opt.last_notes)

def test_unknown_system_prompt_passes_through(tmp_path):
    body = {"messages": [{"role": "system", "content": "# Doing tasks\nkeep me"},
                         {"role": "user", "content": "hi"}]}
    out = Optimizer(_state(tmp_path)).prepare(_dep({"slim_harness": True}), body)
    assert out["messages"][0]["content"] == "# Doing tasks\nkeep me"

def test_tool_output_compressed(tmp_path):
    noisy = "\x1b[31mred\x1b[0m\n" + "same\n" * 3 + "".join(f"line {i}\n" for i in range(210))
    body = {"messages": [{"role": "tool", "content": noisy}]}
    opt = Optimizer(_state(tmp_path))
    out = opt.prepare(_dep({"compress": ["tools"]}), body)
    text = out["messages"][0]["content"]
    assert "\x1b" not in text
    assert "repeated 3×" in text
    assert "lines elided" in text
    assert len(text.splitlines()) <= 200

def test_cache_mode_option_overrides_global(tmp_path):
    opt = Optimizer(_state(tmp_path))
    assert opt.cache_mode(_dep({})) == "exact"
    assert opt.cache_mode(_dep({"cache": "off"})) == "off"
    assert opt.cache_mode(_dep({"cache": "semantic"})) == "semantic"

async def test_cache_off_skips_store(tmp_path):
    state = _state(tmp_path)
    calls: list[str] = []

    class Backend:
        async def chat(self, body):
            calls.append("chat")
            return {"choices": [{"message": {"content": "ok"}}]}

    dep = _dep({"cache": "off"})
    dep.backend = Backend()
    opt = Optimizer(state)
    await opt.chat(dep, {"messages": [{"role": "user", "content": "hi"}]})
    assert calls == ["chat"]
    assert opt.cache.stats.stores == 0
