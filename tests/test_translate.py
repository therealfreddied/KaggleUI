import json

from kaggleui.gateway.translate import (
    anthropic_to_openai,
    ollama_chat_to_openai,
    openai_stream_to_anthropic,
    openai_stream_to_ollama,
    openai_to_anthropic,
)


def test_anthropic_request_with_system_tools_images_and_results():
    body = {
        "model": "x", "max_tokens": 100, "system": [{"type": "text", "text": "Be brief."}],
        "stop_sequences": ["END"], "temperature": 0.2,
        "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {}}}],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Weather?"},
                                         {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                                      "data": "AAA"}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Checking."},
                                              {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                                               "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"}]},
        ],
    }
    out = anthropic_to_openai(body)
    msgs = out["messages"]
    assert msgs[0] == {"role": "system", "content": "Be brief."}
    assert msgs[1]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,AAA"
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    assert msgs[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "Sunny"}
    assert out["stop"] == ["END"] and out["max_tokens"] == 100
    assert out["tools"][0]["function"]["name"] == "get_weather"
    assert out["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_openai_response_to_anthropic_with_tool_call():
    resp = {"choices": [{"message": {"content": "Let me check", "tool_calls": [
        {"id": "call_1", "function": {"name": "f", "arguments": '{"a": 1}'}}]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7}}
    out = openai_to_anthropic(resp, "m")
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0] == {"type": "text", "text": "Let me check"}
    assert out["content"][1]["input"] == {"a": 1}
    assert out["usage"] == {"input_tokens": 5, "output_tokens": 7}


async def _agen(items):
    for i in items:
        yield i


def _events(raw: list[str]) -> list[dict]:
    out = []
    for block in raw:
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


async def test_stream_text_then_tool_call_to_anthropic():
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "Hi"}}]},
        {"choices": [{"delta": {"content": " there"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_9", "function": {"name": "f", "arguments": "{\"x\""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 2}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 9}},
    ]
    events = _events([e async for e in openai_stream_to_anthropic(_agen(chunks), "m")])
    types = [e["type"] for e in events]
    assert types[0] == "message_start" and types[-1] == "message_stop"
    text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta")
    assert text == "Hi there"
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert starts[1]["content_block"]["type"] == "tool_use" and starts[1]["index"] == 1
    args = "".join(e["delta"]["partial_json"] for e in events if e.get("delta", {}).get("type") == "input_json_delta")
    assert json.loads(args) == {"x": 2}
    delta = next(e for e in events if e["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use" and delta["usage"]["output_tokens"] == 9
    # every started block is stopped
    assert types.count("content_block_start") == types.count("content_block_stop")


def test_ollama_chat_request_with_images_and_options():
    out = ollama_chat_to_openai({"messages": [{"role": "user", "content": "what is this", "images": ["QUJD"]}],
                                 "options": {"num_predict": 50, "temperature": 0.1}, "format": "json"})
    assert out["messages"][0]["content"][1]["image_url"]["url"].endswith("QUJD")
    assert out["max_tokens"] == 50 and out["temperature"] == 0.1
    assert out["response_format"] == {"type": "json_object"}
    assert out["stream"] is True  # Ollama streams by default


async def test_openai_stream_to_ollama_ndjson():
    chunks = [{"choices": [{"delta": {"content": "a"}}]}, {"choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}]}]
    lines = [json.loads(line) async for line in openai_stream_to_ollama(_agen(chunks), "m")]
    assert [x["message"]["content"] for x in lines[:-1]] == ["a", "b"]
    assert lines[-1]["done"] is True and lines[-1]["done_reason"] == "stop"


def test_thinking_controls_map_to_reasoning_effort():
    base = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}]}
    assert anthropic_to_openai({**base, "thinking": {"type": "disabled"}})["reasoning_effort"] == "none"
    assert anthropic_to_openai({**base, "thinking": {"type": "enabled", "budget_tokens": 1024}})["reasoning_effort"] == "low"
    assert anthropic_to_openai({**base, "thinking": {"type": "enabled", "budget_tokens": 16000}})["reasoning_effort"] == "high"
    assert "reasoning_effort" not in anthropic_to_openai(base)
    assert ollama_chat_to_openai({"messages": [], "think": False})["reasoning_effort"] == "none"
    assert ollama_chat_to_openai({"messages": [], "think": "medium"})["reasoning_effort"] == "medium"
    assert "reasoning_effort" not in ollama_chat_to_openai({"messages": [], "think": True})


def test_template_engines_get_enable_thinking_switch():
    from kaggleui.engines.base import OpenAIUpstream

    vllm = OpenAIUpstream("http://x", "m", template_thinking=True)
    assert vllm._body({"reasoning_effort": "none"})["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in vllm._body({"reasoning_effort": "high"})
    ollama = OpenAIUpstream("http://x", "m")
    assert "chat_template_kwargs" not in ollama._body({"reasoning_effort": "none"})


def test_anthropic_bad_budget_tokens_is_ignored_not_a_500():
    body = {"messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": "lots"}}
    out = anthropic_to_openai(body)
    assert "reasoning_effort" not in out
