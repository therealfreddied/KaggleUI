from __future__ import annotations

import asyncio

from kaggleui.gateway.reasoning import StreamParser, extract, normalize_response, normalize_stream
from kaggleui.gateway.translate import openai_stream_to_anthropic, openai_to_anthropic, openai_to_ollama_chat


def test_extract_multiple_blocks_merged():
    content, reasoning = extract("<thinking>a</thinking>answer<thought>b</thought>")
    assert content == "answer"
    assert reasoning == "a\n\nb"

def test_extract_unclosed_tag_moves_tail_to_reasoning():
    content, reasoning = extract("start <thinking>still going")
    assert content == "start"
    assert reasoning == "still going"

def test_extract_with_attributes_and_case():
    content, reasoning = extract("<Think type='x'>t</THINK>hi")
    assert (content, reasoning) == ("hi", "t")

def test_extract_keeps_tags_inside_code_fence():
    text = "```\n<think>x</think>\n```"
    assert extract(text) == (text, "")

def test_extract_collapses_blank_runs_in_reasoning():
    _, reasoning = extract("<think>a\n\n\n\nb</think>ok")
    assert reasoning == "a\n\nb"

def test_stream_parser_handles_split_tag():
    parser = StreamParser()
    pieces = []
    for delta in ["<thi", "nk>dea", "d</th", "ink>done"]:
        pieces += parser.feed(delta)
    pieces += parser.flush()
    assert "".join(text for kind, text in pieces if kind == "reasoning") == "dead"
    assert "".join(text for kind, text in pieces if kind == "content") == "done"

def test_stream_parser_fence_passes_tags_as_content():
    parser = StreamParser()
    out = "".join(text for kind, text in parser.feed("```<think>no</think>```") if kind == "content")
    assert out == "```<think>no</think>```"

def test_stream_parser_flush_closes_open_tag_as_reasoning():
    parser = StreamParser()
    assert parser.feed("<think>tail") == [("reasoning", "tail")]
    assert parser.flush() == []

def test_normalize_response_keeps_native_reasoning():
    resp = {"choices": [{"message": {"content": "ok <think>raw</think>", "reasoning_content": "native"}}]}
    assert normalize_response(resp)["choices"][0]["message"]["content"] == "ok <think>raw</think>"

def test_normalize_stream_moves_reasoning_to_field():
    async def chunks():
        for delta in ["<think>plan</think>", "answer"]:
            yield {"choices": [{"index": 0, "delta": {"content": delta}}]}

    async def collect():
        return [chunk async for chunk in normalize_stream(chunks())]

    out = asyncio.run(collect())
    assert out[0]["choices"][0]["delta"] == {"content": "", "reasoning_content": "plan"}
    assert out[1]["choices"][0]["delta"] == {"content": "answer"}

def test_anthropic_buffered_renders_thinking_block():
    resp = {"choices": [{"message": {"content": "answer", "reasoning_content": "plan"},
                         "finish_reason": "stop"}]}
    out = openai_to_anthropic(resp, "m")
    assert out["content"][0] == {"type": "thinking", "thinking": "plan", "signature": ""}
    assert out["content"][1] == {"type": "text", "text": "answer"}

def test_ollama_buffered_carries_thinking():
    resp = {"choices": [{"message": {"content": "answer", "reasoning_content": "plan"},
                         "finish_reason": "stop"}]}
    assert openai_to_ollama_chat(resp, "m")["message"]["thinking"] == "plan"

def test_anthropic_stream_emits_thinking_delta():
    async def chunks():
        yield {"choices": [{"index": 0, "delta": {"reasoning_content": "plan"}}]}
        yield {"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]}

    async def collect():
        return [part async for part in openai_stream_to_anthropic(chunks(), "m")]

    events = asyncio.run(collect())
    assert any('"type": "thinking_delta"' in event and "plan" in event for event in events)
    assert any('"type": "text_delta"' in event and "answer" in event for event in events)
