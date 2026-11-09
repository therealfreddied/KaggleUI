"""Protocol translation: Anthropic Messages ⇄ OpenAI Chat Completions, and Ollama /api/chat ⇄ OpenAI.

The internal lingua franca is OpenAI chat-completions JSON. Engines that speak a flavour natively
(Ollama speaks Anthropic and Ollama) are proxied directly instead.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

# =============================================================================================
# Anthropic → OpenAI
# =============================================================================================

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                "function_call": "tool_use", "content_filter": "end_turn"}


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return "\n".join(out)


def _image_part(block: dict) -> dict | None:
    src = block.get("source") or {}
    if src.get("type") == "base64":
        return {"type": "image_url", "image_url": {"url": f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"}}
    if src.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": src.get("url")}}
    return None


def effort_from_anthropic(thinking: dict | None, output_config: dict | None = None) -> str | None:
    """Anthropic ``thinking`` → OpenAI ``reasoning_effort``."""
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        return "none"
    if isinstance(output_config, dict) and output_config.get("effort") in ("minimal", "low", "medium", "high", "xhigh", "max"):
        return output_config["effort"]
    if not isinstance(thinking, dict):
        return None
    try:
        budget = int(thinking.get("budget_tokens") or 0)
    except (TypeError, ValueError):
        # A malformed budget must not 500 the request; treat it as absent.
        return None
    if not budget:
        return None
    return "low" if budget < 2048 else "medium" if budget < 8192 else "high"


def effort_from_ollama(think) -> str | None:
    """Ollama ``think`` (bool or named effort) → OpenAI ``reasoning_effort``."""
    if think is False:
        return "none"
    if isinstance(think, str) and think.lower() in ("minimal", "low", "medium", "high", "xhigh", "max"):
        return think.lower()
    return None


def anthropic_to_openai(body: dict) -> dict:
    msgs: list[dict] = []
    system = body.get("system")
    if system:
        msgs.append({"role": "system", "content": _blocks_text(system)})
    for m in body.get("messages") or []:
        role, content = m.get("role", "user"), m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = [{"id": b.get("id") or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input") or {})}}
                     for b in content if b.get("type") == "tool_use"]
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
            continue
        parts: list[dict] = []
        for b in content:
            t = b.get("type")
            if t == "text":
                parts.append({"type": "text", "text": b.get("text", "")})
            elif t == "image":
                img = _image_part(b)
                if img:
                    parts.append(img)
            elif t == "tool_result":
                result = b.get("content")
                text = result if isinstance(result, str) else _blocks_text(result)
                if b.get("is_error"):
                    text = f"Error: {text}"
                msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": text})
        if parts:
            only_text = all(p["type"] == "text" for p in parts)
            msgs.append({"role": "user", "content": "\n".join(p["text"] for p in parts) if only_text else parts})

    out: dict = {"messages": msgs, "stream": bool(body.get("stream"))}
    effort = effort_from_anthropic(body.get("thinking"), body.get("output_config"))
    if effort:
        out["reasoning_effort"] = effort
    if body.get("max_tokens"):
        out["max_tokens"] = body["max_tokens"]
    for k in ("temperature", "top_p"):
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    if body.get("tools"):
        out["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                          "parameters": t.get("input_schema") or {"type": "object"}}}
                        for t in body["tools"] if t.get("name")]
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        kind = tc.get("type")
        if kind == "any":
            out["tool_choice"] = "required"
        elif kind == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": tc.get("name")}}
        elif kind in ("auto", "none"):
            out["tool_choice"] = kind
    return out


def _parse_args(raw: str | None) -> dict:
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {"value": v}
    except ValueError:
        return {"_raw": raw}


def openai_to_anthropic(resp: dict, model: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict] = []
    reasoning = msg.get("reasoning_content") or msg.get("thinking")
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        content.append({"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
                        "name": fn.get("name"), "input": _parse_args(fn.get("arguments"))})
    usage = resp.get("usage") or {}
    finish = choice.get("finish_reason") or "stop"
    if msg.get("tool_calls"):
        finish = "tool_calls"
    return {
        "id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant", "model": model,
        "content": content or [{"type": "text", "text": ""}],
        "stop_reason": STOP_REASONS.get(finish, "end_turn"), "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def openai_stream_to_anthropic(chunks: AsyncIterator[dict], model: str) -> AsyncIterator[str]:
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
    index = -1
    open_kind: str | None = None
    tool_slots: dict[int, int] = {}
    finish = "stop"
    usage: dict = {}

    def close() -> str | None:
        nonlocal open_kind
        if open_kind is None:
            return None
        open_kind = None
        return _sse("content_block_stop", {"type": "content_block_stop", "index": index})

    async for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            thought = delta.get("reasoning_content") or delta.get("thinking")
            if thought:
                if open_kind != "thinking":
                    c = close()
                    if c:
                        yield c
                    index += 1
                    open_kind = "thinking"
                    yield _sse("content_block_start", {"type": "content_block_start", "index": index,
                                                       "content_block": {"type": "thinking", "thinking": "",
                                                                         "signature": ""}})
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                   "delta": {"type": "thinking_delta", "thinking": thought}})
            text = delta.get("content")
            if text:
                if open_kind != "text":
                    c = close()
                    if c:
                        yield c
                    index += 1
                    open_kind = "text"
                    yield _sse("content_block_start", {"type": "content_block_start", "index": index,
                                                       "content_block": {"type": "text", "text": ""}})
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                   "delta": {"type": "text_delta", "text": text}})
            for tc in delta.get("tool_calls") or []:
                slot = tc.get("index", 0)
                fn = tc.get("function") or {}
                if slot not in tool_slots:
                    c = close()
                    if c:
                        yield c
                    index += 1
                    tool_slots[slot] = index
                    open_kind = "tool"
                    yield _sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": {
                        "type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
                        "name": fn.get("name") or "", "input": {}}})
                if fn.get("arguments"):
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": tool_slots[slot],
                                                       "delta": {"type": "input_json_delta",
                                                                 "partial_json": fn["arguments"]}})
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    c = close()
    if c:
        yield c
    if tool_slots:
        finish = "tool_calls"
    yield _sse("message_delta", {"type": "message_delta",
                                 "delta": {"stop_reason": STOP_REASONS.get(finish, "end_turn"), "stop_sequence": None},
                                 "usage": {"output_tokens": usage.get("completion_tokens", 0)}})
    yield _sse("message_stop", {"type": "message_stop"})


# =============================================================================================
# Ollama ⇄ OpenAI
# =============================================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ollama_chat_to_openai(body: dict) -> dict:
    msgs = []
    for m in body.get("messages") or []:
        images = m.get("images") or []
        if images:
            parts: list[dict] = [{"type": "text", "text": m.get("content", "")}]
            parts += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in images]
            msgs.append({"role": m.get("role", "user"), "content": parts})
        else:
            msg = {"role": m.get("role", "user"), "content": m.get("content", "")}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                      "function": {"name": tc["function"]["name"],
                                                   "arguments": json.dumps(tc["function"].get("arguments") or {})}}
                                     for i, tc in enumerate(m["tool_calls"])]
            msgs.append(msg)
    out: dict = {"messages": msgs, "stream": body.get("stream", True)}
    effort = effort_from_ollama(body.get("think"))
    if effort:
        out["reasoning_effort"] = effort
    opts = body.get("options") or {}
    if opts.get("num_predict"):
        out["max_tokens"] = opts["num_predict"]
    for k in ("temperature", "top_p", "seed"):
        if opts.get(k) is not None:
            out[k] = opts[k]
    if opts.get("stop"):
        out["stop"] = opts["stop"]
    if body.get("tools"):
        out["tools"] = body["tools"]
    if body.get("format") == "json":
        out["response_format"] = {"type": "json_object"}
    elif isinstance(body.get("format"), dict):
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": body["format"]}}
    return out


def ollama_generate_to_openai(body: dict) -> dict:
    msgs = []
    if body.get("system"):
        msgs.append({"role": "system", "content": body["system"]})
    msgs.append({"role": "user", "content": body.get("prompt", ""),
                 **({"images": body["images"]} if body.get("images") else {})})
    return ollama_chat_to_openai({**body, "messages": msgs})


def openai_to_ollama_chat(resp: dict, model: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    out_msg: dict = {"role": "assistant", "content": msg.get("content") or ""}
    reasoning = msg.get("reasoning_content") or msg.get("thinking")
    if reasoning:
        out_msg["thinking"] = reasoning
    if msg.get("tool_calls"):
        out_msg["tool_calls"] = [{"function": {"name": tc["function"]["name"],
                                               "arguments": _parse_args(tc["function"].get("arguments"))}}
                                 for tc in msg["tool_calls"]]
    usage = resp.get("usage") or {}
    return {"model": model, "created_at": _now_iso(), "message": out_msg, "done": True,
            "done_reason": choice.get("finish_reason") or "stop", "total_duration": 0,
            "prompt_eval_count": usage.get("prompt_tokens", 0), "eval_count": usage.get("completion_tokens", 0)}


async def openai_stream_to_ollama(chunks: AsyncIterator[dict], model: str, generate: bool = False) -> AsyncIterator[str]:
    usage: dict = {}
    finish = "stop"
    started = time.time()
    async for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            thought = delta.get("reasoning_content") or delta.get("thinking")
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            if thought:
                if generate:
                    yield json.dumps({"model": model, "created_at": _now_iso(), "thinking": thought,
                                      "done": False}) + "\n"
                else:
                    yield json.dumps({"model": model, "created_at": _now_iso(),
                                      "message": {"role": "assistant", "content": "", "thinking": thought},
                                      "done": False}) + "\n"
            if text:
                if generate:
                    yield json.dumps({"model": model, "created_at": _now_iso(), "response": text, "done": False}) + "\n"
                else:
                    yield json.dumps({"model": model, "created_at": _now_iso(),
                                      "message": {"role": "assistant", "content": text}, "done": False}) + "\n"
    final: dict = {"model": model, "created_at": _now_iso(), "done": True, "done_reason": finish,
                   "total_duration": int((time.time() - started) * 1e9),
                   "prompt_eval_count": usage.get("prompt_tokens", 0), "eval_count": usage.get("completion_tokens", 0)}
    if generate:
        final["response"] = ""
    else:
        final["message"] = {"role": "assistant", "content": ""}
    yield json.dumps(final) + "\n"
