"""Prompt-length shaping: drop old turns before the server has to reject the request.

The estimate is deliberately crude (four characters per token) and never inspects language: a
CJK-heavy prompt is over-estimated, which trims earlier than needed instead of silently overflowing.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

SYSTEM_WARNING = "system prompt exceeds budget"

def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return ""

def estimate_tokens(body: dict[str, Any]) -> int:
    """Upper bound on prompt tokens: serialized messages plus tool schemas, 4 chars per token."""
    messages = body.get("messages") or []
    size = sum(len(_text_of(m)) + 8 for m in messages if isinstance(m, dict))
    for key in ("tools", "functions"):
        if body.get(key):
            size += len(json.dumps(body[key]))
    return max(1, math.ceil(size / 4))

def _split(messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest

def _keep_last(rest: list[dict], budget: int) -> list[dict]:
    kept: list[dict] = []
    used = 0
    for message in reversed(rest):
        cost = math.ceil(len(_text_of(message)) / 4) + 8
        if kept and used + cost > budget:
            break
        kept.append(message)
        used += cost
    return list(reversed(kept))

def apply_context(body: dict[str, Any], *, max_prompt_tokens: int, strategy: str = "balanced"
                  ) -> tuple[dict, str | None]:
    """Return a trimmed copy of ``body`` and an optional note for the deployment log."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body, None
    system, rest = _split(messages)
    if not rest:
        return body, None
    if estimate_tokens(body) <= max_prompt_tokens:
        return body, None
    system_cost = estimate_tokens({"messages": system}) if system else 0
    if system_cost >= max_prompt_tokens:
        # Instructions are never truncated; the engine may still reject the request.
        return body, SYSTEM_WARNING

    out = copy.deepcopy(body)
    budget = max_prompt_tokens - system_cost
    final_user = rest[-1] if rest[-1].get("role") == "user" else None
    if strategy == "head":
        trimmed = []
        used = 0
        for message in rest:
            if message is final_user:
                continue
            cost = math.ceil(len(_text_of(message)) / 4) + 8
            if trimmed and used + cost > budget:
                break
            trimmed.append(message)
            used += cost
        if final_user is not None and final_user not in trimmed:
            trimmed.append(final_user)
    else:
        trimmed = _keep_last(rest, budget)
        first_user = next((m for m in rest if m.get("role") == "user"), None)
        if strategy == "balanced" and first_user is not None and first_user not in trimmed:
            cost = math.ceil(len(_text_of(first_user)) / 4) + 8
            trimmed_cost = sum(math.ceil(len(_text_of(m)) / 4) + 8 for m in trimmed)
            if system_cost + trimmed_cost + cost <= max_prompt_tokens:
                trimmed = [first_user, *trimmed]
    dropped = len(rest) - len(trimmed)
    out["messages"] = [*system, *trimmed]
    note = f"context: dropped {dropped} message(s) to fit {max_prompt_tokens} tokens" if dropped else None
    return out, note
