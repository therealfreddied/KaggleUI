"""Last-stage prompt shaping: ask for short answers without touching the caller's instructions."""

from __future__ import annotations

import copy
from typing import Any

TERSE_INSTRUCTION = "Answer concisely. Prefer the shortest complete response; skip preambles and restatements."

def apply_terse(body: dict[str, Any]) -> tuple[dict, str | None]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system" and \
                TERSE_INSTRUCTION in str(message.get("content") or ""):
            return body, None
    out = copy.deepcopy(body)
    messages = out["messages"]
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = f"{content.rstrip()}\n\n{TERSE_INSTRUCTION}"
            else:
                message["content"] = [*(content or []), {"type": "text", "text": TERSE_INSTRUCTION}]
            return out, "terse: added brevity instruction to the system prompt"
    out["messages"] = [{"role": "system", "content": TERSE_INSTRUCTION}, *messages]
    return out, "terse: added brevity instruction"
