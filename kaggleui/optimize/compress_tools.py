"""Compress tool output before it reaches the model: ansi, duplicates, progress noise, long tails."""

from __future__ import annotations

import re

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
KEEP_HEAD = 60
KEEP_TAIL = 100
MAX_LINES = 200
NOISE = (
    re.compile(r"^\s*Requirement already satisfied:", re.IGNORECASE),
    re.compile(r"^\s*(Downloading|Using cached|Collecting)\b.*", re.IGNORECASE),
    re.compile(r"^\s*\d+%\s*\|"),
    re.compile(r"^\s*npm (WARN|notice)\b"),
)

def _dedupe(lines: list[str]) -> list[str]:
    out: list[str] = []
    repeat = 0
    previous: str | None = None
    for line in lines:
        if previous is not None and line == previous:
            repeat += 1
            continue
        if repeat:
            out.append(f"… (repeated {repeat + 1}×)")
            repeat = 0
        out.append(line)
        previous = line
    if repeat:
        out.append(f"… (repeated {repeat + 1}×)")
    return out

def compress_text(text: str) -> tuple[str, int]:
    """Return ``(compressed, removed_chars)``; never touches structured (JSON) tool results."""
    if len(text) < 400 or text.lstrip().startswith(("{", "[")):
        return text, 0
    lines = ANSI.sub("", text).splitlines()
    lines = [line for line in lines if not any(rule.match(line) for rule in NOISE)]
    lines = _dedupe(lines)
    if len(lines) > MAX_LINES:
        omitted = len(lines) - KEEP_HEAD - KEEP_TAIL
        marker = f"… {omitted} lines elided by tool-output compression …"
        lines = lines[:KEEP_HEAD] + [marker] + lines[-KEEP_TAIL:]
    out = "\n".join(lines)
    return out, max(0, len(text) - len(out))

def compress_messages(messages: list[dict]) -> tuple[list[dict], int]:
    """Apply ``compress_text`` to tool results inside an OpenAI-shaped message list."""
    removed = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            message["content"], cut = compress_text(message["content"])
            removed += cut
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                block["content"], cut = compress_text(str(block.get("content") or ""))
                removed += cut
    return messages, removed
