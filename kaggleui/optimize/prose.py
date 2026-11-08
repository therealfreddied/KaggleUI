"""Trim conversational padding from chat history.

Deliberately small: only stock acknowledgements and closing offers are removed, and only at the edges of
a message. Sentences in the middle of an answer are never rewritten, because there is no safe local edit
that preserves meaning for prose.
"""

from __future__ import annotations

import re

OPENERS = re.compile(
    r"^(?:(?:sure|certainly|of course|absolutely|great question)[!,.]?\s+"
    r"(?:(?:i'd be happy to help|here(?:'s| is)|i can help|let me)[^.\n]*[.!]\s*)?)+",
    re.IGNORECASE,
)
CLOSERS = re.compile(
    r"\n*(?:let me know if (?:you (?:have|need)|there(?:'s| is))[^.\n]*[.!]?"
    r"|feel free to (?:ask|reach out)[^.\n]*[.!]?"
    r"|hope (?:this|that) helps[^.\n]*[.!]?)\s*$",
    re.IGNORECASE,
)

def compress_text(text: str) -> tuple[str, int]:
    stripped = OPENERS.sub("", text)
    stripped = CLOSERS.sub("", stripped).rstrip()
    if not stripped.strip():
        return text, 0
    return stripped, max(0, len(text) - len(stripped))

def compress_messages(messages: list[dict]) -> tuple[list[dict], int]:
    removed = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"], cut = compress_text(content)
            removed += cut
    return messages, removed
