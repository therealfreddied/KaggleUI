"""Turn inline ``think``-style tags into structured reasoning fields.

Some models emit their reasoning inside the visible content. Front doors that speak a structured field
(OpenAI ``reasoning_content``, Anthropic ``thinking`` blocks, Ollama ``message.thinking``) should see it
there instead of in the answer text. Everything here is streaming-safe: the parser only holds back a
partial tag prefix, so added latency is bounded by the longest tag name.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

TAGS = ("think", "thinking", "thought", "internal_thought")
WRAPPERS = ("content",)
MAX_BLOCKS = 10
LONGEST = max(len(tag) for tag in TAGS + WRAPPERS)

_OPEN = re.compile(r"<(" + "|".join(TAGS) + r")\b[^>]*>", re.IGNORECASE)
_CLOSE = {tag: re.compile(r"</" + tag + r"\s*>", re.IGNORECASE) for tag in TAGS}
_WRAP_OPEN = re.compile(r"<content\b[^>]*>", re.IGNORECASE)
_WRAP_CLOSE = re.compile(r"</content\s*>", re.IGNORECASE)
_OPEN_PREFIX = re.compile(r"<(" + "|".join(TAGS + WRAPPERS) + r")\b[^>]*$", re.IGNORECASE)
_NEWLINES = re.compile(r"\n{3,}")

def _fence_mask(text: str) -> list[bool]:
    """True for every offset inside a triple-backtick fence (tags there are literal text)."""
    mask = [False] * len(text)
    pos = 0
    start = -1
    while True:
        found = text.find("```", pos)
        if found < 0:
            break
        if start < 0:
            start = found
        else:
            for i in range(start, found + 3):
                mask[i] = True
            start = -1
        pos = found + 3
    if start >= 0:
        for i in range(start, len(text)):
            mask[i] = True
    return mask

def _clean_reasoning(blocks: list[str]) -> str:
    kept = [b.strip() for b in blocks if b.strip()]
    return _NEWLINES.sub("\n\n", "\n\n".join(kept[:MAX_BLOCKS]))

def extract(text: str) -> tuple[str, str]:
    """Split ``text`` into ``(content, reasoning)``. An unclosed tag sends the tail to reasoning."""
    if not text or "<" not in text:
        return text, ""
    mask = _fence_mask(text)
    opens = [m for m in _OPEN.finditer(text) if not mask[m.start()]]
    content: list[str] = []
    reasoning: list[str] = []
    pos = 0
    for match in opens:
        if match.start() < pos:
            continue
        tag = match.group(1).lower()
        close = _CLOSE[tag].search(text, match.end())
        content.append(text[pos:match.start()])
        if close is None:
            reasoning.append(text[match.end():])
            pos = len(text)
            break
        reasoning.append(text[match.end():close.start()])
        pos = close.end()
        if len(reasoning) >= MAX_BLOCKS:
            break
    content.append(text[pos:])
    visible = _WRAP_CLOSE.sub("", _WRAP_OPEN.sub("", "".join(content)))
    thought = _clean_reasoning(reasoning)
    return (visible.strip() if thought else visible), thought

def _partial_prefix(buffer: str) -> str:
    """The tail of ``buffer`` that could still become an opening tag; hold it back."""
    cut = buffer.rfind("<")
    if cut < 0:
        return ""
    tail = buffer[cut:]
    if ">" in tail:
        return ""
    if _OPEN_PREFIX.fullmatch(tail):
        return tail
    lowered = tail[1:].lower()
    return tail if any(tag.startswith(lowered) for tag in TAGS + WRAPPERS) else ""

def _partial_suffix(buffer: str, needles: list[str]) -> str:
    """The tail of ``buffer`` that could still become one of ``needles``."""
    lowered = buffer.lower()
    for size in range(min(LONGEST, len(buffer)), 0, -1):
        tail = lowered[-size:]
        if any(needle.startswith(tail) for needle in needles):
            return buffer[-size:]
    return ""

class StreamParser:
    """Incremental tag splitter. ``feed`` returns ``(kind, text)`` pieces, ``kind`` in content|reasoning."""

    def __init__(self) -> None:
        self._buffer = ""
        self._tag: str | None = None
        self._fence = False
        self._reason_tail = ""

    @staticmethod
    def _flush_content(text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        if "```" not in text:
            text = _WRAP_CLOSE.sub("", _WRAP_OPEN.sub("", text))
        return [("content", text)] if text else []

    def _reason(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        if self._reason_tail.endswith("\n\n"):
            text = text.lstrip("\n")
        text = _NEWLINES.sub("\n\n", text)
        self._reason_tail = text[-2:]
        return [("reasoning", text)] if text else []

    def feed(self, delta: str) -> list[tuple[str, str]]:
        if not delta:
            return []
        self._buffer += delta
        pieces: list[tuple[str, str]] = []
        while self._buffer:
            buffer = self._buffer
            if self._tag is not None:
                close = _CLOSE[self._tag].search(buffer)
                if close is not None:
                    pieces += self._reason(buffer[:close.start()])
                    self._buffer = buffer[close.end():]
                    self._tag = None
                    continue
                keep = _partial_suffix(buffer, [f"</{self._tag}"])
                head = buffer[:-len(keep)] if keep else buffer
                self._buffer = buffer[len(head):]
                pieces += self._reason(head)
                break
            if self._fence:
                end = buffer.find("```")
                if end < 0:
                    keep = _partial_suffix(buffer, ["```"])
                    head = buffer[:-len(keep)] if keep else buffer
                    self._buffer = buffer[len(head):]
                    pieces += self._flush_content(head)
                    break
                pieces += self._flush_content(buffer[:end + 3])
                self._buffer = buffer[end + 3:]
                self._fence = False
                continue
            match = _OPEN.search(buffer)
            fence = buffer.find("```")
            if match is not None and (fence < 0 or match.start() < fence):
                pieces += self._flush_content(buffer[:match.start()])
                self._tag = match.group(1).lower()
                self._buffer = buffer[match.end():]
                continue
            if fence >= 0:
                pieces += self._flush_content(buffer[:fence + 3])
                self._buffer = buffer[fence + 3:]
                self._fence = True
                continue
            keep = _partial_prefix(buffer) or _partial_suffix(buffer, ["</content>", "```"])
            head = buffer[:-len(keep)] if keep else buffer
            self._buffer = buffer[len(head):]
            pieces += self._flush_content(head)
            break
        return pieces

    def flush(self) -> list[tuple[str, str]]:
        rest, self._buffer = self._buffer, ""
        if self._tag is not None:
            self._tag = None
            return self._reason(rest)
        return self._flush_content(rest)

def normalize_response(resp: dict) -> dict:
    """Move inline reasoning out of ``choices[].message.content`` into ``reasoning_content``."""
    for choice in resp.get("choices") or []:
        message = choice.get("message") or {}
        if not isinstance(message, dict) or message.get("reasoning_content") or message.get("thinking"):
            continue
        content = message.get("content")
        if not isinstance(content, str) or "<" not in content:
            continue
        visible, thought = extract(content)
        if not thought:
            continue
        message["content"] = visible
        message["reasoning_content"] = thought
    return resp

def _split(pieces: list[tuple[str, str]]) -> tuple[str, str]:
    return ("".join(text for kind, text in pieces if kind == "content"),
            "".join(text for kind, text in pieces if kind == "reasoning"))

async def normalize_stream(chunks: AsyncIterator[dict]) -> AsyncIterator[dict]:
    """Rewrite each chunk's delta so reasoning arrives in ``reasoning_content``."""
    parser: StreamParser | None = None
    async for chunk in chunks:
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            text = delta.get("content")
            if delta.get("reasoning_content") or delta.get("thinking"):
                continue
            if not isinstance(text, str) or not text:
                continue
            if parser is None:
                parser = StreamParser()
            content, thought = _split(parser.feed(text))
            delta["content"] = content
            if thought:
                delta["reasoning_content"] = thought
        yield chunk
    if parser is not None:
        content, thought = _split(parser.flush())
        if content or thought:
            delta: dict = {"content": content}
            if thought:
                delta["reasoning_content"] = thought
            yield {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
