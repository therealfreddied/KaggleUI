"""Drop a harness's static instruction sections from system messages.

Only messages that are recognisably from a supported harness are touched, and only whole headings from
the tables below are removed. Everything else is forwarded byte-for-byte: an unrecognised prompt costs
nothing and changes nothing.
"""

from __future__ import annotations

import re

_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")

# Anchors are stable openings/harness identifiers; section titles are matched case-insensitively and
# bounded to the heading itself, never to phrasing inside the body.
HARNESSES: dict[str, dict] = {
    "claude_code": {
        "markers": ("you are claude code", "anthropic's official cli"),
        "sections": (
            "# tone and style",
            "# proactive vs. agentic",
            "# following conventions",
            "# code style",
            "# task management",
            "# doing tasks",
            "# code references",
        ),
    },
}

def _level(line: str) -> int:
    match = _HEADING.match(line)
    return len(match.group(1)) if match else 0

def _sections(system_text: str) -> tuple[str, ...]:
    lowered = system_text.lower()
    for spec in HARNESSES.values():
        if all(marker in lowered for marker in spec["markers"]):
            return spec["sections"]
    return ()

def strip_harness(system_text: str) -> tuple[str, int]:
    """Return ``(system_text, removed_chars)`` after dropping the harness's static sections."""
    sections = _sections(system_text)
    if not sections:
        return system_text, 0
    kept: list[str] = []
    skipping_at = 0
    for line in system_text.splitlines(keepends=True):
        level = _level(line)
        if level:
            title = "#" * level + " " + _HEADING.match(line).group(2).strip().lower()
            if title in sections:
                skipping_at = level
                continue
            if skipping_at and level <= skipping_at:
                skipping_at = 0
        if not skipping_at:
            kept.append(line)
    cleaned = "".join(kept)
    return cleaned, len(system_text) - len(cleaned)
