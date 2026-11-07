"""Reserve room for the reply so the prompt does not consume the whole context window."""

from __future__ import annotations

import copy
from typing import Any

from .context import estimate_tokens

MIN_COMPLETION = 16

def apply_headroom(body: dict[str, Any], *, ctx: int | None, margin: int = 256) -> tuple[dict, str | None]:
    """Cap ``max_tokens`` so prompt plus completion fits inside ``ctx``."""
    if not ctx:
        return body, None
    requested = body.get("max_tokens", body.get("max_completion_tokens"))
    prompt = estimate_tokens(body)
    room = ctx - prompt - max(margin, 0)
    if room <= 0:
        return body, "headroom: prompt leaves no room for a reply"
    if requested is None:
        if room >= ctx:
            return body, None
        out = copy.deepcopy(body)
        out["max_tokens"] = room
        return out, f"headroom: capped max_tokens at {room}"
    if int(requested) <= room:
        return body, None
    out = copy.deepcopy(body)
    capped = max(MIN_COMPLETION, room)
    if "max_completion_tokens" in out:
        out["max_completion_tokens"] = capped
    else:
        out["max_tokens"] = capped
    return out, f"headroom: max_tokens {requested} -> {capped}"
