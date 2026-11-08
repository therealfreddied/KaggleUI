"""Best-effort model facts from models.dev.

The public catalog publishes one JSON document that repeats the same open-weights model under every
provider that serves it. KaggleUI uses it only to fill gaps: a missing context window, a missing
output cap, and a few facts worth showing next to the model. Closed API models are skipped so a
local repo can never be matched against an endpoint that does not ship weights.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

URL = "https://models.dev/api.json"
CACHE_KEY = "models-dev"
CACHE_TTL = 6 * 3600
MAX_BYTES = 12 * 1024 * 1024


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def normalized(value: str | None) -> str:
    """Punctuation-insensitive join key (``Qwen/Qwen3-8B`` and ``qwen3:8b`` -> ``qwen38b``)."""
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def huggingface_repo(url: str | None) -> str | None:
    """The repo id in a ``huggingface.co/<repo>/resolve/...`` download URL."""
    if not url:
        return None
    path = str(url).split("?", 1)[0].split("#", 1)[0]
    parts = [part for part in path.split("/") if part]
    if "resolve" not in parts:
        return None
    index = parts.index("resolve")
    if index < 2:
        return None
    return "/".join(parts[index - 2:index])


def _entry(raw: dict, model_id: str) -> dict:
    limit = raw.get("limit") if isinstance(raw.get("limit"), dict) else {}
    entry = {"id": model_id, "open_weights": True,
             "context_length": _int(limit.get("context")),
             "max_output_tokens": _int(limit.get("output"))}
    for key in ("name", "family", "knowledge", "release_date", "last_updated", "modalities"):
        if raw.get(key):
            entry[key] = raw[key]
    for key in ("attachment", "reasoning", "tool_call"):
        if raw.get(key) is True:
            entry[key] = True
    return entry


def _merge(into: dict, entry: dict) -> dict:
    """Two providers serving the same weights: keep the larger declared limits and the extra facts."""
    for key in ("context_length", "max_output_tokens"):
        if entry.get(key):
            into[key] = max(into.get(key) or 0, entry[key])
    for key, value in entry.items():
        if key in ("context_length", "max_output_tokens", "id") or value in (None, "", False):
            continue
        into.setdefault(key, value)
    return into


def index(data: Any) -> dict[str, dict]:
    """Normalized aliases -> merged facts, restricted to entries that publish open weights.

    Short aliases (name, basename) are only published when they identify exactly one model, so a
    local repo called ``model`` cannot inherit another provider's limits.
    """
    entries: dict[str, dict] = {}
    aliases: dict[str, set[str]] = {}
    if not isinstance(data, dict):
        return entries
    for provider in data.values():
        models = provider.get("models") if isinstance(provider, dict) else None
        if not isinstance(models, dict):
            continue
        for model_id, raw in models.items():
            if not isinstance(raw, dict) or raw.get("open_weights") is not True:
                continue
            full = normalized(model_id)
            entry = _entry(raw, model_id)
            if not full or (not entry.get("context_length") and not entry.get("max_output_tokens")):
                continue
            if full in entries:
                _merge(entries[full], entry)
            else:
                entries[full] = entry
            for alias in {full, normalized(model_id.rsplit("/", 1)[-1]), normalized(entry.get("name"))}:
                if alias:
                    aliases.setdefault(alias, set()).add(full)
    out = dict(entries)
    for alias, owners in aliases.items():
        if len(owners) == 1 and alias not in out:
            out[alias] = entries[next(iter(owners))]
    return out


def lookup(index: dict[str, dict], *, url: str | None = None, repo: str | None = None,
           name: str | None = None, ref: str | None = None) -> dict | None:
    """Exact joins only: weights URL, repo id, normalized name, then the id/ref basename."""
    parsed = huggingface_repo(url)
    candidates = [parsed, parsed.rsplit("/", 1)[-1] if parsed else None, repo, name, ref,
                  ref.rsplit(":", 1)[0] if ref else None, repo.rsplit("/", 1)[-1] if repo else None]
    for candidate in candidates:
        key = normalized(candidate)
        if key and key in index:
            return index[key]
    return None


def apply(item: dict, entry: dict | None) -> dict:
    """Fill gaps only; a value already read from the model's own metadata always wins."""
    if not entry:
        return item
    if not item.get("context_length") and entry.get("context_length"):
        item["context_length"] = entry["context_length"]
    if not item.get("max_output_tokens") and entry.get("max_output_tokens"):
        item["max_output_tokens"] = entry["max_output_tokens"]
    facts = {key: entry[key] for key in ("id", "name", "family", "knowledge", "release_date",
                                         "last_updated", "attachment", "reasoning", "tool_call",
                                         "open_weights", "modalities") if entry.get(key) is not None}
    if entry.get("context_length"):
        facts["context"] = entry["context_length"]
    if entry.get("max_output_tokens"):
        facts["output"] = entry["max_output_tokens"]
    if facts:
        item["models_dev"] = facts
    return item


async def fetch(client: httpx.AsyncClient) -> dict[str, dict]:
    """The indexed document; an unreachable source is a silent empty result, cached like success."""
    try:
        response = await client.get(URL, timeout=20, follow_redirects=True)
        response.raise_for_status()
        if len(response.content) > MAX_BYTES:
            return {}
        return index(response.json())
    except (httpx.HTTPError, ValueError):
        return {}
