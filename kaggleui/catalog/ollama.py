"""Ollama library discovery.

There's no public search API, so this parses ollama.com/search (with a bundled snapshot as a
fallback) and asks the registry for exact download sizes.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import httpx

from ..util import parse_size

SITE = "https://ollama.com"
REGISTRY = "https://registry.ollama.ai/v2"
SNAPSHOT = Path(__file__).with_name("ollama_snapshot.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (kaggleui; +https://github.com/therealfreddied/KaggleUI)"}

_LI_RE = re.compile(r"<li\s+class=\"border-b[\s\S]*?</li>")
_NAME_RE = re.compile(r'href="/library/([^"]+)"')
_DESC_RE = re.compile(r'<p[^>]*title="([^"]*)"')
_CAP_RE = re.compile(r"text-indigo-600[^>]*>([^<]+)<")
_SIZE_RE = re.compile(r"text-blue-600[^>]*>([^<]+)<")
_DL_RE = re.compile(r'title="([\d,]+) downloads"')
_TAG_RE = re.compile(r'<a href="/library/([^":]+):([^"]+)" class="md:hidden[\s\S]*?</a>')


def parse_search(page: str) -> list[dict]:
    out = []
    for m in _LI_RE.finditer(page):
        block = m.group(0)
        name = _NAME_RE.search(block)
        if not name:
            continue
        desc = _DESC_RE.search(block)
        dl = _DL_RE.search(block)
        out.append({
            "name": name.group(1),
            "description": html.unescape(desc.group(1)) if desc else "",
            "capabilities": [c.strip() for c in _CAP_RE.findall(block)],
            "sizes": [s.strip() for s in _SIZE_RE.findall(block)],
            "downloads": int(dl.group(1).replace(",", "")) if dl else 0,
        })
    return out


def parse_tags(page: str) -> list[dict]:
    tags = []
    seen = set()
    for m in _TAG_RE.finditer(page):
        tag = m.group(2)
        if tag in seen:
            continue
        seen.add(tag)
        text = re.sub(r"<[^>]+>", " ", m.group(0))
        ctx = re.search(r"(\d+[KM]?) context window", text)
        inputs = re.search(r"([A-Za-z, ]+) input", text)
        tags.append({
            "tag": tag,
            "size": parse_size(m.group(0).split("context window")[0].split("•", 1)[-1]),
            "context": ctx.group(1) if ctx else None,
            "input": inputs.group(1).strip() if inputs else None,
        })
    return tags


def is_local_model(item: dict) -> bool:
    """Cloud-only entries on ollama.com have neither sizes nor a download count."""
    return bool(item.get("sizes")) or item.get("downloads", 0) > 0


def parse_context(value: str | None) -> int | None:
    """Tokens from a tag label like ``128K``. K and M are binary, as ollama.com counts them."""
    match = re.fullmatch(r"(\d+)\s*([KM]?)", value or "", re.I)
    if not match:
        return None
    scale = {"": 1, "K": 1024, "M": 1024**2}[match.group(2).upper()]
    return int(match.group(1)) * scale


async def search(client: httpx.AsyncClient, query: str = "", capability: str | None = None,
                 order: str | None = None, page: int = 1) -> list[dict]:
    params = {}
    if query:
        params["q"] = query
    if capability:
        params["c"] = capability
    if order:
        params["o"] = order
    if page > 1:
        params["page"] = str(page)
    # Later pages are only served to htmx requests (the site's "load more").
    headers = {**HEADERS, "HX-Request": "true"} if page > 1 else HEADERS
    r = await client.get(f"{SITE}/search", params=params, headers=headers, timeout=15, follow_redirects=True)
    r.raise_for_status()
    return [i for i in parse_search(r.text) if is_local_model(i)]


def snapshot(query: str = "") -> list[dict]:
    try:
        items = json.loads(SNAPSHOT.read_text())
    except Exception:
        return []
    q = query.lower().strip()
    return [i for i in items if not q or q in i["name"].lower() or q in i.get("description", "").lower()]


async def tags(client: httpx.AsyncClient, name: str) -> list[dict]:
    r = await client.get(f"{SITE}/library/{name}/tags", headers=HEADERS, timeout=15, follow_redirects=True)
    r.raise_for_status()
    return parse_tags(r.text)


async def manifest_size(client: httpx.AsyncClient, name: str, tag: str = "latest") -> int | None:
    """Exact download size of ``name:tag`` from the registry (sum of layer sizes)."""
    path = name if "/" in name else f"library/{name}"
    r = await client.get(f"{REGISTRY}/{path}/manifests/{tag}",
                         headers={**HEADERS, "Accept": "application/vnd.docker.distribution.manifest.v2+json"},
                         timeout=15)
    if r.status_code != 200:
        return None
    layers = r.json().get("layers") or []
    return sum(int(layer.get("size") or 0) for layer in layers) or None


def split_ref(ref: str) -> tuple[str, str]:
    name, _, tag = ref.partition(":")
    return name, tag or "latest"
