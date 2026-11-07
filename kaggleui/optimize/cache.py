"""Exact and semantic response caches for the optimize pipeline.

An exact hit is a byte-identical request body for the same deployment at the same model revision:
key schema v1, ``temperature == 0``, no ``seed`` and no ``tools`` (a sampled or tool-driven answer
must never be replayed). Entries are individual JSON files under ``paths.cache/exact`` written
atomically; the index is rebuilt from mtimes on boot, which is also the LRU order.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KEY_VERSION = 1
MAX_ENTRIES = 512
MAX_BYTES = 64 << 20

# Routing-only fields: dropping them lets a hit survive a client toggling streaming or a user tag.
_VOLATILE = frozenset({"stream", "stream_options", "user", "metadata"})

def cacheable(body: dict[str, Any]) -> bool:
    """Temperature 0 is the only reproducible setting; seeded or tool calls are not replayed."""
    temperature = body.get("temperature")
    return temperature is not None and float(temperature) == 0.0 and "seed" not in body and "tools" not in body

def semantic_eligible(body: dict[str, Any]) -> bool:
    if body.get("tools") or body.get("stream"):
        return False
    messages = body.get("messages") or []
    return not any(isinstance(m, dict) and isinstance(m.get("content"), list) for m in messages)

def _normalized(body: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in body.items() if k not in _VOLATILE}

def cache_key(dep, body: dict[str, Any]) -> str | None:
    if not cacheable(body):
        return None
    payload = {"v": KEY_VERSION, "model": dep.served_name, "model_rev": dep.spec.model.id,
               "body": _normalized(body)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"exact_hits": self.exact_hits, "semantic_hits": self.semantic_hits,
                "misses": self.misses, "stores": self.stores, "evictions": self.evictions}

@dataclass
class ResponseCache:
    root: Path
    max_entries: int = MAX_ENTRIES
    max_bytes: int = MAX_BYTES
    stats: CacheStats = field(default_factory=CacheStats)
    _semantic: list[tuple[str, list[float], float, dict]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.exact_dir = self.root / "exact"
        self.exact_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Path] = {p.stem: p for p in self.exact_dir.glob("*.json")}

    # -- exact -----------------------------------------------------------------------------
    def lookup(self, dep, body: dict[str, Any], *, ttl_s: int) -> dict | None:
        key = cache_key(dep, body)
        if key is None:
            self.stats.misses += 1
            return None
        path = self._index.get(key)
        if path is None or not path.exists():
            self._index.pop(key, None)
            self.stats.misses += 1
            return None
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            self._index.pop(key, None)
            self.stats.misses += 1
            return None
        if time.time() - float(record.get("stored_at", 0)) > ttl_s:
            path.unlink(missing_ok=True)
            self._index.pop(key, None)
            self.stats.misses += 1
            return None
        try:
            os.utime(path, None)
        except OSError:
            pass
        self.stats.exact_hits += 1
        return record.get("response")

    def _prune(self) -> None:
        rows = sorted(self._index.values(), key=lambda p: p.stat().st_mtime if p.exists() else 0)
        total = sum(p.stat().st_size for p in rows if p.exists())
        while rows and (len(rows) > self.max_entries or total > self.max_bytes):
            victim = rows.pop(0)
            try:
                total -= victim.stat().st_size
            except OSError:
                pass
            victim.unlink(missing_ok=True)
            self._index.pop(victim.stem, None)
            self.stats.evictions += 1

    def store(self, dep, body: dict[str, Any], response: dict, *, vector: list[float] | None = None) -> None:
        key = cache_key(dep, body)
        if key is not None:
            record = {"key": key, "stored_at": time.time(), "response": response}
            path = self.exact_dir / f"{key}.json"
            tmp = path.with_suffix(".tmp")
            with self._lock:
                tmp.write_text(json.dumps(record, ensure_ascii=False))
                os.replace(tmp, path)
                self._index[key] = path
                self.stats.stores += 1
                self._prune()
        if vector:
            namespace = f"{dep.served_name}@{dep.spec.model.id}"
            self._semantic.append((namespace, vector, time.time(), response))
            while len(self._semantic) > self.max_entries:
                self._semantic.pop(0)
                self.stats.evictions += 1

    # -- semantic --------------------------------------------------------------------------
    def lookup_semantic(self, dep, vector: list[float], *, threshold: float, ttl_s: int) -> dict | None:
        namespace = f"{dep.served_name}@{dep.spec.model.id}"
        now = time.time()
        best: tuple[float, dict] | None = None
        self._semantic = [row for row in self._semantic if now - row[2] <= ttl_s]
        for row_ns, stored_vec, _stored_at, response in self._semantic:
            if row_ns != namespace:
                continue
            score = cosine(vector, stored_vec)
            if score >= threshold and (best is None or score > best[0]):
                best = (score, response)
        if best is None:
            return None
        self.stats.semantic_hits += 1
        return best[1]

    def clear(self) -> None:
        with self._lock:
            for path in list(self._index.values()):
                path.unlink(missing_ok=True)
            self._index.clear()
            self._semantic.clear()
