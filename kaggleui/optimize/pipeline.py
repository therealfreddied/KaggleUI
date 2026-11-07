"""Request optimization: context shaping, output headroom, terse replies, and response caching."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .cache import ResponseCache, semantic_eligible
from .context import apply_context
from .headroom import apply_headroom
from .terse import apply_terse

if TYPE_CHECKING:
    from ..deploy.manager import Deployment
    from ..server.state import AppState

STREAM_NOT_CACHED = "cache: streaming requests bypass the response cache"

class Optimizer:
    """Stateless front door over the settings profiles; one instance lives on ``AppState``."""

    def __init__(self, state: AppState, profile=None, cache=None):
        self.state = state
        self._profile = profile
        self._cache_profile = cache
        self.cache = ResponseCache(state.paths.cache / "optimize")
        self.last_notes: list[str] = []

    @property
    def profile(self):
        return self._profile if self._profile is not None else self.state.settings.optimize

    @property
    def cache_profile(self):
        return self._cache_profile if self._cache_profile is not None else self.state.settings.cache

    # -- shaping ---------------------------------------------------------------------------
    def _ctx(self, dep: Deployment) -> int | None:
        value = dep.spec.options.get("ctx") or dep.info.get("ctx")
        try:
            return int(value) if value else None
        except (TypeError, ValueError):
            return None

    def prepare(self, dep: Deployment, body: dict[str, Any]) -> dict[str, Any]:
        profile = self.profile
        self.last_notes = []
        if not profile.enabled:
            return body
        out = body
        if profile.max_prompt_tokens:
            ctx = self._ctx(dep)
            budget = profile.max_prompt_tokens
            if ctx:
                budget = min(budget, max(ctx - 256, 256))
            out, note = apply_context(out, max_prompt_tokens=budget, strategy=profile.context_strategy)
            self._note(dep, note)
        out, note = apply_headroom(out, ctx=self._ctx(dep), margin=profile.output_headroom_margin)
        self._note(dep, note)
        if profile.terse_replies:
            out, note = apply_terse(out)
            self._note(dep, note)
        return out

    def _note(self, dep: Deployment, note: str | None) -> None:
        if note:
            self.last_notes.append(note)
            self._log(dep, note)

    def _log(self, dep: Deployment, line: str) -> None:
        """Same shape as ``DeploymentManager._log`` so the deployment drawer shows one stream."""
        dep.logs.append(line)
        self.state.bus.publish("log", {"id": dep.id, "line": line})

    # -- chat ------------------------------------------------------------------------------
    async def _embedding(self, text: str) -> list[float] | None:
        cache = self.cache_profile
        if not cache.semantic:
            return None
        dep = self._embedding_deployment()
        if dep is None or dep.backend is None:
            return None
        try:
            payload = await dep.backend.embeddings({"model": dep.served_name, "input": text})
            row = (payload.get("data") or [{}])[0]
            vector = row.get("embedding")
            return [float(x) for x in vector] if vector else None
        except Exception:
            return None

    def _embedding_deployment(self) -> Deployment | None:
        wanted = self.cache_profile.embedding_model
        candidates = [dep for dep in self.state.manager.deployments.values()
                      if dep.status == "ready" and dep.spec.modality == "text"
                      and dep.spec.options.get("embeddings")]
        if wanted:
            for dep in candidates:
                if wanted in (dep.served_name, dep.id):
                    return dep
        return candidates[0] if candidates else None

    @staticmethod
    def _prompt_text(body: dict[str, Any]) -> str:
        """Only user turns are embedded: system instructions are not part of the question."""
        parts = [m["content"] for m in body.get("messages") or []
                 if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)]
        return "\n".join(parts)

    async def chat(self, dep: Deployment, body: dict[str, Any]) -> dict[str, Any]:
        prepared = self.prepare(dep, body)
        cache = self.cache_profile
        if cache.exact:
            hit = self.cache.lookup(dep, prepared, ttl_s=cache.exact_ttl_s)
            if hit is not None:
                self._log(dep, "cache: exact hit")
                return hit
        vector = None
        if cache.semantic and semantic_eligible(prepared):
            vector = await self._embedding(self._prompt_text(prepared))
            if vector:
                hit = self.cache.lookup_semantic(dep, vector, threshold=cache.semantic_threshold,
                                                 ttl_s=cache.exact_ttl_s)
                if hit is not None:
                    self._log(dep, "cache: semantic hit")
                    return hit
        assert dep.backend is not None
        started = time.perf_counter()
        response = await dep.backend.chat(prepared)
        response.setdefault("model", dep.served_name)
        if cache.exact or vector:
            self.cache.store(dep, prepared, response, vector=vector)
        self._log(dep, f"chat: {time.perf_counter() - started:.2f}s, "
                       f"cache={'on' if cache.exact or cache.semantic else 'off'}")
        return response

    async def chat_stream(self, dep: Deployment, body: dict[str, Any]) -> AsyncIterator[dict]:
        self.prepare(dep, body)
        if self.cache_profile.exact or self.cache_profile.semantic:
            self._log(dep, STREAM_NOT_CACHED)
        assert dep.backend is not None
        async for chunk in dep.backend.chat_stream(body):
            yield chunk

    def stats(self) -> dict[str, Any]:
        return {"cache": self.cache.stats.as_dict(), "notes": list(self.last_notes),
                "enabled": self.profile.enabled}
