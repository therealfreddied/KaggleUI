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

    def _apply_prompt(self, dep: Deployment, body: dict[str, Any], options: dict) -> dict[str, Any]:
        """Insert or replace the system message from model options, else the global default."""
        prompt = options.get("system_prompt") or self.profile.system_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            return body
        messages = list(body.get("messages") or [])
        index = next((i for i, m in enumerate(messages)
                      if isinstance(m, dict) and m.get("role") == "system"), None)
        if index is None:
            messages.insert(0, {"role": "system", "content": prompt})
        elif options.get("system_prompt_mode") == "replace":
            messages[index] = {**messages[index], "content": prompt}
        else:
            existing = str(messages[index].get("content") or "")
            messages[index] = {**messages[index], "content": f"{prompt}\n\n{existing}".strip()}
        return {**body, "messages": messages}

    def prepare(self, dep: Deployment, body: dict[str, Any]) -> dict[str, Any]:
        profile = self.profile
        options = dep.spec.options or {}
        self.last_notes = []
        out = self._apply_prompt(dep, body, options)
        compress = options.get("compress") or []
        if isinstance(compress, str):
            compress = [compress]
        messages = out.get("messages")
        if isinstance(messages, list):
            if "tools" in compress:
                from .compress_tools import compress_messages

                messages, cut = compress_messages(messages)
                if cut:
                    self._note(dep, f"compress: tools removed {cut} chars")
            if "prose" in compress:
                from .prose import compress_messages as compress_prose

                messages, cut = compress_prose(messages)
                if cut:
                    self._note(dep, f"compress: prose removed {cut} chars")
            if options.get("slim_harness"):
                from .harness import strip_harness

                removed = 0
                for message in messages:
                    if isinstance(message, dict) and message.get("role") == "system" \
                            and isinstance(message.get("content"), str):
                        message["content"], cut = strip_harness(message["content"])
                        removed += cut
                if removed:
                    self._note(dep, f"harness: removed {removed} chars of static instructions")
        if not profile.enabled and not any(
                options.get(k) is not None for k in ("terse", "max_prompt_tokens", "context_strategy")):
            return out
        # Per-model options win over the global profile; unset means "inherit the profile".
        terse = options.get("terse")
        terse_on = profile.terse_replies if terse is None else bool(terse)
        try:
            max_tokens = int(options["max_prompt_tokens"]) if options.get("max_prompt_tokens") \
                else profile.max_prompt_tokens
        except (TypeError, ValueError):
            max_tokens = profile.max_prompt_tokens
        strategy = options.get("context_strategy")
        if strategy not in ("head", "tail", "middle", "balanced"):
            strategy = profile.context_strategy
        if max_tokens:
            ctx = self._ctx(dep)
            budget = max_tokens
            if ctx:
                budget = min(budget, max(ctx - 256, 256))
            out, note = apply_context(out, max_prompt_tokens=budget, strategy=strategy)
            self._note(dep, note)
        out, note = apply_headroom(out, ctx=self._ctx(dep), margin=profile.output_headroom_margin)
        self._note(dep, note)
        if terse_on:
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
        mode = self.cache_mode(dep)
        exact = mode in ("exact", "semantic")
        semantic = mode == "semantic"
        if exact:
            hit = self.cache.lookup(dep, prepared, ttl_s=cache.exact_ttl_s)
            if hit is not None:
                self._log(dep, "cache: exact hit")
                return hit
        vector = None
        if semantic and semantic_eligible(prepared):
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
        if exact or vector:
            self.cache.store(dep, prepared, response, vector=vector)
        self._log(dep, f"chat: {time.perf_counter() - started:.2f}s, "
                       f"cache={'on' if exact or semantic else 'off'}")
        return response

    def cache_mode(self, dep: Deployment) -> str:
        """``off`` | ``exact`` | ``semantic``; a model's own option wins over the global profile."""
        value = (dep.spec.options or {}).get("cache")
        if value is False:
            # Opt-out only: a per-model flag never enables caching the global profile leaves off.
            return "off"
        if value in ("off", "exact", "semantic"):
            return str(value)
        if self.cache_profile.semantic:
            return "semantic"
        return "exact" if self.cache_profile.exact else "off"

    async def chat_stream(self, dep: Deployment, body: dict[str, Any]) -> AsyncIterator[dict]:
        self.prepare(dep, body)
        if self.cache_mode(dep) != "off":
            self._log(dep, STREAM_NOT_CACHED)
        assert dep.backend is not None
        async for chunk in dep.backend.chat_stream(body):
            yield chunk

    def stats(self) -> dict[str, Any]:
        return {"cache": self.cache.stats.as_dict(), "notes": list(self.last_notes),
                "enabled": self.profile.enabled}
