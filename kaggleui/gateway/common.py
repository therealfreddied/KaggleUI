"""Gateway auth + model lookup shared by the OpenAI, Anthropic and Ollama-compatible routers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from ..engines.base import Unsupported, UpstreamError

if TYPE_CHECKING:
    from ..deploy.manager import Deployment
    from ..server.state import AppState

MODALITY_LABEL = {"text": "text model", "image": "image model", "video": "video model",
                  "tts": "text-to-speech model", "audio": "audio generation model", "mesh": "3D model",
                  "stt": "speech-to-text model", "world": "world model"}


class GatewayError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error", code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind
        self.code = code


def state_of(request: Request) -> AppState:
    return request.app.state.kui


def check_access(request: Request, surface: str) -> None:
    """Allow internal callers, panel admins, Studio users (access cookie), or valid API keys."""
    st = state_of(request)
    if st.updates.blocked:
        raise GatewayError(426, st.updates.status()["message"], "upgrade_required")
    auth = st.auth
    if auth.is_internal(request) or auth.is_admin(request):
        return
    if request.headers.get("x-kui-studio") == "1" and auth.has_access(request, st.settings):
        return
    api = st.settings.api
    if not api.enabled or not getattr(api, surface, False):
        raise GatewayError(403, f"The {surface} API is turned off in KaggleUI (API settings).", "permission_error")
    if api.require_key and not auth.check_api_key(request):
        ip = auth.client_ip(request)
        if auth.is_rate_limited(ip):
            raise GatewayError(429, "Too many failed attempts. Try again in a few minutes.", "rate_limit_error")
        auth.record_failure(ip)
        raise GatewayError(401, "Invalid or missing API key. Send it as 'Authorization: Bearer <key>' or 'x-api-key'.",
                           "authentication_error", "invalid_api_key")


async def find(request: Request, model: str | None, modality: str) -> Deployment:
    """Resolve a request's model, cold-starting an on-demand deployment when needed."""
    st = state_of(request)
    dep = st.manager.resolve(model, modality)
    if dep is None:
        running = [d.served_name for d in st.manager.ready(modality)]
        hint = f" Running: {', '.join(running)}." if running else f" No {MODALITY_LABEL[modality]} is running yet."
        raise GatewayError(404, f"Model '{model}' not found.{hint}", "not_found_error", "model_not_found")
    if dep.status == "cold" or dep.backend is None:
        try:
            await st.manager.ensure_ready(dep)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean API error
            raise GatewayError(503, f"{dep.served_name} could not start: {exc}", "api_error") from exc
    st.residency.touch(dep)
    return dep


def require_messages(body: dict, *, anthropic: bool = False) -> None:
    """Reject malformed message lists here, so translation never sees a shape it cannot handle."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GatewayError(400, "'messages' must be a non-empty array")
    roles = {"user", "assistant"} if anthropic else {"user", "assistant", "system", "tool", "developer"}
    for message in messages:
        if not isinstance(message, dict):
            raise GatewayError(400, "each entry of 'messages' must be an object")
        role = message.get("role")
        if role not in roles:
            raise GatewayError(400, f"unsupported message role: {role!r}")
        if anthropic and "content" not in message:
            raise GatewayError(400, "each message needs 'content'")

def openai_error(e: GatewayError) -> JSONResponse:
    return JSONResponse({"error": {"message": e.message, "type": e.kind, "code": e.code}}, status_code=e.status)


def anthropic_error(e: GatewayError) -> JSONResponse:
    kind = {"invalid_request_error": "invalid_request_error", "not_found_error": "not_found_error",
            "authentication_error": "authentication_error", "permission_error": "permission_error",
            "rate_limit_error": "rate_limit_error"}.get(e.kind, "api_error")
    return JSONResponse({"type": "error", "error": {"type": kind, "message": e.message}}, status_code=e.status)


def ollama_error(e: GatewayError) -> JSONResponse:
    return JSONResponse({"error": e.message}, status_code=e.status)


def wrap(exc: Exception) -> GatewayError:
    if isinstance(exc, GatewayError):
        return exc
    if isinstance(exc, Unsupported):
        return GatewayError(400, f"This model doesn't support {exc}.")
    if isinstance(exc, UpstreamError):
        detail = exc.detail
        if isinstance(detail, dict):
            detail = (detail.get("error") or {}).get("message") if isinstance(detail.get("error"), dict) else \
                detail.get("error") or detail.get("detail") or json.dumps(detail)
        return GatewayError(exc.status if 400 <= exc.status < 600 else 502, f"Model server error: {detail}", "api_error")
    if isinstance(exc, HTTPException):
        return GatewayError(exc.status_code, str(exc.detail))
    return GatewayError(500, f"{type(exc).__name__}: {exc}", "api_error")


async def sse_openai(chunks: AsyncIterator[dict], served: str) -> AsyncIterator[str]:
    try:
        async for chunk in chunks:
            chunk["model"] = served
            yield "data: " + json.dumps(chunk) + "\n\n"
    except Exception as e:  # mid-stream failure: surface it in-band
        err = wrap(e)
        yield "data: " + json.dumps({"error": {"message": err.message, "type": err.kind}}) + "\n\n"
    yield "data: [DONE]\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
