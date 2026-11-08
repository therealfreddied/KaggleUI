"""Anthropic Messages API: proxied natively to Ollama, translated to OpenAI chat for other engines."""

from __future__ import annotations

import json

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .common import (
    SSE_HEADERS,
    GatewayError,
    anthropic_error,
    check_access,
    find,
    require_messages,
    state_of,
    wrap,
)
from .reasoning import normalize_response, normalize_stream
from .translate import anthropic_to_openai, openai_stream_to_anthropic, openai_to_anthropic

router = APIRouter()
_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10))


async def _body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception as e:
        raise GatewayError(400, "Request body must be JSON") from e
    if not isinstance(body, dict) or not body.get("messages"):
        raise GatewayError(400, "'messages' is required")
    require_messages(body, anthropic=True)
    return body


@router.post("/v1/messages")
async def messages(request: Request):
    try:
        check_access(request, "anthropic")
        body = await _body(request)
        dep = await find(request, body.get("model"), "text")
        backend = dep.backend
        assert backend
        served = dep.served_name

        if "anthropic" in backend.native and backend.base_url:
            upstream = {**body, "model": backend.upstream_model}
            url = f"{backend.base_url}/v1/messages"
            headers = {"content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": "kaggleui"}
            if body.get("stream"):
                async def relay():
                    async with _client.stream("POST", url, json=upstream, headers=headers) as r:
                        if r.status_code >= 400:
                            err = (await r.aread()).decode(errors="replace")[:500]
                            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': err}})}\n\n"
                            return
                        async for line in r.aiter_lines():
                            yield line.replace(backend.upstream_model or "\0", served) + "\n"

                return StreamingResponse(relay(), media_type="text/event-stream", headers=SSE_HEADERS)
            r = await _client.post(url, json=upstream, headers=headers)
            data = r.json()
            if isinstance(data, dict) and data.get("model"):
                data["model"] = served
            return JSONResponse(data, status_code=r.status_code)

        oai = anthropic_to_openai(body)
        optimize = state_of(request).optimize
        if body.get("stream"):
            stream = normalize_stream(optimize.chat_stream(dep, oai))
            return StreamingResponse(openai_stream_to_anthropic(stream, served),
                                     media_type="text/event-stream", headers=SSE_HEADERS)
        return openai_to_anthropic(normalize_response(await optimize.chat(dep, oai)), served)
    except Exception as e:  # noqa: BLE001
        return anthropic_error(wrap(e))


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        check_access(request, "anthropic")
        body = await _body(request)
        text = json.dumps(body.get("messages")) + json.dumps(body.get("system") or "") + json.dumps(body.get("tools") or [])
        return {"input_tokens": max(1, len(text) // 4)}
    except Exception as e:  # noqa: BLE001
        return anthropic_error(wrap(e))
