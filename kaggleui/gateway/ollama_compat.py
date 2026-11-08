"""Ollama-compatible API (/api/chat, /api/generate, /api/tags, ...), for apps that speak Ollama."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import __version__
from .common import GatewayError, check_access, find, ollama_error, state_of, wrap
from .reasoning import normalize_response, normalize_stream
from .translate import (
    ollama_chat_to_openai,
    ollama_generate_to_openai,
    openai_stream_to_ollama,
    openai_to_ollama_chat,
)

router = APIRouter()
_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10))


async def _body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        body = {}
    return body if isinstance(body, dict) else {}


@router.get("/api/version")
async def version(request: Request):
    try:
        check_access(request, "ollama")
    except GatewayError as e:
        return ollama_error(e)
    return {"version": f"0.12.0-kaggleui-{__version__}"}


@router.get("/api/tags")
async def tags(request: Request):
    try:
        check_access(request, "ollama")
    except GatewayError as e:
        return ollama_error(e)
    models = []
    for d in state_of(request).manager.ready("text"):
        ts = datetime.fromtimestamp(d.ready_at or d.created_at, timezone.utc).isoformat()
        models.append({"name": d.served_name, "model": d.served_name, "modified_at": ts, "size": d.vram_estimate,
                       "digest": d.id, "details": {"format": "gguf" if d.spec.model.engine == "ollama" else "safetensors",
                                                   "family": "", "parameter_size": "", "quantization_level":
                                                   d.spec.model.variant or ""}})
    return {"models": models}


@router.get("/api/ps")
async def ps(request: Request):
    return await tags(request)


async def _native(request: Request, path: str, body: dict, dep) -> StreamingResponse | JSONResponse:
    backend = dep.backend
    upstream = {**body, "model": backend.upstream_model}
    url = f"{backend.base_url}{path}"
    if body.get("stream", True) and path in ("/api/chat", "/api/generate"):
        async def relay():
            async with _client.stream("POST", url, json=upstream) as r:
                async for line in r.aiter_lines():
                    if line:
                        yield line.replace(backend.upstream_model, dep.served_name) + "\n"

        return StreamingResponse(relay(), media_type="application/x-ndjson")
    r = await _client.post(url, json={**upstream, "stream": False} if path in ("/api/chat", "/api/generate") else upstream)
    try:
        data = r.json()
    except ValueError:
        return JSONResponse({"error": r.text[:300]}, status_code=r.status_code)
    if isinstance(data, dict) and "model" in data:
        data["model"] = dep.served_name
    return JSONResponse(data, status_code=r.status_code)


async def _chatlike(request: Request, generate: bool):
    check_access(request, "ollama")
    body = await _body(request)
    dep = await find(request, body.get("model"), "text")
    backend = dep.backend
    assert backend
    path = "/api/generate" if generate else "/api/chat"
    if "ollama" in backend.native and backend.base_url:
        return await _native(request, path, body, dep)
    oai = ollama_generate_to_openai(body) if generate else ollama_chat_to_openai(body)
    optimize = state_of(request).optimize
    if body.get("stream", True):
        stream = normalize_stream(optimize.chat_stream(dep, oai))
        return StreamingResponse(openai_stream_to_ollama(stream, dep.served_name, generate),
                                 media_type="application/x-ndjson")
    resp = normalize_response(await optimize.chat(dep, {**oai, "stream": False}))
    out = openai_to_ollama_chat(resp, dep.served_name)
    if generate:
        out["response"] = out.pop("message", {}).get("content", "")
    return out


@router.post("/api/chat")
async def chat(request: Request):
    try:
        return await _chatlike(request, generate=False)
    except Exception as e:  # noqa: BLE001
        return ollama_error(wrap(e))


@router.post("/api/generate")
async def generate(request: Request):
    try:
        return await _chatlike(request, generate=True)
    except Exception as e:  # noqa: BLE001
        return ollama_error(wrap(e))


@router.post("/api/embed")
@router.post("/api/embeddings")
async def embed(request: Request):
    try:
        check_access(request, "ollama")
        body = await _body(request)
        dep = await find(request, body.get("model"), "text")
        backend = dep.backend
        assert backend
        if "ollama" in backend.native and backend.base_url:
            return await _native(request, request.url.path, body, dep)
        inputs = body.get("input") if "input" in body else body.get("prompt", "")
        resp = await backend.embeddings({"input": inputs})
        vecs = [d["embedding"] for d in resp.get("data", [])]
        if request.url.path.endswith("/embeddings"):
            return {"embedding": vecs[0] if vecs else []}
        return {"model": dep.served_name, "embeddings": vecs}
    except Exception as e:  # noqa: BLE001
        return ollama_error(wrap(e))


@router.post("/api/show")
async def show(request: Request):
    try:
        check_access(request, "ollama")
        body = await _body(request)
        dep = await find(request, body.get("model") or body.get("name"), "text")
        if "ollama" in dep.backend.native and dep.backend.base_url:
            return await _native(request, "/api/show", {"model": dep.backend.upstream_model}, dep)
        caps = ["thinking" if cap == "reasoning" else cap for cap in dep.info.get("capabilities", [])]
        reason = dep.info.get("reasoning") or {}
        result = {"modelfile": "", "parameters": "", "template": "", "details": {"format": "safetensors"},
                  "capabilities": list(dict.fromkeys(["completion", *caps])), "model_info": {}}
        if reason.get("efforts"):
            result["thinking"] = {"values": [False if value == "none" else True if value == "on" else value
                                             for value in reason["efforts"]], "default": reason.get("default")}
        return result
    except Exception as e:  # noqa: BLE001
        return ollama_error(wrap(e))


@router.post("/api/pull")
async def pull(request: Request):
    return JSONResponse({"error": "Pull models from the KaggleUI control panel instead."}, status_code=400)

