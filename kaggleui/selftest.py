"""``kaggleui selftest``: deploy the smallest curated model per modality and exercise every API surface.

Writes a JSON report; exit code 0 only if every check passed. Runs on Kaggle (real engines) or locally
with ``--mock-engines --fake-hw t4x2``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import struct
import time
import wave

import httpx

from .config import load_preset
from .server.run import build_state, make_server

SMALL = {
    "image": {"width": 512, "height": 512, "steps": 2},
    "video": {"frames": 9, "steps": 4, "width": 480, "height": 272},
    "world": {"frames": 8, "steps": 4},
}


def _sine_wav(seconds: float = 2.0) -> bytes:
    sr = 16000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sr)))
                               for i in range(int(sr * seconds))))
    return buf.getvalue()


class Checker:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.checks: list[dict] = []

    async def check(self, name: str, coro) -> object:
        t0 = time.time()
        try:
            detail = await coro
            self.checks.append({"name": name, "ok": True, "detail": str(detail)[:300], "seconds": round(time.time() - t0, 2)})
            return detail
        except Exception as e:  # noqa: BLE001
            self.checks.append({"name": name, "ok": False, "detail": f"{type(e).__name__}: {e}"[:500],
                                "seconds": round(time.time() - t0, 2)})
            return None


async def _sse_text(client: httpx.AsyncClient, url: str, body: dict, headers: dict | None = None) -> str:
    text = ""
    async with client.stream("POST", url, json=body, headers=headers or {}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line.startswith("data:") and line.strip() != "data: [DONE]":
                try:
                    ev = json.loads(line[5:])
                except ValueError:
                    continue
                # Thinking models (e.g. qwen3) stream reasoning first; it counts as streamed output.
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    text += delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or ""
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta") or {}
                    text += delta.get("text") or delta.get("thinking") or ""
    return text


async def _text_checks(c: Checker, name: str) -> None:
    cl = c.client
    msgs = [{"role": "user", "content": "Reply with the single word: pong"}]

    async def oai():
        r = await cl.post("/v1/chat/completions", json={"model": name, "messages": msgs, "max_tokens": 256})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def oai_stream():
        t = await _sse_text(cl, "/v1/chat/completions", {"model": name, "messages": msgs, "stream": True, "max_tokens": 256})
        assert t, "empty stream"
        return t

    async def anth():
        r = await cl.post("/v1/messages", json={"model": name, "max_tokens": 256, "messages": msgs},
                          headers={"anthropic-version": "2023-06-01"})
        r.raise_for_status()
        data = r.json()
        assert data.get("type") == "message", data
        return "".join(b.get("text", "") for b in data["content"])

    async def anth_stream():
        t = await _sse_text(cl, "/v1/messages", {"model": name, "max_tokens": 256, "messages": msgs, "stream": True},
                            {"anthropic-version": "2023-06-01"})
        assert t, "empty stream"
        return t

    async def ollama():
        r = await cl.post("/api/chat", json={"model": name, "messages": msgs, "stream": False})
        r.raise_for_status()
        return r.json()["message"]["content"]

    async def models():
        r = await cl.get("/v1/models")
        r.raise_for_status()
        ids = [m["id"] for m in r.json()["data"]]
        assert name in ids, ids
        return ids

    await c.check("openai.models", models())
    await c.check("openai.chat", oai())
    await c.check("openai.chat.stream", oai_stream())
    await c.check("anthropic.messages", anth())
    await c.check("anthropic.messages.stream", anth_stream())
    await c.check("ollama.chat", ollama())


async def _job_checks(c: Checker, name: str, modality: str) -> None:
    cl = c.client

    async def run():
        body = {"model": name, "prompt": "a red fox running through snow", **SMALL[modality]}
        if modality == "world":
            # This check runs one model at a time. Without an image deployment,
            # world models must use their sample frame rather than a text prompt.
            body.pop("prompt")
            body["actions"] = [{"keys": ["forward"], "frames": 4}, {"keys": ["left"], "camera": [0.3, 0], "frames": 4}]
        r = await cl.post("/v1/videos", json=body)
        r.raise_for_status()
        vid = r.json()["id"]
        for _ in range(1800):
            s = (await cl.get(f"/v1/videos/{vid}")).json()
            if s.get("status") == "completed":
                break
            if s.get("status") == "failed":
                raise RuntimeError(s.get("error"))
            await asyncio.sleep(2)
        data = (await cl.get(f"/v1/videos/{vid}/content")).content
        assert len(data) > 1000, f"video too small ({len(data)} bytes)"
        return f"{len(data)} bytes"

    await c.check(f"openai.videos ({modality})", run())


async def _checks(c: Checker, modality: str, name: str) -> None:
    cl = c.client
    if modality == "text":
        await _text_checks(c, name)
    elif modality == "image":
        async def img():
            r = await cl.post("/v1/images/generations", json={"model": name, "prompt": "a lighthouse at dusk", "n": 1,
                                                                **SMALL["image"]})
            r.raise_for_status()
            png = base64.b64decode(r.json()["data"][0]["b64_json"])
            assert png[:4] == b"\x89PNG", "not a PNG"
            return f"{len(png)} bytes"

        await c.check("openai.images", img())
    elif modality == "tts":
        async def tts():
            r = await cl.post("/v1/audio/speech", json={"model": name, "input": "Hello from Kaggle UI.", "response_format": "wav"})
            r.raise_for_status()
            assert len(r.content) > 1000, "audio too small"
            return f"{len(r.content)} bytes {r.headers.get('content-type')}"

        await c.check("openai.audio.speech", tts())
    elif modality == "stt":
        async def stt():
            r = await cl.post("/v1/audio/transcriptions", data={"model": name, "response_format": "verbose_json"},
                              files={"file": ("tone.wav", _sine_wav(), "audio/wav")})
            r.raise_for_status()
            assert "text" in r.json()
            return r.json()

        await c.check("openai.audio.transcriptions", stt())
    elif modality in ("video", "world"):
        await _job_checks(c, name, modality)


async def _run(args) -> int:
    state = build_state(port=args.port, tunnel="none", preset=load_preset(args.preset), mock=args.mock_engines,
                        fake_hw=args.fake_hw, home=args.home)
    uv_server = make_server(state)
    server = asyncio.create_task(uv_server.serve())
    wanted = [m.strip() for m in args.modalities.split(",") if m.strip()]
    base = f"http://127.0.0.1:{args.port}"
    report: dict = {"started": time.time(), "mock": args.mock_engines, "results": []}
    exit_code = 0
    async with httpx.AsyncClient(base_url=base, timeout=httpx.Timeout(connect=10, read=1800, write=120, pool=10),
                                 headers={"Authorization": f"Bearer {state.auth.admin_token}"}) as admin:
        for _ in range(60):
            try:
                if (await admin.get("/_kui/api/session")).status_code == 200:
                    break
            except httpx.HTTPError:
                await asyncio.sleep(0.5)
        report["hardware"] = (await admin.get("/_kui/api/system")).json()
        api = httpx.AsyncClient(base_url=base, timeout=admin.timeout,
                                headers={"Authorization": f"Bearer {state.auth.api_key}"})
        state.settings.api.ollama = True
        specs = [s for s in state.catalog.selftest_specs() if s["modality"] in wanted]
        if "text" in wanted:
            # The native llama.cpp engine is a separate runner from Ollama; exercise it with a 0.6B GGUF.
            specs.append({"modality": "text", "options": {"ctx": 2048},
                          "model": {"source": "hf", "id": "unsloth/Qwen3-0.6B-GGUF", "variant": "Q4_K_M",
                                    "engine": "llamacpp"}})
        for spec in specs:
            spec["options"] = {**spec.get("options", {}), **SMALL.get(spec["modality"], {})}
            entry = {"modality": spec["modality"], "model": spec["model"]["id"], "engine": spec["model"]["engine"]}
            print(f"[selftest] deploying {entry['model']} ({entry['engine']})", flush=True)
            t0 = time.time()
            dep = (await admin.post("/_kui/api/deployments", json=spec)).json()
            status = dep
            while status.get("status") not in ("ready", "error"):
                await asyncio.sleep(2)
                status = next((d for d in (await admin.get("/_kui/api/deployments")).json() if d["id"] == dep["id"]), {})
                if not status:
                    break
            entry["deploy_seconds"] = round(time.time() - t0, 1)
            if status.get("status") != "ready":
                entry["error"] = status.get("error") or "deployment vanished"
                logs = (await admin.get(f"/_kui/api/deployments/{dep['id']}/logs")).json().get("lines", [])
                entry["log_tail"] = logs[-40:]
                exit_code = 1
            else:
                checker = Checker(api)
                await _checks(checker, spec["modality"], status["name"])
                entry["checks"] = checker.checks
                if not all(ch["ok"] for ch in checker.checks):
                    exit_code = 1
            await admin.delete(f"/_kui/api/deployments/{dep['id']}")
            report["results"].append(entry)
            ok = "OK" if not entry.get("error") and all(ch["ok"] for ch in entry.get("checks", [])) else "FAIL"
            print(f"[selftest] {ok} {entry['modality']}: {entry['model']} in {entry['deploy_seconds']}s", flush=True)
            for ch in entry.get("checks", []):
                print(f"    {'✓' if ch['ok'] else '✗'} {ch['name']} ({ch['seconds']}s) {'' if ch['ok'] else ch['detail']}",
                      flush=True)
            if entry.get("error"):
                print(f"    ✗ {entry['error']}", flush=True)
        await api.aclose()
    report["finished"] = time.time()
    report["passed"] = exit_code == 0
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[selftest] report written to {args.out}: {'PASSED' if exit_code == 0 else 'FAILED'}", flush=True)
    uv_server.should_exit = True
    try:
        await asyncio.wait_for(server, timeout=30)
    except (asyncio.TimeoutError, Exception):
        server.cancel()
    return exit_code


def run(args) -> int:
    return asyncio.run(_run(args))

